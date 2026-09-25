# SPDX-License-Identifier: Apache-2.0
"""glm-5.3-Flash (``glm5_next``) blockwise-FP8 modeling.

The stack is hybrid: 45 decoder layers, 11 carrying multi-head latent attention
(MLA) with the DSA sparse indexer and 34 carrying gated-delta linear attention
(KDA). Both halves of every layer are mixed through multi-hyper-connection
(mHC) residual streams, and layers at or past ``first_k_dense_replace`` run
routed blockwise-FP8 experts beside an always-on shared expert.

Parameters are declared with ``register_parameter(name, None)`` and given
storage only when the checkpoint is read, so building the tree allocates
nothing. The attribute paths and their checkpoint keys come from
``weight_loaders_fp8.build_weight_mappings``; a leaf is a flat
``<name>_weight`` on the parent module rather than a ``.weight`` on a child
``Linear``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FLOAT32_PLAIN_LEAVES,
    FP8_SCALE_SUFFIX,
    KDA_BARE_LEAVES,
    MAPPED_KEY_QUANTISED_WEIGHT,
    MAPPED_KEY_SCALE_GRID,
    MAPPED_KEY_STACKED_BANK,
    MHC_LEAVES,
    build_weight_mappings,
    classify_mapped_keys,
    compensate_block_scales,
    DeferredShardGeometry,
    dense_consumer_block_quant_size,
    dequantise_blockwise,
    loader_for_mapped_keys,
    report_floored_blocks,
    routed_bank_consumer_block_quant_size,
    scale_keys,
    sharded_scale_grid_loader,
    ShardGeometry,
    stacked_expert_scale_loader,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader

if TYPE_CHECKING:
    # Annotation only: this module imports no vllm symbol at runtime, so the
    # guard keeps the return type of ``_resolve_tp_group`` a real class name.
    from vllm.distributed.parallel_state import GroupCoordinator

# ---------------------------------------------------------------------------
# Parameter declaration
# ---------------------------------------------------------------------------


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    """True for any fp8 dtype the installed ``torch`` has."""
    return dtype.itemsize == 1 and "float8" in str(dtype)


def _is_on_device(where: torch.device, target: torch.device) -> bool:
    """Is a tensor sitting on the device this load targets?

    ``torch.device("cpu")`` and ``torch.device("cpu", 0)`` name the same place
    but compare unequal, so the kind must match and the index only has to match
    when the target names one.
    """
    if where.type != target.type:
        return False
    if target.index is None:
        return True
    return where.index == target.index


def _declare_parameters(module: nn.Module, *names: str) -> None:
    """Reserve parameter attribute paths on ``module`` without allocating.

    ``named_parameters()`` and ``state_dict()`` skip ``None`` entries, so the
    declared set is also recorded on ``declared_param_names``; that tuple is
    what :meth:`declared_parameter_names` walks.
    """
    for name in names:
        module.register_parameter(name, None)
    module.declared_param_names = (
        *getattr(module, "declared_param_names", ()),
        *names,
    )


#: Where a module records the checkpoint tensors it released once a load-time
#: prep had copied them into kernel orientation, ``{attribute: (shape, dtype)}``.
#: A plain attribute, so it enters neither ``state_dict()`` nor
#: ``named_parameters()``.
RELEASED_PARAMETERS_ATTR = "_released_checkpoint_parameters"


def _release_replaced_parameters(module: nn.Module, *names: str) -> int:
    """Drop the checkpoint tensors a prep copied into kernel orientation. Returns how many."""
    released = dict(getattr(module, RELEASED_PARAMETERS_ATTR, {}))
    dropped = 0
    for name in names:
        tensor = getattr(module, name, None)
        if tensor is None:
            continue
        released[name] = (tuple(tensor.shape), tensor.dtype)
        if name in module._parameters:
            module.register_parameter(name, None)
        else:
            setattr(module, name, None)
        dropped += 1
    setattr(module, RELEASED_PARAMETERS_ATTR, released)
    return dropped


# ---------------------------------------------------------------------------
# Relayout on the host, because the device refuses a strided copy
# ---------------------------------------------------------------------------


def _on_the_host(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The operands a load-time prep computes on, where a strided read is legal."""
    # A meta tensor is returned as it is: meta has no data to copy, so ``.cpu()``
    # raises on it, and a strided read is already legal there. The shape-only
    # rehearsal ``load_weights_lite`` runs on meta therefore needs no host copy.
    return tuple(tensor if tensor.is_meta else tensor.cpu() for tensor in tensors)


def _on_the_device(
    device: torch.device, *tensors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """The finished operands, moved to ``device`` in one step each."""
    # The other half of the pair above, so a prep reads as one hop out and one hop
    # back. For meta ``device`` every move is a no-op.
    return tuple(tensor.to(device) for tensor in tensors)


def _relaid_out_on_the_host(
    tensor: torch.Tensor,
    relayout: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``relayout(tensor)`` made dense, with every strided read taken on the host."""
    # Neuron tensors do not support ``.contiguous()``, so materialising a
    # transpose, a permute or a strided slice on the device raises
    # ``Expected self.is_contiguous() to be true, but got false``. ``tensor`` must
    # already be dense -- the move to the host is itself a copy and a strided one
    # refuses for the same reason -- so the strided step belongs inside
    # ``relayout``, where it runs on the host copy.
    (host,) = _on_the_host(tensor)
    (dense,) = _on_the_device(tensor.device, relayout(host).contiguous())
    return dense


def _transposed_on_the_host(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """``tensor`` with two axes swapped, dense, and never transposed on the device."""
    # The common shape of the rule above, spelled once so twelve call sites do not each
    # ``transpose`` rather than ``.t()`` because the stacked operand banks swap axes
    # 1 and 2 while the 2-D projections swap 0 and 1.
    return _relaid_out_on_the_host(tensor, lambda host: host.transpose(dim0, dim1))


# ---------------------------------------------------------------------------
# Geometry resolution for ``get_kv_spec``
# ---------------------------------------------------------------------------


def _resolve_world_size() -> int:
    """The rank count to divide head counts by, or 1 when not distributed.

    ``NeuronConfig`` carries no tensor-parallel degree field, so the process
    group is the only available source and its absence is not an error.
    """
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def _resolve_rank() -> int:
    """This process's rank in the tensor-parallel group, or 0 when not distributed.

    Read the same way as :func:`_resolve_world_size`, because the two values are
    read together and a pair that disagreed about what "not distributed" means
    would shard against a rank the world size does not contain.
    """
    try:
        return torch.distributed.get_rank()
    except (RuntimeError, ValueError):
        return 0


def _per_rank(count: int, world_size: int) -> int:
    """Per-rank head count, floored at 1."""
    return max(1, count // max(world_size, 1))


def _resolve_tp_group() -> GroupCoordinator | None:
    """The tensor-parallel group to reduce a row-parallel partial across.

    ``None`` means there is nothing to reduce: one rank, so the partial sum is
    already the whole sum. The classes here are built before any world size is
    known, so the group is resolved at the call site rather than bound in a
    constructor. The import is function-local and sits behind the single-rank
    guard because this module imports no vllm symbol at runtime.
    """
    if _resolve_world_size() <= 1:
        return None
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    return group if group.world_size > 1 else None


def _reduce_tp_rows(
    partials: torch.Tensor, *, num_requests: int = 1
) -> torch.Tensor:
    """Sum TP partials with concurrent requests on the contiguous minor axis.

    Interleaving matches singleton sums in the validated TP64 B2 control.
    Single-request decode and prefill keep their original in-place collective.
    The caller owns the final dtype cast.
    """
    if type(num_requests) is not int or num_requests < 1:
        raise ValueError("num_requests must be a positive plain integer")
    if num_requests > 1 and (
        partials.ndim != 2
        or int(partials.shape[0]) != num_requests
        or int(partials.shape[1]) < 1
        or partials.dtype != torch.float32
    ):
        raise ValueError(
            "concurrent TP reduction requires FP32 [num_requests, hidden] partials"
        )
    group = _resolve_tp_group()
    if group is None:
        return partials
    if num_requests == 1:
        group.all_reduce(partials)
        return partials
    # XLA can retain the original physical layout for transpose().contiguous().
    # Stack makes the request axis explicit in the collective's input layout.
    packed = torch.stack(
        tuple(partials[row].reshape(-1) for row in range(num_requests)), dim=1
    ).contiguous()
    group.all_reduce(packed)
    return packed.transpose(0, 1).contiguous()


# --------------------------------------------------------------------------- #
# which parameter families are sharded, and on which dim.
#
# The geometry is declared per class here rather than attached to a parameter
# object inside each ``__init__``: parameters are declared as
# ``register_parameter(name, None)`` and the objects are built only in
# ``_materialise_declared_parameters``, and of the ten declaring classes only
# ``Glm5NextKDAAttention`` and ``Glm5NextModel`` are told a world size. It is
# consumed at that one attachment site, where the root's world size lives.
#
# ``shard_dim`` is the dimension in the final parameter shape, which in this
# package is the checkpoint shape ``[out_features, in_features]`` -- the
# projection weights are transposed once by ``prepare_projection_weights``,
# never at load, so what a loader slices is the untransposed tensor. ``width``
# returns this rank's extent along that dim. Every family not named here is
# replicated and reaches ``loader_for_mapped_keys`` with ``geometry=None``.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _DeclaredShard:
    """One family's declared shard: the dim, and this rank's extent along it.

    ``width`` is ``None`` for a family whose extent arrives with the tensor
    rather than from config, and the loader holding the tensor resolves it.
    ``pad_to_consumer_block`` rounds the full width up to the smallest multiple
    of ``world_size * dense_consumer_block_quant_size()`` before dividing, so
    every rank's shard is a whole consumer block; it is a flag rather than the
    number because the number lives in the consumer and is imported at
    attachment time.
    """

    shard_dim: int
    width: Callable[[nn.Module, int], int] | None
    #: Why this family is sharded on this dim, in one clause.
    because: str
    pad_to_consumer_block: bool = False
    #: True for the routed expert bank alone. Its intermediate width is divided by
    #: ``tp_per_ep = world_size // ep_degree`` -- the ranks inside one
    #: expert-parallel group -- and not by the whole world, because the experts
    #: themselves are already divided across the groups.
    shards_within_expert_parallel_group: bool = False
    #: This family's per-rank shard must already be a whole consumer block, and the
    #: load refuses by name when it is not. An inadmissible width is either padded
    #: up to admissibility or refused, and the bank's answer is refuse, because its
    #: sharded axis carries per-expert structure that a pad would silently inflate.
    #: Mutually exclusive with the pad, enforced in
    #: ``DeferredShardGeometry.__post_init__`` rather than trusted here.
    require_consumer_block: bool = False


#: The value a pad row carries, and no activation on either dense path does.
#:
#: ``blockwise_fp8_mm`` tiles ``M`` over the PSUM partition axis and does not
#: pad, so it refuses any token count that is not a whole number of
#: ``TILE_SIZE`` rows -- a one-token decode step is exactly that case. Both
#: dense paths pad here and slice the result back inside the same call.
#:
#: ``-2 ** 15`` is exact in ``bfloat16`` and orders of magnitude outside the
#: widest pre-activation either dense path produces, so a pad row is
#: identifiable by equality rather than by a tolerance. It never reaches a
#: caller: the slice removes every pad row at the single return.
_TOKEN_PAD_SENTINEL = -32768.0


def _pad_tokens_to_tile(
    hidden_states: torch.Tensor, tile: int
) -> tuple[torch.Tensor, int]:
    """Grow ``[T, H]`` up to a whole tile of rows. Returns it with the caller's T.

    A count that is already a whole tile is returned unchanged and pays no copy.
    A count of zero or less is also returned unchanged: the kernel refuses it by
    name, and inventing rows for an empty call would replace a clear refusal
    with a silently different function.
    """
    tokens = int(hidden_states.shape[0])
    if tokens <= 0 or tokens % tile == 0:
        return hidden_states, tokens
    pad = hidden_states.new_full(
        (tile - tokens % tile, int(hidden_states.shape[1])), _TOKEN_PAD_SENTINEL
    )
    return torch.cat((hidden_states, pad), dim=0), tokens


def _unpad_rows(out: torch.Tensor, tokens: int) -> torch.Tensor:
    """Give the caller back its own rows. The pad never leaves the call."""
    if int(out.shape[0]) == tokens:
        return out
    return out[:tokens]


def _kda_head_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``heads * head_dim``, already divided by the world size.

    ``Glm5NextKDAAttention`` divides in its own constructor, so ``world_size``
    is accepted and deliberately unused here, which keeps the table's width
    signature uniform and makes dividing twice impossible to write by accident.
    """
    del world_size
    return int(module.num_kv_heads_per_rank) * int(module.head_dim)


def _kda_head_count(module: nn.Module, world_size: int) -> int:
    """This rank's head count, for the three families whose extent is H, not H*D."""
    del world_size
    return int(module.num_kv_heads_per_rank)


# The MLA families' three widths. ``Glm5NextMLAAttention`` keeps
# ``num_attention_heads`` as the model's head count at every world size -- the
# attribute is never overwritten with a per-rank value -- so the division
# happens here, unlike ``_kda_head_count``, which reads an attribute that is
# already per-rank. The class's own reader,
# :meth:`Glm5NextMLAAttention._heads_per_rank`, floors through the same
# ``_per_rank``, so the width a loader slices to and the width
# ``projection_widths`` expects are the same expression on the same two inputs.
#
# ``q_a_proj``, ``kv_a_proj_with_mqa`` and both latent layernorms take no entry
# below because they are replicated: they read or write the compressed latent,
# of which MLA keeps one per token rather than one per head, so there is no head
# axis to split.


def _mla_head_count(module: nn.Module, world_size: int) -> int:
    """This rank's MLA head count. The one divisor the three widths below share."""
    return _per_rank(int(module.num_attention_heads), world_size)


def _mla_q_b_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``q_b_proj`` output width: its heads at the full query width.

    The query head width is ``qk_nope_head_dim + qk_rope_head_dim``, summed
    rather than taken from the nope width alone: the rotary slice is 0 on this
    checkpoint, and a config that had one would otherwise be short by it.
    """
    return _mla_head_count(module, world_size) * (
        int(module.qk_nope_head_dim) + int(module.qk_rope_head_dim)
    )


def _mla_kv_b_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``kv_b_proj`` output width: its heads' key and value halves.

    Both halves are one weight and are sharded as one, because the absorb split
    cuts this same weight per head -- a rank holding the keys of its heads and
    the values of another's would mix heads at exactly the right total width.
    """
    return _mla_head_count(module, world_size) * (
        int(module.qk_nope_head_dim) + int(module.v_head_dim)
    )


def _mla_o_proj_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``o_proj`` input width: its heads' value width.

    Row-parallel, so each rank computes a partial sum over its own heads and
    :meth:`Glm5NextMLAAttention.project_output` reduces it across the group.
    """
    return _mla_head_count(module, world_size) * int(module.v_head_dim)


#: Declaring class name -> declared leaf -> its shard. Keyed by name rather than
#: by the class object so this table can sit above every class it names.
_SHARD_GEOMETRY: dict[str, dict[str, _DeclaredShard]] = {
    "Glm5NextKDAAttention": {
        # Column-parallel: each rank owns whole heads of the projection's output.
        **{
            leaf: _DeclaredShard(0, _kda_head_width, "column-parallel over whole heads")
            for leaf in (
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "f_b_proj_weight",
                "g_b_proj_weight",
                "q_conv1d_weight",
                "k_conv1d_weight",
                "v_conv1d_weight",
            )
        },
        # Row-parallel: the head width is o_proj's input, so the partial sum is
        # per-rank and the reduction is the consumer's business, not the loader's.
        "o_proj_weight": _DeclaredShard(
            1, _kda_head_width, "row-parallel -- the head width is its input"
        ),
        # One value per head, not per channel.
        "b_proj_weight": _DeclaredShard(0, _kda_head_count, "one row per head"),
        "A_log": _DeclaredShard(0, _kda_head_count, "one decay per head"),
        # One value per channel, not per head: the forward reshapes this bias flat
        # and takes ``[h * head_dim : (h + 1) * head_dim]`` from it per head, so the
        # extent it reads is the head width.
        "dt_bias": _DeclaredShard(
            0, _kda_head_width, "one bias per channel -- the forward slices it per head width"
        ),
    },
    # Three MLA leaves, one per projection whose declared width carries the head
    # count. The two latent projections are absent because they are replicated.
    "Glm5NextMLAAttention": {
        # Column-parallel: each rank owns whole heads of the projection's output.
        # ``q_b_proj`` expands the query latent into per-head queries, so its
        # output carries the head axis and dim 0 is that axis in the checkpoint
        # orientation ``[out_features, in_features]``.
        "q_b_proj_weight": _DeclaredShard(
            0, _mla_q_b_width, "column-parallel over whole heads"
        ),
        # Also column-parallel, and its head axis carries the key and value halves
        # together -- the absorb split cuts the same weight per head, so splitting
        # the two halves across ranks would mix heads at the right total width.
        "kv_b_proj_weight": _DeclaredShard(
            0, _mla_kv_b_width, "column-parallel over whole heads, keys with values"
        ),
        # Row-parallel: the head width is o_proj's input, so each rank computes a
        # partial sum and ``project_output`` reduces it across the group.
        "o_proj_weight": _DeclaredShard(
            1, _mla_o_proj_width, "row-parallel -- the head width is its input"
        ),
    },
    "Glm5NextDenseMLP": {
        "gate_proj_weight": _DeclaredShard(
            0, None, "column-parallel on the intermediate width", True
        ),
        "up_proj_weight": _DeclaredShard(
            0, None, "column-parallel on the intermediate width", True
        ),
        "down_proj_weight": _DeclaredShard(
            1, None, "row-parallel -- the intermediate width is its input", True
        ),
    },
    # The shared expert's three, on the same route as the dense three above and
    # for the stronger version of the same reason: this class holds no width to
    # divide. It is not an expert-parallel entity -- it runs on every token, so it
    # shards its intermediate width across the full world like the dense MLP, and
    # its consumer is the dense blockwise route.
    "Glm5NextSharedExperts": {
        "gate_proj_weight": _DeclaredShard(
            0, None, "column-parallel on the intermediate width", True
        ),
        "up_proj_weight": _DeclaredShard(
            0, None, "column-parallel on the intermediate width", True
        ),
        "down_proj_weight": _DeclaredShard(
            1, None, "row-parallel -- the intermediate width is its input", True
        ),
    },
    # The routed bank's three, and the one family whose divisor is not the world
    # size. Its experts are already divided across the expert-parallel groups, so
    # what remains to divide inside a group is the intermediate width, by
    # ``tp_per_ep``.
    #
    # ``shard_dim`` here is the per-expert checkpoint dim, not the stacked
    # parameter's. The bank's parameter carries a leading expert axis, so its dims
    # are ``[E, out, in]`` and the intermediate sits one place to the right of the
    # number below. The number is the checkpoint tensor's because that is what the
    # bank loader slices: it selects each expert's columns before stacking, so a
    # dim counted in the stacked shape would slice the wrong axis of every expert.
    # The flags are spelled as keywords so that dropping one cannot shift another
    # into its place.
    "Glm5NextRoutedExperts": {
        "gate_proj_weight": _DeclaredShard(
            0,
            None,
            "column-parallel on the intermediate width, inside the EP group",
            shards_within_expert_parallel_group=True,
            require_consumer_block=True,
        ),
        "up_proj_weight": _DeclaredShard(
            0,
            None,
            "column-parallel on the intermediate width, inside the EP group",
            shards_within_expert_parallel_group=True,
            require_consumer_block=True,
        ),
        "down_proj_weight": _DeclaredShard(
            1,
            None,
            "row-parallel -- the intermediate width is its input, inside the EP group",
            shards_within_expert_parallel_group=True,
            require_consumer_block=True,
        ),
    },
}


def _shard_geometry_for(
    module: nn.Module, leaf: str, world_size: int
) -> ShardGeometry | DeferredShardGeometry | None:
    """This parameter's resolved shard, or ``None`` when it is replicated.

    The single reader of :data:`_SHARD_GEOMETRY`, called from
    ``_materialise_declared_parameters`` for every declared parameter. At world
    size 1 everything is replicated, because a one-rank shard is the whole
    tensor.

    A family that declares a width function gets its extent resolved here; a
    family declaring ``width=None`` gets a :class:`DeferredShardGeometry`
    carrying the dim, the rank count and the pad multiple, and leaves the extent
    to the loader that will hold the tensor. The routed bank divides by
    ``tp_per_ep = world_size // ep_degree`` rather than by the world, and
    answers ``None`` when that is 1 -- one rank per group, so every member holds
    its experts whole.

    A scale grid answers with its weight's geometry: a blockwise grid holds one
    value per weight tile, so it is sharded if and only if its weight is, on the
    same dimension. The table therefore keeps one row per weight.
    """
    if world_size <= 1:
        return None
    family = _SHARD_GEOMETRY.get(type(module).__name__, {})
    declared = family.get(leaf)
    if declared is None and leaf.endswith(f"_{FP8_SCALE_SUFFIX}"):
        # ``q_b_proj_weight_scale_inv`` -> ``q_b_proj_weight``, the inverse of
        # ``_sibling_scale_grid_name``. Only a grid whose weight the table names
        # resolves; every other declared grid still answers ``None``.
        weight_leaf = (
            f"{leaf[: -len(f'_{FP8_SCALE_SUFFIX}')]}{_WEIGHT_LEAF_SUFFIX}"
        )
        declared = family.get(weight_leaf)
    if declared is None:
        return None
    num_shards = world_size
    # Declared before the branch because the geometry carries it: the EP-TP group
    # the column reader consults only exists above 1, so a bank geometry that did
    # not say which degree it was built at would make the reader treat degree 1 as
    # a missing group.
    ep_degree = 1
    if declared.shards_within_expert_parallel_group:
        # The divisor is the ranks inside one expert-parallel group, read off the
        # module that already resolved both degrees at construction, so the two
        # halves cannot disagree with the partition the same object built.
        ep_degree = max(1, int(getattr(module, "ep_degree", 1)))
        num_shards = max(1, world_size // ep_degree)
        if num_shards <= 1:
            # One rank per expert-parallel group: every group member holds its
            # experts whole, so there is no intermediate shard and the replicated
            # path is the right one.
            return None
    if declared.width is None:
        # The two flags mean opposite things about an inadmissible width -- pad it
        # up, or refuse it -- and they read two different numbers, each from the
        # producer that enforces it. The pad belongs to the dense MLP and the shared
        # expert, whose weights ``blockwise_fp8_mm`` dequantises at
        # ``SCALE_BLOCK_SIZE``; the requirement belongs to the routed bank alone,
        # whose scale operands are built at ``BLOCK_QUANT_SIZE``. Both are imported,
        # never typed here.
        return DeferredShardGeometry(
            shard_dim=declared.shard_dim,
            num_shards=num_shards,
            pad_to_multiple_of=(
                dense_consumer_block_quant_size()
                if declared.pad_to_consumer_block
                else None
            ),
            expert_parallel_degree=ep_degree,
            require_multiple_of=(
                routed_bank_consumer_block_quant_size()
                if declared.require_consumer_block
                else None
            ),
        )
    if declared.shards_within_expert_parallel_group:
        # No family declares both, and this refusal keeps it that way: a resolved
        # width is divided by the world size below, so a family that also asked for
        # the expert-parallel divisor would get one divisor in its extent and
        # another in its rank count, and shard itself wrong.
        raise ValueError(
            f"{type(module).__name__}.{leaf} declares both a width function and "
            f"the expert-parallel divisor; the width would be divided by the world "
            f"size while the shard count came from tp_per_ep, so the two would "
            f"disagree. Declare width=None for an expert-parallel family."
        )
    return ShardGeometry(
        shard_dim=declared.shard_dim,
        shard_size=declared.width(module, world_size),
        num_shards=world_size,
    )


def _linear_attn_field(text_config: Glm5NextTextConfig, key: str) -> int:
    """Read one required ``linear_attn_config`` entry, loudly.

    A missing key is raised rather than defaulted: silently substituting a head
    count would make the KDA half of the KV spec wrong in a way no downstream
    shape assertion could attribute back to here.
    """
    config = text_config.linear_attn_config or {}
    if key not in config:
        raise ValueError(
            f"text_config.linear_attn_config is missing {key!r}; the KDA cache "
            f"geometry cannot be resolved (present keys: {sorted(config)})"
        )
    return int(config[key])


def _resolve_model_dtype(text_config: Glm5NextTextConfig) -> torch.dtype:
    """The stack's compute dtype, which is also the MLA latent cache dtype.

    The MLA latent is not a quantised cache: this checkpoint's
    ``quantization_config`` declares no ``kv_cache_quant_algo``, and the
    blockwise weight scheme is rejected for KV caches.
    """
    dtype = text_config.torch_dtype
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"text_config.torch_dtype must be a torch.dtype, got {dtype!r}; "
            "the KV cache dtype cannot be resolved"
        )
    return dtype


def _resolve_kda_state_dtype(text_config: Glm5NextTextConfig) -> torch.dtype:
    """dtype for the KDA recurrent-state buffers.

    ``NeuronConfig.kda_state_dtype`` is a torch dtype *name* (e.g.
    ``"bfloat16"``) so it survives a JSON ``additional_config`` round-trip, and
    ``None`` follows the model's own dtype. A name that does not resolve is
    raised rather than passed through, because a ``str`` reaching
    ``LayerSpec.dtype`` would allocate the cache wrong. The result is never
    ``None``.
    """
    neuron_config = text_config.neuron_config
    override = getattr(neuron_config, "kda_state_dtype", None)
    if override is None:
        return _resolve_model_dtype(text_config)
    if isinstance(override, torch.dtype):
        return override
    resolved = getattr(torch, str(override), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(
            f"NeuronConfig.kda_state_dtype={override!r} does not name a torch "
            "dtype; it is declared as a dtype NAME such as 'bfloat16'"
        )
    return resolved


def _resolve_mla_head_size(text_config: Glm5NextTextConfig) -> int:
    """MLA latent cache width: ``kv_lora_rank + qk_rope_head_dim``.

    ``qk_rope_head_dim`` is 0 on this checkpoint because ``mla_use_nope`` is
    set, so the sum is ``kv_lora_rank == 512``. The sum is used rather than the
    bare rank because the cached latent is the compressed KV vector
    concatenated with whatever rotary slice exists.
    """
    return int(text_config.kv_lora_rank) + int(text_config.qk_rope_head_dim)


# ---------------------------------------------------------------------------
# ``Glm5NextQuantConfig`` / quant-method selection
#
# ``quantization.py`` parses the checkpoint's ``quantization_config`` into a
# ``QuantizationSpec`` carrying ``weight_block_size (128, 128)``. This class is
# what turns that spec into a method, so a call site can ask what to run for a
# given module. The vendor quantisation enum has no blockwise member today, so
# the route is a direct inner-kernel call and this class carries a
# plugin-side method object.
#
# Imports are function-local, which is the file family's own idiom --
# ``llama3/quantization.py`` imports its modeling module inside
# ``resolve_attention_mlp_classes`` for the same reason.
# ---------------------------------------------------------------------------


class Glm5NextQuantConfig:
    """Per-module blockwise-FP8 quantisation policy for this arch.

    Holds the parsed
    :class:`~vllm_neuron.model.glm5_next.quantization.QuantizationSpec` and the
    method resolved from it, and answers the one question a modeling call site
    has: what do I run for this module?

    Attributes:
        spec: The parsed spec, or ``None`` for an unquantized checkpoint.
        method: The method resolved for the model-wide scheme, or ``None`` when
            nothing is quantized. Per-module resolution goes through
            :meth:`get_quant_method`, which is the form that survives mixed
            precision.

    Construction raises rather than degrading when the checkpoint declares a
    block shape this build has no authored path for -- see
    :data:`~vllm_neuron.model.glm5_next.quantization.SUPPORTED_WEIGHT_BLOCK_SIZES`.
    """

    def __init__(self, spec: object | None) -> None:
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        self.spec = spec
        self.method = resolve_quant_method(spec)

    @classmethod
    def from_model_config(cls, config: Glm5NextConfig) -> Glm5NextQuantConfig:
        """Build from the config the model itself is built from.

        The checkpoint's ``quantization_config`` is lifted onto
        :class:`Glm5NextConfig` by ``config.py``, parsed into a spec by
        :meth:`QuantizationSpec.from_model_config`, and resolved to a method
        here. Nothing in the chain is hand-fed.
        """
        from vllm_neuron.model.glm5_next.quantization import QuantizationSpec

        return cls(QuantizationSpec.from_model_config(config))

    def get_quant_method(
        self,
        layer_index: int | None = None,
        prefix: str = "",
    ) -> object | None:
        """Return the method for the module at ``(layer_index, prefix)``.

        ``None`` means that module is not quantized, so the unquantized path
        runs. Both arguments are forwarded to
        :meth:`QuantizationSpec.get_scheme`, which is where per-layer dispatch
        would land.
        """
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        return resolve_quant_method(self.spec, layer_index, prefix)

    @property
    def is_block_quantized(self) -> bool:
        """True when a blockwise-FP8 method was resolved."""
        return self.method is not None

    @property
    def block_shape(self) -> tuple[int, int] | None:
        """``(block_h, block_w)`` of the resolved method, or ``None``.

        Delegates rather than storing a copy: the method is the authority for the
        shape, and a copy is a second thing that can go stale.
        """
        return None if self.method is None else self.method.block_shape


# ---------------------------------------------------------------------------
# ``Glm5NextHyperConnection`` -- mHC wiring
# ---------------------------------------------------------------------------


class Glm5NextHyperConnectionError(ValueError):
    """A rank, extent or configuration this layer refuses, named not coerced.

    Only the cross-argument agreements the two kernels cannot see are checked
    here. Each kernel already refuses its own extents by name, and restating
    those bounds would create a second authority that can drift from the first.
    """


class Glm5NextHyperConnection(nn.Module):
    """Multi-hyper-connection (mHC) residual mixing, one layer call.

    The arithmetic is the pinned base's own ``MHCPreOp`` / ``MHCPostOp`` pair
    (``vllm/model_executor/layers/mhc.py`` at ``vllm==0.24.0``), with the two
    device-side pieces routed to this fork's NKI kernels:

    1. **pre** -- project the ``hc_mult`` residual streams through ``fn``, RMS
       scale, and split the result into three heads: ``pre_mix`` (which folds
       the streams into the sub-block's single input), ``post_mix``, and the
       ``[S, S]`` per-token stream-mixing matrix, which is then
       Sinkhorn-normalised by ``sinkhorn_normalise_blocks``.
    2. The caller's sub-block runs on the folded input.
    3. **post** -- mix the streams back out through
       ``hyper_connection_combine``.

    Both kernels take fp32 and return fp32, so this layer computes in fp32
    across them and casts back to the streams' own dtype at each return.

    Two differences from the base live inside the Sinkhorn kernel and cannot be
    removed here: it adds an inert ``1e-30`` to each denominator rather than
    ``hc_sinkhorn_eps``, and its schedule is ``R`` row/column pairs starting
    with a row pass rather than ``softmax``, one column pass and ``R-1`` pairs.
    Sinkhorn-Knopp has one fixed point, so at the target's 20 iterations the two
    agree to well inside this layer's tolerance.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        neuron_config: NeuronConfig | None = None,
        post_mult_value: float = 2.0,
    ) -> None:
        """Size the layer from the checkpoint's own dials.

        Args:
            text_config: carries ``hc_mult``, ``hc_sinkhorn_iters``, ``hc_eps``
                and ``rms_norm_eps``.
            neuron_config: framework overrides. ``mhc_sinkhorn_iters`` and
                ``mhc_eps`` win when not ``None``.
            post_mult_value: the multiplier on the post gate. ``2.0`` is the
                target model's own number -- it computes
                ``post = 2 * sigmoid(post_w * post_scale + post_b)``, with the
                range ``[0, 2]`` named in its shape guide. No fork config field
                carries the multiplier, so it stays a constructor argument
                rather than an invented config default.

        Two epsilons, each at the site the target model places it.
        ``hc_pre_eps`` and ``hc_sinkhorn_eps`` are mHC-native and both read
        ``text_config.hc_eps`` (``1e-06``): the target adds it after the pre
        sigmoid and after the comb softmax, the two sites :meth:`mhc_pre` adds
        it at. ``rms_eps`` is the RMSNorm epsilon and reads
        ``text_config.rms_norm_eps`` (``1e-05``), a different number on the same
        config object. It reaches exactly one mHC line, the RMS denominator in
        :meth:`mhc_pre`, because the target normalises the folded input through
        its own unweighted RMSNorm before the projection. So
        ``neuron_config.mhc_eps`` overrides the pre and comb sites but not the
        RMS denominator, and no ``mhc_rms_eps`` override exists.

        Raises:
            Glm5NextHyperConnectionError: on a non-positive ``hc_mult``,
                ``hidden_size`` or iteration count.
        """
        super().__init__()
        hc_mult = int(text_config.hc_mult)
        hidden = int(text_config.hidden_size)
        iters = int(text_config.hc_sinkhorn_iters)
        eps = float(text_config.hc_eps)
        if neuron_config is not None:
            if neuron_config.mhc_sinkhorn_iters is not None:
                iters = int(neuron_config.mhc_sinkhorn_iters)
            if neuron_config.mhc_eps is not None:
                eps = float(neuron_config.mhc_eps)

        problems: list[str] = []
        if hc_mult <= 0:
            problems.append(f"hc_mult={hc_mult} must be positive")
        if hidden <= 0:
            problems.append(f"hidden_size={hidden} must be positive")
        if iters <= 0:
            problems.append(
                f"sinkhorn iterations={iters} must be positive; the checkpoint "
                f"declares hc_sinkhorn_iters={text_config.hc_sinkhorn_iters}"
            )
        if problems:
            raise Glm5NextHyperConnectionError(
                "mHC layer refuses this configuration: " + "; ".join(problems)
            )

        self.hc_mult = hc_mult
        self.hidden_size = hidden
        self.sinkhorn_iters = iters
        self.hc_eps = eps
        # The RMSNorm epsilon, at the one mHC site the target model puts it:
        # ``mhc_pre``'s RMS denominator. Read off the same config object as
        # ``hc_eps`` and deliberately not overridable by ``neuron_config.mhc_eps``,
        # because it is the model's RMSNorm constant rather than an mHC dial.
        self.rms_eps = float(text_config.rms_norm_eps)
        self.post_mult_value = float(post_mult_value)

        # ``hc_mult3`` is the base's own name for the projection's output width:
        # ``hc_mult`` pre weights + ``hc_mult`` post weights + ``hc_mult ** 2``
        # mixing weights, in that order, which is the order the three heads are
        # sliced out of ``mixes`` below.
        self.hc_mult3 = 2 * hc_mult + hc_mult * hc_mult

        # Ordinary parameters rather than ``_declare_parameters``' reservations.
        # The six mHC leaves the weight map emits are reserved flat on
        # ``Glm5NextKDALayer`` / ``Glm5NextDSALayer``, and
        # :func:`_bind_hyper_connection_sites` assigns three of them into each
        # instance by ``.data``; the two instances are held in a plain dict rather
        # than registered as submodules. So the three names below are this class's
        # own signature -- the base's ``mhc_pre`` argument names -- and not map
        # names.
        #
        # ``requires_grad=False`` because serving never differentiates and
        # ``.data`` assignment does not carry the flag over, so a default
        # ``nn.Parameter`` would leave one storage reachable both as a grad-free
        # leaf and as a grad-requiring parameter, which graph extraction refuses.
        self.fn = nn.Parameter(
            torch.zeros(self.hc_mult3, hc_mult * hidden, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_scale = nn.Parameter(
            torch.zeros(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_base = nn.Parameter(
            torch.zeros(self.hc_mult3, dtype=torch.float32), requires_grad=False
        )

    # ── mHC pre: the folded input, and one Sinkhorn call ──────────────────
    def mhc_pre(
        self, residual: torch.Tensor, *, num_requests: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The base's ``mhc_pre``, preserving singleton arithmetic for decode.

        Args:
            residual: ``[T, S, H]`` -- the ``S = hc_mult`` residual streams.
            num_requests: concurrent decode requests, with one row per request.
                Each request uses the original preparation and Sinkhorn call.
                The default retains the original batched path for every token count.

        Returns:
            ``(post_mix, comb_mix, layer_input)`` -- ``[T, S, 1]``,
            ``[T, S, S]`` and ``[T, H]``. ``post_mix`` and ``comb_mix`` are
            fp32; ``layer_input`` is the streams' own dtype, the reference's own
            form, so a bfloat16 carrier is handed bfloat16 and an fp32 fixture
            still gets fp32. ``comb_mix[t, i, j]`` weights input stream ``i``
            into output stream ``j``, the base's convention and the one the
            combine kernel reads.

        Nothing here bounds the token count: the Sinkhorn is called on
        ``[T, S, S]`` blocks and both kernels walk the token axis in tiles of
        ``nl.tile_size.pmax``. The bounds that remain are
        ``block <= PARTITION_MAX`` and ``cols <= MOVING_FMAX`` (512), and both
        are on ``S``, which is 4 here.

        Raises:
            Glm5NextHyperConnectionError: on a non-3-D ``residual`` or a stream
                or hidden extent that contradicts this layer's configuration.
            SinkhornError: from the kernel, on a non-3-D block input, a
                non-square block, or an unavailable NKI route -- there is no
                torch path, so an absent route raises rather than falling back.
                Propagated, never caught.
        """
        from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise_blocks

        tokens, streams, hidden = self._require_streams(residual)
        if type(num_requests) is not int or num_requests < 1:
            raise Glm5NextHyperConnectionError(
                "num_requests must be a positive integer"
            )
        if num_requests > 1 and num_requests != tokens:
            raise Glm5NextHyperConnectionError(
                "concurrent decode requires one stream row per request; "
                f"got {tokens} rows for {num_requests} requests"
            )
        if num_requests > 1:
            # Preserve the complete singleton preparation, including elementwise
            # operations whose rounding can change with the batch geometry.
            rows = [
                self.mhc_pre(residual[row : row + 1])
                for row in range(num_requests)
            ]
            layer_inputs = [row[2] for row in rows]
            if residual.dtype == torch.bfloat16:
                # Join in FP32 so the compiled norm retains singleton rounding.
                # Restore the existing BF16 contract before the sublayer.
                layer_inputs = [value.to(torch.float32) for value in layer_inputs]
            return tuple(
                torch.cat([row[field] for row in rows], dim=0)
                for field in range(2)
            ) + (torch.cat(layer_inputs, dim=0).to(residual.dtype),)

        flat = residual.reshape(tokens, streams * hidden).to(torch.float32)
        mixes = flat @ self.fn.to(torch.float32).t()
        # The RMS scale. Both upstream spellings divide the squared sum by the
        # projection's own input width -- ``hc_mult * hidden_size`` and
        # ``fn.shape[-1]`` -- and those are the same number. The epsilon is the
        # model's RMSNorm epsilon, not the mHC one: the target builds this norm
        # with ``rms_norm_eps`` and applies it to the folded input before the
        # projection. Scaling ``mixes`` after the matmul is the same result as
        # normalising ``flat`` before it, because the projection carries no bias
        # and is therefore homogeneous.
        sqrsum = flat.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / float(streams * hidden) + self.rms_eps)

        scale = self.hc_scale.to(torch.float32)
        base = self.hc_base.to(torch.float32)
        # ``+ hc_eps`` on the pre gate and nothing on the post gate, which is the
        # target's own asymmetry rather than an omission here:
        # ``pre = sigmoid(...) + hc_eps`` against ``post = 2 * sigmoid(...)``.
        pre_mix = (
            torch.sigmoid(mixes[:, :streams] * scale[0] + base[:streams])
            + self.hc_eps
        )
        # The multiply is written after the sigmoid rather than before it, which is
        # the same number in ieee-754 and keeps the value a settable argument.
        post_mix = (
            torch.sigmoid(
                mixes[:, streams : 2 * streams] * scale[1]
                + base[streams : 2 * streams]
            )
            * self.post_mult_value
        )
        comb_logits = mixes[:, 2 * streams :].reshape(
            tokens, streams, streams
        ) * scale[2] + base[2 * streams :].reshape(1, streams, streams)
        # ``softmax`` and the ``+ eps`` are the base's, and the target agrees line
        # for line, so this is the second of the two sites ``hc_eps`` belongs at.
        # They sit outside the kernel, which starts from an affinity matrix; this
        # is elementwise glue.
        comb_start = torch.softmax(comb_logits, dim=-1) + self.hc_eps

        # ``comb_start`` is already ``[T, S, S]``, the kernel's own input shape, so
        # this is a call rather than a translation. The blocks cost ``T * S * S``
        # values where embedding them down the diagonal of a ``[T*S, T*S]`` matrix
        # would cost ``(T*S)^2`` -- 128 KB of fp32 at 2048 tokens against 256 MB --
        # and the off-diagonal carries no information, since a zero stays zero
        # under row and column rescaling. There is no torch path: an absent NKI
        # route raises rather than quietly normalising 2048 tokens in torch.
        comb_mix = sinkhorn_normalise_blocks(
            comb_start, iters=self.sinkhorn_iters
        )

        # The collapse is computed in fp32 and returned in the streams' dtype,
        # which is the reference's own form. The cast is load-bearing rather than
        # cosmetic: this value is what every sublayer is handed, and the dense and
        # MoE kernels downstream state their contract as bfloat16 activations, so
        # an fp32 collapse would reach a kernel that loads x at its own dtype with
        # no gate to catch it. It preserves the dtype rather than naming bfloat16,
        # again because that is what the reference does.
        layer_input = (pre_mix.unsqueeze(-1) * residual.to(torch.float32)).sum(dim=1)
        return (
            post_mix.reshape(tokens, streams, 1),
            comb_mix,
            layer_input.to(residual.dtype),
        )

    # ── mHC post: one combine call ────────────────────────────────────────
    def mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """The base's ``mhc_post``, on the combine kernel.

        Args:
            x: ``[T, H]`` -- the sub-block's single-stream output.
            residual: ``[T, S, H]`` -- the streams, unchanged from
                :meth:`mhc_pre`'s input.
            post_layer_mix: ``[T, S, 1]`` from :meth:`mhc_pre`.
            comb_res_mix: ``[T, S, S]`` from :meth:`mhc_pre`.

        Returns:
            ``[T, S, H]`` in ``residual``'s dtype. The kernel computes in fp32
            and the cast back is here, which is the reference's form. This value
            is the carrier between layers, so its dtype is the dtype the next
            sublayer is handed.

        Raises:
            HyperConnectionError: from the kernel, on any inadmissible rank or
                extent. Propagated, never caught: a geometry the kernel cannot
                serve must not quietly reach a torch path.
        """
        from vllm_neuron.functional.mhc.hyper_connection import (
            hyper_connection_combine,
        )

        # Argument names and order are the kernel's, which are the base's, so this
        # is a call rather than a translation. The kernel takes fp32 and computes
        # in fp32; only the return is cast back to the carrier's dtype.
        mixed = hyper_connection_combine(
            x=x.to(torch.float32),
            residual=residual.to(torch.float32),
            post_layer_mix=post_layer_mix.to(torch.float32),
            comb_res_mix=comb_res_mix.to(torch.float32),
        )
        return mixed.to(residual.dtype)

    # ── one layer call ────────────────────────────────────────────────────
    def forward(
        self, residual: torch.Tensor, sublayer: object, *, num_requests: int = 1
    ) -> torch.Tensor:
        """One mHC layer call: pre, then the sub-block, then post.

        Args:
            residual: ``[T, S, H]`` residual streams.
            sublayer: the wrapped sub-block, a callable ``[T, H] -> [T, H]``.
                Annotated ``object`` rather than ``Callable`` because every
                import in this section is function-local.
            num_requests: concurrent decode requests, forwarded to ``mhc_pre``.

        Returns:
            ``[T, S, H]`` in the streams' own dtype -- the re-mixed streams,
            whatever :meth:`mhc_post` returns.

        Raises:
            Glm5NextHyperConnectionError: if ``sublayer`` is not callable, or if
                its output does not have the ``[T, H]`` shape the combine needs.
        """
        if not callable(sublayer):
            raise Glm5NextHyperConnectionError(
                f"sublayer must be a callable [T, H] -> [T, H], got "
                f"{type(sublayer).__name__}"
            )
        if type(num_requests) is not int or num_requests != 1:
            post_mix, comb_mix, layer_input = self.mhc_pre(
                residual, num_requests=num_requests
            )
        else:
            post_mix, comb_mix, layer_input = self.mhc_pre(residual)
        x = sublayer(layer_input)
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != tuple(
            layer_input.shape
        ):
            got = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
            raise Glm5NextHyperConnectionError(
                f"sublayer returned {got}, expected the [T, H] shape it was "
                f"given, {tuple(layer_input.shape)} -- the combine reads x and "
                f"the streams on the same token and hidden extents"
            )
        return self.mhc_post(x, residual, post_mix, comb_mix)

    # ── layout helpers ────────────────────────────────────────────────────
    def _require_streams(self, residual: torch.Tensor) -> tuple[int, int, int]:
        """``(T, S, H)``, once the cross-argument agreements hold.

        Only what the kernels cannot see: they read their extents off the tensors
        they are handed, so neither can tell that a stream or hidden extent
        disagrees with the config this layer was built from -- and a wrong stream
        count would silently mis-slice ``mixes`` instead of failing.
        """
        if residual.dim() != 3:
            raise Glm5NextHyperConnectionError(
                f"residual must be 3-D [T, S, H], got shape "
                f"{tuple(residual.shape)}"
            )
        tokens, streams, hidden = (int(v) for v in residual.shape)
        if streams != self.hc_mult:
            raise Glm5NextHyperConnectionError(
                f"residual carries S={streams} streams and this layer was built "
                f"for hc_mult={self.hc_mult}; the projection's output width "
                f"{self.hc_mult3} is sliced into three heads by that number, so "
                f"a mismatch would mis-slice rather than fail"
            )
        if hidden != self.hidden_size:
            raise Glm5NextHyperConnectionError(
                f"residual carries H={hidden} and this layer was built for "
                f"hidden_size={self.hidden_size}; fn is "
                f"[{self.hc_mult3}, {self.hc_mult * self.hidden_size}]"
            )
        return tokens, streams, hidden


# ---------------------------------------------------------------------------
# ``Glm5NextMoEBlock`` and its two expert containers
# ---------------------------------------------------------------------------


class Glm5NextRoutedExperts(nn.Module):
    """The routed-expert bank at ``mlp.experts``.

    One parameter per projection covers all ``n_routed_experts`` experts: the
    weight map sends ``mlp.experts.<leaf>_weight`` to a *list* of
    ``n_routed_experts`` checkpoint keys, because this checkpoint stores one
    tensor per expert while the fork's parameter side is per-projection. The
    router lives here too, and carries a bias because
    ``topk_method == "noaux_tc"``.
    """

    # An expert bank divides by the expert-parallel degree, never by the
    # tensor-parallel one, which shards the intermediate dimension instead. With
    # expert parallelism off the degree is 1, all ``n_routed_experts`` are local
    # on every rank, and nothing raises at any tensor-parallel width. The
    # raggedness gate below refuses a non-uniform partition by name rather than
    # floor-dividing silently.
    #
    # The partition arithmetic lives in ``factory.py`` and is imported
    # function-local, the file family's own idiom, because nothing on the
    # arch-lookup path pulls the modeling module in.

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        world_size: int | None = None,
        ep_degree: int | None = None,
    ) -> None:
        super().__init__()
        from vllm_neuron.model.glm5_next.factory import (
            _resolve_ep_degree,
            require_uniform_expert_partition,
        )

        self.num_routed_experts = int(text_config.n_routed_experts)
        self.num_experts_per_tok = int(text_config.num_experts_per_tok)
        # An explicit ``world_size`` is what a caller with a degree supplies;
        # ``None`` means read the process group, which is 1 when this stack is built
        # undistributed. ``tp_degree`` is the tensor-parallel world size, which is
        # what shards the intermediate dimension.
        self.tp_degree = (
            _resolve_world_size() if world_size is None else int(world_size)
        )
        # The divisor for the experts themselves.
        self.ep_degree = _resolve_ep_degree(ep_degree)
        self.expert_partition = require_uniform_expert_partition(
            self.num_routed_experts, self.ep_degree
        )
        # Uniform by the gate above, so rank 0's count is every rank's count.
        self.num_local_experts = self.expert_partition.counts[0]
        # The checkpoint's SwiGLU bound. The reference clamps the routed bank and
        # not only the MLP: it clamps ``gate`` from above and ``up`` on both sides
        # before ``F.silu(gate) * up``, so an unclamped kernel computes a different
        # function on every token whose projection leaves the box.
        #
        # It refuses rather than defaulting. A literal here would be a bound this
        # code invented, and a ``None`` reaching the kernel is the silent omission
        # itself -- all four of the kernel's limit parameters default to ``None``.
        # The read is at construction, like the shared expert's, so no call site
        # can hand this path a bound the checkpoint never declared.
        limit = getattr(text_config, "swiglu_limit", None)
        if limit is None:
            raise Glm5NextBlockQuantRouteError(
                "text_config carries no swiglu_limit, so the routed bank has no "
                "checkpoint bound to clamp its SwiGLU with. Refusing to build: "
                "running unclamped is what computed a different function from "
                "the reference, and defaulting would invent a bound."
            )
        self.swiglu_limit = float(limit)
        _declare_parameters(
            self,
            "router_weight",
            "router_bias",
            "gate_proj_weight",
            "up_proj_weight",
            "down_proj_weight",
        )

    def local_expert_indices(self, rank: int) -> tuple[int, ...]:
        """The global expert indices ``rank`` owns, ascending.

        Delegates to the partition rather than recomputing the arithmetic, so there is
        one partition and not two that can disagree.
        """
        return self.expert_partition.local_expert_indices(rank)

    # The routing hyperparameters live on ``Glm5NextTextConfig`` and are threaded
    # in at the call rather than retained as state.
    #
    # ``route_tokens`` returns global expert indices over all
    # ``n_routed_experts``, exactly as the checkpoint's router does. Mapping those
    # onto this rank's ``expert_partition`` slice is the dispatch step, and it
    # belongs to :meth:`block_quant_expert_mm`.

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        gamma: torch.Tensor,
        text_config: Glm5NextTextConfig,
        eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route ``[B, S, H]`` tokens to the top-``k`` experts with ``noaux_tc``.

        Args:
            hidden_states: ``[B, S, H]`` pre-norm decoder activations.
            gamma: ``[H]`` or ``[1, H]`` router RMSNorm weights.
            text_config: the decoder config the routing hyperparameters live on.
            eps: RMSNorm epsilon. ``None``, the default, resolves it from
                ``text_config.rms_norm_eps`` -- the checkpoint's ``1e-05``. Pass
                a float to override it.

        Returns:
            ``(router_logits [T, E], expert_index [T, k] int32,
            expert_affinities [T, E] float32)``. ``expert_affinities`` is the
            scattered form the downstream MoE consumes: the gate weight at each
            selected expert's column, zero elsewhere.

        ``correction_bias`` is ``mlp.gate.e_score_correction_bias``, the
        ``noaux_tc`` correction bias and not a router projection bias. The
        kernel's signature keeps the two apart by name, because adding this
        tensor to the logits instead of to the sigmoid scores would compute a
        different router that no shape check could catch.
        """
        # The config's epsilon is the operative one unless a caller overrides it.
        # Resolved here rather than as a signature default, which would hard-wire
        # one number that belongs to the checkpoint.
        if eps is None:
            eps = float(text_config.rms_norm_eps)

        from vllm_neuron.functional.moe.router import (
            noaux_tc_rmsnorm_router_topk,
        )

        logits, expert_index, expert_affinities, _substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=self.router_weight,
                correction_bias=self.router_bias,
                top_k=int(text_config.num_experts_per_tok),
                eps=eps,
                norm_topk_prob=bool(text_config.norm_topk_prob),
                routed_scaling_factor=float(text_config.routed_scaling_factor),
            )
        )
        return logits, expert_index, expert_affinities

    # The public ``moe_cte`` dispatcher does not forward block scales. Normal
    # model calls use the fused kernel with load-time packed banks; direct callers
    # can still pass the four legacy operands for the three limbs. Both routes
    # consume the same device mapping and emit fp32 contributions.
    #
    # The not-block-quant branch raises by name rather than falling through to
    # the substrate's ``QuantizationType.none`` default, which is reached by
    # omission -- a call site that forgot to route would get it with no error at
    # all.
    #
    # The weights, scales and config are arguments rather than state: ``__init__``
    # declares the three projection parameters and no scale parameter, and the
    # fused ``[E, H, 2, I_TP]`` gate/up tensor the kernel needs is not the
    # checkpoint's per-projection storage -- building it is the weight loader's
    # step. What is read off ``self`` is the partition.
    #
    # Two ``to_kernel_scale_layout`` helpers exist under the same name with
    # different signatures (``functional/blockwise_fp8_mm.py`` takes
    # ``(weight_scale, rows, cols)``; ``functional/moe/moe_blockwise_fp8.py``
    # takes ``(consumer_scales, num_experts, rows, cols, projection)``), so the
    # one this site needs is imported under an alias that names its module -- an
    # unqualified import of both would be an arity failure that reads as a shape
    # bug.

    def block_quant_expert_mm(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        gate_up_proj_weight: torch.Tensor | None,
        down_proj_weight: torch.Tensor | None,
        gate_up_scale_operands: torch.Tensor | None,
        down_scale_operands: torch.Tensor | None,
        quant_config: Glm5NextQuantConfig,
        *,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        packed_weights: torch.Tensor | None = None,
        packed_scales: torch.Tensor | None = None,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route, compute and combine this rank's block-quant expert outputs.

        Normal model calls supply both packed banks and leave the four legacy
        operands ``None``. Direct callers can still supply those four operands to
        run the three-kernel chain. Both paths use the same routing, padding,
        fp32 contribution combine and final activation dtype.

        Args:
            hidden_states: ``[T, H]`` real tokens only -- the kernel's
                padding-token slot is appended here, not by the caller.
            expert_affinities: ``[T, E]`` scattered router scores over all
                ``n_routed_experts``, which is the form :meth:`route_tokens`
                returns: the gate weight at each selected expert's column and
                zero elsewhere. This site does the global-to-local mapping, so
                the caller hands the router's output straight through and slices
                nothing.
            gate_up_proj_weight: ``[E_local, H, 2, I_TP]`` fp8-e4m3, the
                checkpoint's own bytes at its own ``[128, 128]`` granularity.
            down_proj_weight: ``[E_local, I_TP, H]`` fp8-e4m3, the same.
            gate_up_scale_operands: ``[E_local, TILE_SIZE, n_blocks]`` fp32, one
                per-expert kernel scale operand as
                :func:`~vllm_neuron.functional.moe.moe_blockwise_fp8.to_gate_up_kernel_scale_operand`
                builds it from the checkpoint grid. The grid carries both halves
                already, so there is no fusion merge to do.
            down_scale_operands: the same, from
                :func:`~vllm_neuron.functional.moe.moe_blockwise_fp8.to_down_kernel_scale_operand`.
            quant_config: the resolved per-model quantisation policy, and the
                route selector.
            block_size: tokens per block, a multiple of ``BLOCK_QUANT_SIZE``.
                Defaults to ``BLOCK_QUANT_SIZE`` itself.
            moe_group: the MoE ``GroupCoordinator``, forwarded verbatim to
                ``build_blockwise_mapping``, which reads it only when
                ``tp_degree > 1`` -- so an undistributed call site may leave it
                ``None`` rather than have this site invent one.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select. An int64
                device tensor of shape ``[1]``, which is what the runner hands
                over, makes the slice an input of the captured graph so every
                rank shares one graph; a python int keeps it a constant of the
                trace, and either form is selected by name. The default 0 is the
                only rank that exists at degree 1.
            packed_weights: optional packed FP8 bank
                ``[E, 3*(I/128), 128, H/128, 128]``.
            packed_scales: matching fp32 scales ``[E, 3*(I/128), H/128]``. Supply
                both packed banks or neither; they replace the four legacy
                operands.
            collector: when given, receives the mapping's token-position tensor.

        Returns:
            ``[T, H]`` -- the padding-token row is sliced off. The dtype is the
            kernel's own (``bfloat16`` on the NKI route) and is not re-cast here;
            the layer forward decides the residual dtype.

        Raises:
            Glm5NextBlockQuantRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one the
                retile bridges, or when an operand extent contradicts another.
                Named rather than coerced, so a mis-wired call site fails where
                it is wrong instead of computing a different function.
        """
        from vllm_neuron.functional import (
            build_blockwise_mapping,
            get_local_expert_affinities,
        )
        from vllm_neuron.functional.moe.blockwise_fp8_retile import (
            BLOCK_QUANT_SIZE,
            TILE_SIZE,
        )
        from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
            moe_down_blockwise_fp8,
            moe_gate_up_blockwise_fp8,
            moe_swiglu_transposed,
        )

        # ---- Route selection ---------------------------------------------- #
        if not quant_config.is_block_quantized:
            raise Glm5NextBlockQuantRouteError(
                "block_quant_expert_mm is the block-quant route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised expert path at this "
                "site, and silently continuing is what would reach the "
                "substrate's QuantizationType.NONE default by omission."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextBlockQuantRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes the checkpoint's own "
                f"({TILE_SIZE}, {TILE_SIZE}) block scales, at that granularity "
                f"and unaltered, and has no path for any other checkpoint "
                f"block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        if hidden_states.dim() != 2:
            raise Glm5NextBlockQuantRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        tokens, hidden = (int(extent) for extent in hidden_states.shape)
        using_packed = packed_weights is not None or packed_scales is not None
        if using_packed:
            if packed_weights is None or packed_scales is None:
                raise Glm5NextBlockQuantRouteError("Supply both packed weights and scales")
            if any(value is not None for value in (
                gate_up_proj_weight, down_proj_weight,
                gate_up_scale_operands, down_scale_operands,
            )):
                raise Glm5NextBlockQuantRouteError(
                    "Packed banks replace the four legacy expert operands; do not supply both"
                )
            if (
                packed_weights.ndim != 5
                or packed_weights.shape[1] < 3
                or packed_weights.shape[1] % 3
                or packed_weights.shape[2] != TILE_SIZE
                or packed_weights.shape[4] != TILE_SIZE
                or packed_weights.shape[3] * TILE_SIZE != hidden
            ):
                raise Glm5NextBlockQuantRouteError(
                    "packed_weights must be [E, 3*(I/128), 128, H/128, 128] "
                    f"for hidden width {hidden}, got {tuple(packed_weights.shape)}"
                )
            num_experts = int(packed_weights.shape[0])
            intermediate = int(packed_weights.shape[1]) // 3 * TILE_SIZE
            if tuple(packed_scales.shape) != (
                num_experts, packed_weights.shape[1], packed_weights.shape[3]
            ):
                raise Glm5NextBlockQuantRouteError(
                    "packed_scales must match the packed weight tiles"
                )
        else:
            if any(value is None for value in (
                gate_up_proj_weight, down_proj_weight,
                gate_up_scale_operands, down_scale_operands,
            )):
                raise Glm5NextBlockQuantRouteError(
                    "Supply both packed banks or all four legacy expert operands"
                )
            if gate_up_proj_weight.dim() != 4 or gate_up_proj_weight.shape[2] != 2:
                raise Glm5NextBlockQuantRouteError(
                    f"gate_up_proj_weight must be [E, H, 2, I_TP], got shape "
                    f"{tuple(gate_up_proj_weight.shape)}"
                )
            num_experts = int(gate_up_proj_weight.shape[0])
            intermediate = int(gate_up_proj_weight.shape[-1])
            if int(gate_up_proj_weight.shape[1]) != hidden:
                raise Glm5NextBlockQuantRouteError(
                    f"gate_up_proj_weight has H={int(gate_up_proj_weight.shape[1])} "
                    f"but hidden_states has H={hidden}"
                )
        # The partition is read before the second operand's shape, because the
        # expert count comes off the first operand and every later shape is stated
        # in terms of it. Only this order names the cause when a bank is truncated
        # to another rank's width: the bank disagrees with the partition, rather
        # than a down projection whose extent "must be" a number this rank should
        # never have derived.
        if num_experts != int(self.num_local_experts):
            raise Glm5NextBlockQuantRouteError(
                f"the expert bank carries {num_experts} experts but this "
                f"rank owns {self.num_local_experts}; the bank and the "
                f"partition must agree"
            )
        if down_proj_weight is not None and tuple(down_proj_weight.shape) != (
            num_experts, intermediate, hidden
        ):
            raise Glm5NextBlockQuantRouteError(
                f"down_proj_weight must be [E={num_experts}, "
                f"I_TP={intermediate}, H={hidden}], got shape "
                f"{tuple(down_proj_weight.shape)}"
            )
        # ---- The dispatch step: global router columns -> this rank's slice.  #
        # the fork's own MoE call site is the precedent, including the guard: it
        # maps only when the degree is above 1 and passes the affinities through
        # untouched at degree 1.
        routed = int(self.num_routed_experts)
        if tuple(expert_affinities.shape) != (tokens, routed):
            raise Glm5NextBlockQuantRouteError(
                f"expert_affinities must be [T={tokens}, E={routed}] -- the "
                f"GLOBAL router width, which is what route_tokens returns and "
                f"what this site maps onto this rank's {num_experts} experts -- "
                f"got shape {tuple(expert_affinities.shape)}"
            )
        if routed != num_experts:
            # The rank arrives as a device tensor, so the slice it selects is an
            # input of the captured graph and every rank shares one graph; a python
            # int would bake the group into the trace, one graph per rank. The
            # partition is uniform and contiguous, so the first expert this rank
            # owns is ``rank * E_local``, and an ``arange`` from zero builds the run
            # without a tensor made from python data, which graph extraction
            # refuses beside fake ones.
            if num_experts != int(self.num_local_experts):
                raise Glm5NextBlockQuantRouteError(
                    f"the prepared weights hold {num_experts} experts and the "
                    f"partition assigns {int(self.num_local_experts)} per rank; "
                    f"the rank offset is right only when the two agree"
                )
            if isinstance(expert_parallel_rank, int) and not (
                0 <= expert_parallel_rank < int(self.ep_degree)
            ):
                raise Glm5NextBlockQuantRouteError(
                    f"expert_parallel_rank={expert_parallel_rank} is outside the "
                    f"partition's {int(self.ep_degree)} ranks"
                )
            local_indices = (
                torch.arange(
                    0,
                    num_experts,
                    dtype=torch.int64,
                    device=expert_affinities.device,
                )
                + expert_parallel_rank * num_experts
            )
            expert_affinities = get_local_expert_affinities(
                expert_affinities, local_indices
            )
        if tuple(expert_affinities.shape) != (tokens, num_experts):
            raise Glm5NextBlockQuantRouteError(
                f"the global-to-local mapping produced "
                f"{tuple(expert_affinities.shape)} and this rank owns "
                f"{num_experts} experts over {tokens} tokens; the partition and "
                f"the gather disagree"
            )
        block = BLOCK_QUANT_SIZE if block_size is None else int(block_size)
        if block <= 0 or block % BLOCK_QUANT_SIZE:
            raise Glm5NextBlockQuantRouteError(
                f"block_size={block} is not a positive multiple of "
                f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE} "
                f"(bwmm_shard_on_I.py)"
            )

        # ---- The token-block mapping, over the real token count. --------- #
        # the order is load-bearing. The kernel needs a padding-token slot, and
        # appending it to the affinities before the mapping would make the mapping
        # see ``T + 1`` tokens -- odd for every even real count -- while its two
        # NKI subkernels need ``chunk_size % 128 == 0`` and
        # ``total_tokens % f_len == 0``. Both fail on an odd count, so every call
        # would fall to the torch mapping instead.
        #
        # ``conditions`` is deliberately unconsumed: it feeds the ``*_hybrid``
        # dynamic-while variant of the vendor kernel, and this route calls the
        # non-hybrid inner member, whose signature has no such parameter. Named
        # with a leading underscore rather than dropped, so the unused fourth
        # return is visible instead of implied.
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _conditions,
        ) = build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=num_experts,
            num_experts_per_token=int(self.num_experts_per_tok),
            block_size=block,
            moe_group=moe_group,
            tp_degree=tp_degree,
        )
        if collector is not None:
            # The mapping is a device-only reading: it is what says whether every
            # real token reached all of its experts once the padded rows were
            # blocked alongside it.
            collector.append(token_position_to_id)

        # ---- The padding-token slot, appended after the mapping. --------- #
        # the kernel reads a ``-1`` token position as the last row of
        # ``hidden_states``, so the hidden tensor grows one zero row and the result
        # is sliced back at the end.
        #
        # The affinities grow the same slot in their flat form, because that is
        # what the mapping returns: ``expert_affinities_masked`` is
        # ``[T * E_local, 1]`` and the layout is token-major, a ``view(-1, 1)`` of
        # ``[T, E]``. Appending ``E_local`` zero entries to the flat tensor is
        # therefore the same tensor as appending one zero row before the view.
        # Only this order keeps ``total_tokens`` even.
        pad_hidden = torch.zeros(
            1, hidden, dtype=hidden_states.dtype, device=hidden_states.device
        )
        padded_hidden = torch.cat([hidden_states, pad_hidden], dim=0)
        pad_masked = torch.zeros(
            num_experts,
            1,
            dtype=expert_affinities_masked.dtype,
            device=expert_affinities_masked.device,
        )
        expert_affinities_masked = torch.cat(
            [expert_affinities_masked, pad_masked], dim=0
        )

        # Both paths gather tokens and expert tiles on device, clamp gate from
        # above and up on both sides, then apply affinity after down projection.
        if using_packed:
            from vllm_neuron.functional.moe.fused_fp8 import (
                PackedExperts,
                fused_fp8_experts,
            )
            from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
                _swiglu_bound_operand,
            )

            # Each block stores its real tokens first and has at most T of
            # them. Keep that prefix for the kernel and the FP32 combine.
            rows = min(tokens, block)
            kernel_row_ids = token_position_to_id.reshape(-1, block)[:, :rows].contiguous()
            token_position_to_id = kernel_row_ids.reshape(-1)
            contribution = fused_fp8_experts(
                padded_hidden.to(torch.bfloat16),
                PackedExperts(packed_weights, packed_scales),
                kernel_row_ids,
                block_to_expert.reshape(-1, 1),
                expert_affinities_masked.reshape(tokens + 1, num_experts),
                _swiglu_bound_operand(
                    self.swiglu_limit, self.swiglu_limit, hidden_states.device
                ),
            )
        else:
            # Preserve the direct-call interface for the existing chain.
            pre_activation = moe_gate_up_blockwise_fp8(
                padded_hidden,
                gate_up_proj_weight,
                gate_up_scale_operands,
                token_position_to_id,
                block_to_expert,
                block,
            )
            contribution = moe_down_blockwise_fp8(
                moe_swiglu_transposed(
                    pre_activation, self.swiglu_limit, self.swiglu_limit
                ),
                down_proj_weight,
                down_scale_operands,
                expert_affinities_masked,
                token_position_to_id,
                block_to_expert,
                block,
                tokens,
            )

        # ---- Back to token order: one scatter-add over the whole emission. -- #
        # the router weight multiplies after the down projection, inside the kernel
        # above, and never into the hidden states. The checkpoint projects the
        # unscaled token and then scales, and the two are different functions rather
        # than rearrangements of one, because the nonlinear activation breaks the
        # linearity.
        #
        # Accumulate every expert contribution in token order with one device
        # ``index_add``, in fp32, before restoring the input dtype.
        # The captured Torch mapping gives each real token at most one row
        # per block. Preserve block order and the existing final dtype cast.
        if using_packed and torch.compiler.is_compiling() and tokens > 1:
            from vllm_neuron.functional.moe.ordered_combine import ordered_combine

            block_h = 512
            for tile in (2048, 1024):
                if contribution.shape[1] % (2 * tile) == 0:
                    block_h = tile
                    break
            return ordered_combine(
                contribution, kernel_row_ids, tokens, BLOCK_H=block_h
            ).to(hidden_states.dtype)

        wanted = torch.where(
            token_position_to_id < 0,
            torch.full_like(token_position_to_id, tokens),
            token_position_to_id,
        ).long()
        accumulated = torch.zeros(
            tokens + 1, hidden, dtype=torch.float32, device=hidden_states.device
        ).index_add(0, wanted, contribution)
        # The padding row is dropped rather than masked: it accumulated whatever the
        # padded positions computed, and no real token indexes it.
        return accumulated[:tokens].to(hidden_states.dtype)

    # ── load-time operand prep ────────────────────────────────────────────
    #
    # ``block_quant_expert_mm`` consumes one packed weight bank and one packed
    # scale bank; the checkpoint supplies three weights and three matching scale
    # grids. Both packed banks are built once, at load time, and retained.
    #
    # ``Glm5NextForConditionalGeneration._run_load_time_preps`` is the single
    # production caller of either prep and enrols a module by
    # ``hasattr(type(module), "prepare_scale_operands")``, deriving the operand
    # names from this module's own declaration tuple through
    # ``_scale_prep_leaves``. It hands exactly the six operands below, by keyword.
    #
    # One method builds the weights as well, though it is named for scales: a
    # weight and its scale grid are one pair -- the kernel indexes the grid by the
    # weight's own tiles -- and the two orientations below move together, or the
    # pair means something else.

    #: Where :meth:`prepare_scale_operands` leaves its two packed banks. A class
    #: attribute because the name is part of the contract between the builder and
    #: the reader, and neither should spell it twice.
    PREPARED_KERNEL_OPERANDS_ATTR = "_prepared_kernel_operands"

    #: The checkpoint tensors the forward stops reading once
    #: :meth:`prepare_scale_operands` has copied them into the kernel operands:
    #: the three weights and their three scale grids. The load releases these.
    RELEASED_AFTER_PREP: tuple[str, ...] = (
        "gate_proj_weight",
        "up_proj_weight",
        "down_proj_weight",
        f"gate_proj_{FP8_SCALE_SUFFIX}",
        f"up_proj_{FP8_SCALE_SUFFIX}",
        f"down_proj_{FP8_SCALE_SUFFIX}",
    )

    #: Where the producer's three health counts are left. Recorded rather than
    #: refused on -- see the method's Raises note.
    RETILE_HEALTH_ATTR = "_retile_health"

    def prepare_scale_operands(
        self,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
    ) -> int:
        """Pack this rank's prepared FP8 weights and scales once. Returns 2.

        The loader supplies gate/up weights ``[E, I, H]``, down weights
        ``[E, H, I]`` and their matching ``[128, 128]`` block grids, already
        squeezed into the finite Trainium2 range with compensated scales. This
        method only changes layout; it does not rescale or requantize.

        Real tensors are packed on the host and moved back to the source device.
        Meta tensors follow the same layout without reading values. Only the two
        packed banks remain in the prepared operand dictionary, and the load hook
        releases the six source tensors afterwards.
        """
        from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts

        supplied = {
            "gate_proj_weight": gate_proj_weight,
            "up_proj_weight": up_proj_weight,
            "down_proj_weight": down_proj_weight,
            "gate_proj_scale": gate_proj_scale,
            "up_proj_scale": up_proj_scale,
            "down_proj_scale": down_proj_scale,
        }
        missing = sorted(name for name, value in supplied.items() if value is None)
        if missing:
            raise Glm5NextBlockQuantRouteError(
                f"prepare_scale_operands needs all six operands and {missing} "
                "are absent; load the checkpoint before preparing the kernel operands"
            )
        for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            weight = supplied[name]
            if weight.dim() != 3 or int(weight.shape[0]) != int(self.num_local_experts):
                raise Glm5NextBlockQuantRouteError(
                    f"{name} must be [E_local={self.num_local_experts}, ., .], "
                    f"got shape {tuple(weight.shape)}"
                )

        device = gate_proj_weight.device
        (
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            gate_proj_scale,
            up_proj_scale,
            down_proj_scale,
        ) = _on_the_host(*supplied.values())
        # Each scale grid moves with its weight. Gate and up share a fusion
        # axis; down keeps contraction-major order. These are host/meta copies.
        gate_up = torch.stack(
            (gate_proj_weight.transpose(1, 2), up_proj_weight.transpose(1, 2)),
            dim=2,
        ).contiguous()
        gate_up_grid = torch.stack(
            (gate_proj_scale.transpose(1, 2), up_proj_scale.transpose(1, 2)),
            dim=2,
        ).contiguous()
        packed = pack_experts(
            gate_up.reshape(gate_up.shape[0], gate_up.shape[1], -1),
            down_proj_weight.transpose(1, 2).contiguous(),
            gate_up_grid,
            down_proj_scale.transpose(1, 2).contiguous(),
        )
        weights, scales = _on_the_device(device, packed.weights, packed.scales)
        prepared = {"packed_weights": weights, "packed_scales": scales}
        setattr(self, self.PREPARED_KERNEL_OPERANDS_ATTR, prepared)
        # Packing is a byte permutation. No scale is dropped or merged and
        # no weight is rescaled. Retain the existing health record's meaning.
        setattr(
            self,
            self.RETILE_HEALTH_ATTR,
            {name: (0, 0, 0) for name in ("gate", "up", "down")},
        )
        return len(prepared)

    def _prepared_kernel_operand(self, name: str) -> torch.Tensor:
        """One prebuilt kernel operand, or a refusal naming what was not done.

        Refusing is what makes "built once at load time, never per forward step"
        checkable. Building on demand would put a whole-bank pack inside the
        per-token path with nothing to report it.
        """
        prepared = getattr(self, self.PREPARED_KERNEL_OPERANDS_ATTR, None)
        if not prepared:
            raise Glm5NextBlockQuantRouteError(
                "prepare_scale_operands() has not run; this bank's kernel "
                "operands are packed once at load time, never per forward step"
            )
        return prepared[name]

    # The bank's forward is the composition and nothing else.
    # ``block_quant_expert_mm`` above does the route dispatch, the global-to-local
    # expert mapping, the padding slot and the kernel call, and
    # ``prepare_scale_operands`` built the two packed banks it takes. A refusal
    # here would be a second authority on an extent the callee already checks, and
    # two refusals on one extent is how they come to disagree.
    #
    # The affinities are an argument rather than routed here: ``route_tokens``
    # needs the router norm's gamma and the text config, neither of which this
    # bank retains, and the MoE block's forward is where the router and this bank
    # meet. There is no cast, because the layer forward decides the residual
    # dtype.

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
        *,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run this rank's routed experts over ``[T, H]`` tokens.

        Args:
            hidden_states: ``[T, H]`` real tokens only.
            expert_affinities: ``[T, E]`` scattered router scores over all
                ``n_routed_experts`` -- what :meth:`route_tokens` returns, handed
                straight through.
            quant_config: the resolved quantisation policy, the route selector.
            block_size: tokens per block; defaults to ``BLOCK_QUANT_SIZE``.
            moe_group: the MoE ``GroupCoordinator``, unread at ``tp_degree`` 1.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select, as an
                int64 device tensor (one graph for every rank) or a python int.

        Returns:
            ``[T, H]`` in the kernel's own dtype.

        Raises:
            Glm5NextBlockQuantRouteError: if the load-time prep has not run, and
                on any extent disagreement :meth:`block_quant_expert_mm`
                refuses. Both are raised by the methods that own them, which is
                why this one raises nothing.
        """
        return self.block_quant_expert_mm(
            hidden_states=hidden_states,
            expert_affinities=expert_affinities,
            gate_up_proj_weight=None,
            down_proj_weight=None,
            gate_up_scale_operands=None,
            down_scale_operands=None,
            quant_config=quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
            packed_weights=self._prepared_kernel_operand("packed_weights"),
            packed_scales=self._prepared_kernel_operand("packed_scales"),
            collector=collector,
        )


# The block-quant route's named refusal, at module level because an exception a
# caller catches belongs in the module namespace rather than nested in the class
# that raises it.
class Glm5NextBlockQuantRouteError(ValueError):
    """A block-quant expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a call
    site that reaches the substrate's ``QuantizationType.none`` default by
    omission and computes a different function while every shape check passes.
    """


class Glm5NextSharedExperts(nn.Module):
    """The always-on shared expert at ``mlp.shared_experts``.

    Declared only when ``n_shared_experts`` is nonzero, mirroring the weight
    map's own condition.
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_shared_experts = int(text_config.n_shared_experts)
        # The checkpoint's SwiGLU bound, resolved here rather than at the call. The
        # reference clamps both MLP projections with this value before their
        # product, so it is a model scalar like ``n_shared_experts`` above and
        # belongs on the object the config builds.
        self.swiglu_limit = float(text_config.swiglu_limit)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    # ── the shared-expert path ────────────────────────────────────────────
    #
    # Three sequential calls into ``blockwise_fp8_mm``, one per projection.
    # ``silu`` is non-linear, so ``down`` cannot fold into either predecessor, and
    # the kernel takes a 2-D weight. Gate and up stay separate parameters,
    # matching the weight map and the dense MLP below; concatenating them into one
    # ``[H, 2I]`` operand would need a scale-grid concatenation this path does not
    # author.
    #
    # The residual add lives on ``Glm5NextMoEBlock``, because adding the routed
    # and shared halves needs both children and neither child owns the other.
    #
    # Model scalars resolve at construction; public weight-loader products arrive
    # at the call; and an operand derived from a weight-loader product that never
    # changes is built once at load time and held. That is why ``swiglu_limit``
    # sits on the object while the three public grids stay arguments of
    # :meth:`shared_expert_mm` -- the kernel's torch-oracle fallback consumes the
    # public grid, and a prebuilt kernel operand cannot stand in for it.
    #
    # Why the derived operand is built at load time rather than cached inside the
    # kernel's bridge: ``to_kernel_scale_layout`` allocates and then scatters one
    # element at a time, which is 128 device writes per projection and 384 per
    # shared-expert call at this geometry. A dict keyed on a traced tensor is also
    # not expressible on the traced path, and capture feeds ``meta`` tensors, which
    # carry no values to key on. The operand has to arrive as a graph input, which
    # is what building it before capture makes it.
    #
    # Skipping the bridge also skips its grid check, and the kernel's replacement
    # check is weaker -- it pins the element count, so a transposed grid would pass
    # it. The strong check still runs, because ``prepare_scale_operands`` builds
    # through ``to_kernel_scale_layout`` rather than around it: the public grid is
    # still compared against ``(rows, cols)`` before it is ever flattened, once per
    # load instead of once per forward step.
    #
    # Two helpers named ``to_kernel_scale_layout`` exist at different arities
    # (``functional/blockwise_fp8_mm.py`` takes ``(weight_scale, rows, cols)``;
    # ``functional/moe/moe_blockwise_fp8.py`` takes
    # ``(consumer_scales, num_experts, rows, cols, projection)``), so the import
    # below names its module in full.

    #: Where :meth:`prepare_scale_operands` leaves its three operands. A class
    #: attribute because the name is part of the contract between the builder and
    #: the reader, and neither should spell it twice.
    PREPARED_SCALE_OPERANDS_ATTR = "_prepared_scale_operands"

    #: Nothing. :meth:`prepare_scale_operands` builds scale operands only, and
    #: :meth:`scale_route_operands` reads the raw weights and grids on every
    #: forward, so the load releases no tensor of this class.
    RELEASED_AFTER_PREP: tuple[str, ...] = ()

    #: Where :meth:`retile_checkpoint_scale_grids` records what it published per
    #: projection, and which projections it left alone. The three producer counters
    #: remain in the record and read zero by absence, now that nothing is
    #: coarsened. No consumer reads it.
    SHARED_RETILE_HEALTH_ATTR = "_shared_expert_retile_health"

    # ── the load-path grid publish ────────────────────────────────────────
    #
    # The checkpoint stores one scale per ``128 x 128`` tile, which is the
    # granularity the dense kernel indexes, so there is nothing to coarsen: this
    # step publishes the grid the loader delivered and transposes it into the frame
    # ``blockwise_fp8_mm`` multiplies in. It rescales no weight byte.
    #
    # Coarsening was needed while the kernel indexed ``256 x 256`` blocks, and it
    # is why the removal is safe rather than convenient: keeping one of four tile
    # scales per block left the other three tiles' values wrong against the
    # retained scale until rescaled by their own ratio. Publishing a coarser grid
    # beside an unscaled weight would change no shape and every number, and the
    # kernel's element-count check would pass it.
    #
    # The skip is recorded, never silent. A weight whose extents are not whole
    # ``128`` blocks has no grid the kernel can index, so this method leaves that
    # projection exactly as the loader left it and says so in the health record.

    def retile_checkpoint_scale_grids(self) -> int:
        """Publish this module's grids at the kernel's granularity and set the frame.

        The body lives in :func:`_publish_compute_frame_operands`, which the dense
        MLP shares: each weight and its grid are transposed once into the frame
        ``blockwise_fp8_mm`` multiplies in.

        The name is a misnomer, kept on purpose. The dense kernel indexes the
        checkpoint's own ``128``-tile grid, so nothing is coarsened here and no
        weight byte is rescaled; the name is public API that callers read, so renaming
        it is a separate change.

        Returns:
            How many projections were published -- ``3`` on a checkpoint whose
            extents are whole ``128`` blocks, ``0`` on a miniature whose are not.
            What the transpose did is in the health record, per projection, under
            :attr:`SHARED_RETILE_HEALTH_ATTR`.

        Raises:
            Glm5NextSharedExpertRouteError: if a weight or grid is not 2-D, or a
                grid is not at the checkpoint's own ``128``-tile granularity. A
                grid at any other granularity is refused rather than reshaped: it
                was built for a different consumer and no shape would object.
        """
        return _publish_compute_frame_operands(
            self, Glm5NextSharedExpertRouteError, self.SHARED_RETILE_HEALTH_ATTR
        )

    def prepare_scale_operands(
        self,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
    ) -> int:
        """Build the three kernel scale operands once. Returns how many.

        The kernel's bridge scatters one element at a time, so building the
        operand inside the per-forward path costs 128 device writes per
        projection at this dense geometry. A block scale is a weight-loader
        product that never changes after a load, so the operand it implies is
        built here, once, and held on this module.

        The argument list mirrors :meth:`shared_expert_mm`'s own -- same operands,
        same order -- because the two methods consume the same things and a
        divergent order at one of them is a mis-wiring no shape check would see.

        Args:
            gate_proj_weight: ``[H, I]`` fp8-e4m3. Read for its extents only.
            up_proj_weight: ``[H, I]`` fp8-e4m3.
            down_proj_weight: ``[I, H]`` fp8-e4m3.
            gate_proj_scale: ``[H//128, I//128]`` fp32 -- the checkpoint's own
                grid, which is the grid the kernel indexes.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[I//128, H//128]`` fp32, for ``down_proj_weight``.

        Returns:
            How many operands were built -- ``3`` on every successful call.

        Raises:
            Glm5NextSharedExpertRouteError: if an operand is missing or a weight
                is not 2-D. The grid's own agreement with the weight extents is
                checked by ``to_kernel_scale_layout`` below rather than here, so
                there is one authority for it and not two that can drift.

        The extents come from the weights and not from the grid's own shape:
        deriving them from the grid would make this method unable to detect a
        transposed grid, because the kernel's replacement check pins only the
        element count.
        """
        from vllm_neuron.functional.blockwise_fp8_mm import to_kernel_scale_layout

        prepared: dict[str, torch.Tensor] = {}
        for name, weight, scale in (
            ("gate_proj", gate_proj_weight, gate_proj_scale),
            ("up_proj", up_proj_weight, up_proj_scale),
            ("down_proj", down_proj_weight, down_proj_scale),
        ):
            if weight is None or scale is None:
                raise Glm5NextSharedExpertRouteError(
                    f"prepare_scale_operands needs both {name}_weight and "
                    f"{name}_scale; load the checkpoint before preparing the "
                    f"scale operands"
                )
            if weight.dim() != 2:
                raise Glm5NextSharedExpertRouteError(
                    f"{name}_weight must be 2-D to give the scale operand its "
                    f"extents, got shape {tuple(weight.shape)}"
                )
            rows, cols = int(weight.shape[0]), int(weight.shape[1])
            # The producer runs on a host copy and its result moves back once, the
            # same rule the routed bank's prep follows. ``to_kernel_scale_layout``
            # broadcasts the flat grid across the partition axis with
            # ``expand(...).contiguous()``, and the Neuron backend has no strided
            # copy to make that dense, so on a device-resident grid it raises
            # ``Expected self.is_contiguous() to be true, but got false``. Nothing
            # about the operand changes: the grid check it performs is on shapes,
            # which the copy preserves.
            (host_scale,) = _on_the_host(scale)
            (operand,) = _on_the_device(
                scale.device, to_kernel_scale_layout(host_scale, rows, cols)
            )
            prepared[name] = operand
        setattr(self, self.PREPARED_SCALE_OPERANDS_ATTR, prepared)
        return len(prepared)

    def _prepared_scale_operand(self, name: str) -> torch.Tensor:
        """One prebuilt scale operand, or a refusal naming what was not done.

        Refusing is what makes "never per forward step" checkable. Falling back
        to building on demand would put the per-call scatter back silently, with
        nothing to report it.
        """
        prepared = getattr(self, self.PREPARED_SCALE_OPERANDS_ATTR, None)
        if not prepared:
            raise Glm5NextSharedExpertRouteError(
                "prepare_scale_operands() has not run; the kernel scale "
                "operands are built once at load time, never per forward step"
            )
        return prepared[name]

    def shared_expert_mm(
        self,
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """Run the always-on shared expert through the dense block gemm.

        The SwiGLU the checkpoint stores, which clamps both projections before the
        product: ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))``, where ``L``
        is ``self.swiglu_limit``, the checkpoint's own bound resolved from the
        config when this object was built. Each of the three projections is a
        separate entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`.

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. ``T`` may be any
                positive count: the kernel tiles ``M`` over the PSUM partition
                axis and does not pad, so this method pads to a whole
                ``TILE_SIZE`` and slices the result back before returning. A
                ``T`` that is already a whole tile is not copied.
            gate_proj_weight: ``[H, I]`` fp8-e4m3, expressed against
                ``gate_proj_scale``.
            up_proj_weight: ``[H, I]`` fp8-e4m3.
            down_proj_weight: ``[I, H]`` fp8-e4m3.
            gate_proj_scale: ``[H//128, I//128]`` fp32, the block-scale grid
                :func:`~vllm_neuron.functional.blockwise_fp8_mm.scale_grid_shape`
                declares -- one scale per ``128 x 128`` weight block, which is
                the checkpoint's own grid.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[I//128, H//128]`` fp32, for ``down_proj_weight``.
            quant_config: the resolved per-model quantisation policy, and the
                route selector.

        The SwiGLU bound is not an argument. It is ``self.swiglu_limit``,
        resolved from ``text_config.swiglu_limit`` in ``__init__``, so this method
        cannot be called with a bound other than the one the checkpoint declared
        for this model, and no literal bound appears on this path.

        Returns:
            ``[T, H]`` fp32 -- the kernel's own return dtype, not re-cast here.
            The layer forward decides the residual dtype.

        Raises:
            Glm5NextSharedExpertRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one the grids
                are published at, or when two operand extents contradict each
                other. Named rather than coerced, so a mis-wired call site fails
                where it is wrong instead of computing a different function.
        """
        from torch.nn.functional import silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm
        from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

        # ---- Route selection ---------------------------------------------- #
        # the route is selected by which function is called, and the
        # not-block-quant branch raises by name, so it cannot fall through to the
        # substrate's ``QuantizationType.none`` default, which is reached by
        # omission.
        if not quant_config.is_block_quantized:
            raise Glm5NextSharedExpertRouteError(
                "shared_expert_mm is the block-quant shared-expert route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised shared-expert path at "
                "this site, and silently continuing is what would reach the "
                "substrate's QuantizationType.NONE default by omission."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextSharedExpertRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes the checkpoint's own ({TILE_SIZE}, {TILE_SIZE}) "
                f"blocks directly -- the dense kernel "
                f"indexes at that granularity and nothing is retiled -- and has "
                f"no path for any other checkpoint block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        # only the cross-operand agreements the kernel cannot see are checked here.
        # ``blockwise_fp8_mm`` already refuses a K mismatch, a mis-sized or
        # non-fp32 scale grid and every blocking condition, and repeating those
        # would create a second authority that can drift from the first.
        if hidden_states.dim() != 2:
            raise Glm5NextSharedExpertRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        hidden = int(hidden_states.shape[1])
        if tuple(gate_proj_weight.shape) != tuple(up_proj_weight.shape):
            raise Glm5NextSharedExpertRouteError(
                f"gate_proj_weight {tuple(gate_proj_weight.shape)} and "
                f"up_proj_weight {tuple(up_proj_weight.shape)} must have the "
                f"same [H, I] extents; they are multiplied elementwise after "
                f"the activation"
            )
        if gate_proj_weight.dim() != 2 or int(gate_proj_weight.shape[0]) != hidden:
            raise Glm5NextSharedExpertRouteError(
                f"gate_proj_weight must be [H={hidden}, I], got shape "
                f"{tuple(gate_proj_weight.shape)}"
            )
        intermediate = int(gate_proj_weight.shape[1])
        if tuple(down_proj_weight.shape) != (intermediate, hidden):
            raise Glm5NextSharedExpertRouteError(
                f"down_proj_weight must be [I={intermediate}, H={hidden}], got "
                f"shape {tuple(down_proj_weight.shape)}"
            )

        # ---- Pad to a whole tile ------------------------------------------ #
        # the pad is created here, consumed by the three calls below and removed at
        # the single return, so it never leaves this call: nothing outside can
        # observe it. The extent checks above ran on the caller's own tensor, so a
        # mis-shaped operand still fails on what the caller passed.
        hidden_states, tokens = _pad_tokens_to_tile(hidden_states, TILE_SIZE)

        # ---- The three projection sites ----------------------------------- #
        # each passes the operand ``prepare_scale_operands`` built at load time, by
        # keyword. The public grid is still passed too: the kernel's torch-oracle
        # fallback consumes it, and the prebuilt operand cannot stand in for it.
        gate = blockwise_fp8_mm(
            hidden_states,
            gate_proj_weight,
            gate_proj_scale,
            prebuilt_scale_t=self._prepared_scale_operand("gate_proj"),
        )
        up = blockwise_fp8_mm(
            hidden_states,
            up_proj_weight,
            up_proj_scale,
            prebuilt_scale_t=self._prepared_scale_operand("up_proj"),
        )

        # ---- The checkpoint's two clamps ---------------------------------- #
        # the correctness reference -- transformers v5.16.1,
        # ``Glm5NextTextMLP.forward``, which is also what builds
        # ``Glm5NextTextMoE.shared_experts`` -- bounds the gate above and the up
        # operand on both sides before multiplying:
        #
        #     gate = gate.clamp(min=None, max=self.swiglu_limit)
        #     up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        #
        # The two lines below are that transliteration, ``min=None`` on the gate
        # included: the gate's bound is one-sided in the reference, and copying it
        # as a two-sided clamp would be a second wrong function rather than a
        # tidier one. Unclamped, every token whose pre-activations leave the
        # checkpoint's ``[-10, 10]`` box enters the residual stream wrong, with no
        # shape moving and nothing raising.
        #
        # The bound is ``self.swiglu_limit``, resolved from
        # ``text_config.swiglu_limit`` in ``__init__``, so no caller can hand this
        # path a bound the checkpoint never declared.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # The SwiGLU wiring: ``silu`` is torch's own, the product is elementwise,
        # and both run in the kernel's fp32 return dtype so no precision is thrown
        # away between the projections. ``silu`` is also why ``down`` cannot fold
        # into either predecessor -- the product must be materialised here.
        activated = silu(gate) * up

        # The down projection re-enters the kernel, whose declared input dtype is
        # ``bfloat16``, so the fp32 intermediate is cast back to the activation
        # dtype. The cast is named rather than implicit because it is a real
        # precision step. The slice back is the last thing that happens: the caller
        # receives its own row count, never the padded one.
        return _unpad_rows(
            blockwise_fp8_mm(
                activated.to(hidden_states.dtype),
                down_proj_weight,
                down_proj_scale,
                prebuilt_scale_t=self._prepared_scale_operand("down_proj"),
            ),
            tokens,
        )

    # The shared expert's forward is the operand lookup and nothing else.
    # :meth:`shared_expert_mm` above is the whole compute path and takes its six
    # operands as arguments; this method is the ``nn.Module`` entry point, so it is
    # where "at the call" resolves to "off this module". It reads the three weights
    # the declaration tuple names and the three grids the loader attached beside
    # them, and hands them over.
    #
    # The grid names are derived through ``_sibling_scale_grid_name`` rather than
    # spelled, because that is the single definition of the convention the loader,
    # the prep loop and the grid publish all read.
    #
    # It raises on a missing grid and on nothing else. A missing grid is not
    # something the callee can check -- it receives grids, so an absent attribute
    # reaches it as ``AttributeError`` from inside a path that is not at fault.
    # Every extent agreement is the callee's, and a second check here is how two
    # authorities on one extent come to disagree.
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """This module's always-on contribution to one MoE layer.

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. The padding
                contract is :meth:`shared_expert_mm`'s and is documented there.
            quant_config: the resolved per-model quantisation policy, and the
                route selector. An argument rather than a field, matching the
                other methods of this family and
                :meth:`Glm5NextDenseMLP.forward`; no module in this file holds a
                policy.

        Returns:
            ``[T, H]`` fp32, exactly what :meth:`shared_expert_mm` returns.

        Raises:
            Glm5NextSharedExpertRouteError: when a scale grid the loader should
                have attached is absent. Refusing rather than running an
                unscaled matmul, which returns plausible numbers.
        """
        return self.shared_expert_mm(
            hidden_states, *self.scale_route_operands(), quant_config
        )

    #: The three projections this module routes, in the order
    #: :meth:`shared_expert_mm` and :meth:`prepare_scale_operands` both declare. A
    #: class attribute because the order is part of a contract between several
    #: readers and none of them should spell it a second time.
    SCALE_ROUTE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

    def scale_route_operands(self) -> tuple[torch.Tensor, ...]:
        """The six operands the shared route takes: three weights, then three grids.

        One definition of the lookup, because two callers need it: this module's
        own :meth:`forward` above, and :meth:`Glm5NextMoEBlock.forward`, which
        hands the same six to
        :meth:`Glm5NextMoEBlock.combine_routed_and_shared`. A second copy of the
        lookup in the parent is how the two come to disagree about which grid
        belongs to which weight.

        The order is the two consuming methods' own -- weights first, then grids,
        each triple in declaration order -- so a caller can splat this straight
        into either signature. A divergent order at one call site is a mis-wiring
        no shape check would see.

        Returns:
            ``(gate_w, up_w, down_w, gate_grid, up_grid, down_grid)``.

        Raises:
            Glm5NextSharedExpertRouteError: when a scale grid the loader should
                have attached is absent. The grid names are derived from the
                weight leaves by ``_sibling_scale_grid_name``, which is the single
                definition of that convention.
        """
        weights: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        for leaf in self.SCALE_ROUTE_LEAVES:
            grid_name = (
                Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)
            )
            grid = getattr(self, grid_name, None)
            if grid is None:
                raise Glm5NextSharedExpertRouteError(
                    f"{grid_name} is not on this module. The block-scale grids "
                    f"are plain attributes the weight loader attaches beside "
                    f"each declared weight, and this route consumes the grid at "
                    f"blockwise_fp8_mm.SCALE_BLOCK_SIZE granularity that "
                    f"retile_checkpoint_scale_grids publishes -- the checkpoint's "
                    f"own, and NOT the routed bank's "
                    f"larger block. Refusing rather "
                    f"than running an unscaled matmul, which returns plausible "
                    f"numbers."
                )
            weights.append(getattr(self, leaf))
            grids.append(grid)
        return (*weights, *grids)


# The shared-expert path's named refusal, at module level because an exception a
# caller catches belongs in the module namespace rather than nested in the class
# that raises it.
class Glm5NextSharedExpertRouteError(ValueError):
    """A shared-expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a call
    site that reaches the substrate's ``QuantizationType.none`` default by
    omission and computes a different function while every shape check passes.
    """


class Glm5NextMoEBlock(nn.Module):
    """The sparse MLP at ``mlp`` on layers at or past ``first_k_dense_replace``.

    Holds no parameter of its own: the map places every sparse-MLP parameter
    under ``mlp.experts`` or ``mlp.shared_experts``.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        world_size: int | None = None,
        ep_degree: int | None = None,
    ) -> None:
        super().__init__()
        # ``world_size`` is the tensor-parallel degree and ``ep_degree`` the
        # expert-parallel one. Both are optional and threaded to the routed bank
        # only; the bank divides its experts by the second and its intermediate
        # width by the first.
        self.experts = Glm5NextRoutedExperts(
            text_config, world_size=world_size, ep_degree=ep_degree
        )
        if text_config.n_shared_experts:
            self.shared_experts = Glm5NextSharedExperts(text_config)

    # The residual add lives here and not on either child: it needs the routed
    # half and the shared half, and neither ``Glm5NextRoutedExperts`` nor
    # ``Glm5NextSharedExperts`` owns the other. The compute sits on the child that
    # owns the weights, and the combination on their parent.
    #
    # "Added exactly once" is structural rather than asserted: this method contains
    # exactly one call to ``shared_expert_mm`` and exactly one ``+``, and it is the
    # only place in this file that adds a shared contribution to a routed one.
    # ``routed_output`` arrives as an argument precisely so this method composes
    # the two halves without owning either.

    def combine_routed_and_shared(
        self,
        routed_output: torch.Tensor,
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``routed_output + shared_expert(hidden_states)`` -- the layer output.

        Args:
            routed_output: ``[T, H]`` the routed experts' contribution, as the
                bank's ``block_quant_expert_mm`` returns it. Passed in rather
                than computed here.
            hidden_states: ``[T, H]`` the shared expert's own input -- the same
                pre-norm activations the routed half consumed.
            gate_proj_weight: forwarded verbatim to
                :meth:`Glm5NextSharedExperts.shared_expert_mm`, which documents
                every operand's layout and is the single authority for it.
            up_proj_weight: as above.
            down_proj_weight: as above.
            gate_proj_scale: as above.
            up_proj_scale: as above.
            down_proj_scale: as above.
            quant_config: as above.
            collector: when given, the shared half's own result is appended to it
                on the way through, for the layer dump. Nothing else changes.

        The SwiGLU bound is neither an argument here nor forwarded: the shared
        expert this class built holds it, so the bound reaches the clamp without
        passing through this method at all.

        Returns:
            ``[T, H]`` fp32 -- the sum, in the shared half's fp32 kernel dtype.
            The residual dtype for the decoder layer is the layer forward's.

        Raises:
            Glm5NextSharedExpertRouteError: when this class declares no shared
                expert, or when ``routed_output`` and the shared contribution
                disagree in extent. A silent broadcast is the failure this
                refuses: ``[T, H] + [1, H]`` and ``[T, H] + [T, 1]`` both
                broadcast without error and both compute a different layer.
        """
        # ``n_shared_experts == 0`` leaves the attribute undeclared, mirroring the
        # weight map's own condition. Checked rather than assumed, because
        # ``getattr`` on a missing module would raise ``AttributeError`` from
        # inside the shared path and read as a wiring bug rather than a config.
        shared_experts = getattr(self, "shared_experts", None)
        if shared_experts is None:
            raise Glm5NextSharedExpertRouteError(
                "combine_routed_and_shared was called on a block that declares "
                "no shared expert (n_shared_experts == 0), so there is no shared "
                "contribution to add. The routed output is already the layer "
                "output on such a block and calling this method is the bug."
            )

        # The one call to the shared path.
        shared_output = shared_experts.shared_expert_mm(
            hidden_states,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            gate_proj_scale,
            up_proj_scale,
            down_proj_scale,
            quant_config,
        )
        if collector is not None:
            collector.append(shared_output)

        # Extents compared exactly, before the add. ``torch`` would broadcast a
        # disagreeing extent silently and return a plausible tensor of the wrong
        # shape, which no downstream shape check would catch.
        if tuple(routed_output.shape) != tuple(shared_output.shape):
            raise Glm5NextSharedExpertRouteError(
                f"routed_output has shape {tuple(routed_output.shape)} and the "
                f"shared contribution has shape {tuple(shared_output.shape)}; "
                f"they must agree exactly. Refusing to add: torch would "
                f"broadcast these and compute a different layer without error."
            )

        # The one add. The shared contribution enters the layer output here and
        # nowhere else in this file.
        return routed_output + shared_output

    # The sparse MLP's forward is the three pieces in the reference's own order:
    # ``route_tokens`` on the bank, the bank's own forward, and
    # :meth:`combine_routed_and_shared` above. It authors no numerics and no
    # refusal; every extent is checked by the method that owns it.
    #
    # It takes the activations twice, and that is the load-bearing reading. The
    # reference's MoE block receives one already-normalised tensor and hands it to
    # the router and to both expert halves. This fork's router is fused -- the
    # RMSNorm is inside the router kernel -- so ``route_tokens`` must be handed the
    # pre-norm activations together with the norm's gain. Handing it the normalised
    # tensor would normalise twice and compute a different router; computing the
    # norm here for the experts instead would put a second authority on the layer's
    # own FFN norm. So the layer normalises once, and passes both what it started
    # with and what it produced.
    #
    # The router's gain is the decoder layer's ``post_attention_layernorm_weight``:
    # the checkpoint declares no router-norm tensor at all, so the gain the fused
    # kernel needs belongs to the layer, which is where this class's caller sits.
    # ``text_config`` arrives at the call for the same reason ``route_tokens``
    # takes it there.
    #
    # A block built with ``n_shared_experts == 0`` declares no shared module, and
    # :meth:`combine_routed_and_shared` refuses such a call by name, so this method
    # returns the routed half directly there rather than calling into a refusal.
    def forward(
        self,
        hidden_states: torch.Tensor,
        normed_hidden_states: torch.Tensor,
        *,
        router_gamma: torch.Tensor,
        text_config: Glm5NextTextConfig,
        quant_config: Glm5NextQuantConfig,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One sparse layer's MLP: route, run this rank's experts, add the shared.

        Args:
            hidden_states: ``[T, H]`` the layer's pre-norm activations, for the
                fused router only. See the section note above for why both forms
                arrive.
            normed_hidden_states: ``[T, H]`` the same activations after the
                layer's FFN norm -- what both expert halves consume.
            router_gamma: ``[H]`` or ``[1, H]`` the FFN norm's gain, which the
                fused router applies itself.
            text_config: the decoder config the routing hyperparameters live on,
                including the RMSNorm epsilon ``route_tokens`` resolves from it.
            quant_config: the resolved quantisation policy, the route selector.
            block_size: tokens per block, forwarded to the bank unread.
            moe_group: the MoE ``GroupCoordinator``, forwarded unread.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select, as an
                int64 device tensor (one graph for every rank) or a python int.

        Returns:
            ``[T, H]`` in the kernels' own dtype. The residual dtype is the layer
            forward's decision, not this method's.

        Raises:
            Glm5NextBlockQuantRouteError: whatever the bank refuses.
            Glm5NextSharedExpertRouteError: whatever the shared route refuses,
                including the extent disagreement the add checks.
        """
        # ``route_tokens`` declares ``[B, S, H]`` and the kernel flattens ``B`` and
        # ``S`` into one token axis, so a 2-D activation tensor is spelled
        # ``[1, T, H]`` here rather than reshaped inside the callee. Only the
        # affinities are consumed: the logits are the oracle's and the index set is
        # the affinities' own support.
        _logits, _expert_index, expert_affinities = self.experts.route_tokens(
            hidden_states.unsqueeze(0), router_gamma, text_config
        )
        if collector is not None:
            # Both forms of the weights: the whole table as the bank returns it,
            # scattered over every expert, and the chosen few gathered out of it.
            # The gather rides the collecting graph, so a run with no dump never
            # performs it.
            collector += [
                _logits,
                _expert_index,
                torch.gather(expert_affinities, -1, _expert_index.long()),
                expert_affinities,
            ]

        routed_output = self.experts(
            normed_hidden_states,
            expert_affinities,
            quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
            **({"collector": collector} if collector is not None else {}),
        )

        if collector is not None:
            collector.append(routed_output)

        shared_experts = getattr(self, "shared_experts", None)
        if shared_experts is None:
            return routed_output
        return self.combine_routed_and_shared(
            routed_output,
            normed_hidden_states,
            *shared_experts.scale_route_operands(),
            quant_config,
            collector=collector,
        )


class Glm5NextDenseMLP(nn.Module):
    """The dense MLP on the first ``first_k_dense_replace`` layers.

    Gate and up stay *separate* parameters, matching the weight map, which follows
    the fork's own dense precedent rather than fusing them.
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.intermediate_size = int(text_config.intermediate_size)
        # The checkpoint's SwiGLU bound, resolved here and not at the call -- the
        # same construction-time read :class:`Glm5NextSharedExperts` makes, on the
        # same scalar, from the same config field.
        #
        # The dense MLP carries it too because the reference has one MLP class,
        # constructed both as ``shared_experts`` and as the dense arm of
        # ``self.mlp``. That one class stores the bound and clamps with it, so the
        # dense MLP clamps with the same value and there is no separate dense
        # reading to get wrong. Unclamped, this path would compute a different
        # function from the checkpoint's on every token leaving the bound's box,
        # with no shape moving and nothing raising.
        self.swiglu_limit = float(text_config.swiglu_limit)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    #: Where :meth:`retile_checkpoint_scale_grids` records what it did, per
    #: projection. Its own attribute and not the shared expert's, so a reader who
    #: finds a record knows which module wrote it.
    DENSE_RETILE_HEALTH_ATTR = "_dense_mlp_retile_health"

    def retile_checkpoint_scale_grids(self) -> int:
        """Publish this module's grids at the kernel's granularity and set the frame.

        The loader delivers the checkpoint's own layout -- gate and up as
        ``[I, H]``, down as ``[H, I]`` -- while :meth:`forward` consumes the frame
        ``blockwise_fp8_mm`` multiplies in, so the frame is bridged here, once, on
        the load path. A caller that binds weights straight onto the module -- a
        fixture, say -- must bind what this step would have published.

        The whole body is :func:`_publish_compute_frame_operands`, shared with
        :class:`Glm5NextSharedExperts`: one definition of the frame rule rather
        than a mirrored copy that can drift. ``_run_load_time_preps`` enrols this
        class by ``hasattr``, so declaring the method is the whole enrolment.

        Returns:
            How many projections were published -- ``3`` when the extents are
            whole ``128`` blocks, ``0`` when they are not. Nothing is coarsened,
            since the kernel indexes the checkpoint's own grid, so the method name
            is a misnomer kept because it is public API. The transpose is not
            conditional on the count, and both frames are in the health record
            under :attr:`DENSE_RETILE_HEALTH_ATTR`.

        Raises:
            Glm5NextDenseMLPRouteError: if a weight or grid is not 2-D, or a grid
                is not at the checkpoint's own ``128``-tile granularity.
        """
        return _publish_compute_frame_operands(
            self, Glm5NextDenseMLPRouteError, self.DENSE_RETILE_HEALTH_ATTR
        )

    # The shared expert's ``shared_expert_mm`` is the same arithmetic on the same
    # kernel, and it is not called from here because it reads that class's prepared
    # scale operands off ``self`` and this class has no prep. Reaching into it
    # would either move that method or bind this forward to another module's
    # instance state.
    #
    # ``blockwise_fp8_mm``'s ``prebuilt_scale_t`` is optional and keyword-only, so
    # omitting it makes the call behave exactly as it did before the load-time prep
    # existed. ``_run_load_time_preps`` reaches a prep through
    # ``hasattr(type(module), "prepare_scale_operands")``, a per-class opt-in this
    # class does not take: adding it would change what the load path does.
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """One dense layer's MLP: three projections and a clamped SwiGLU.

        ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))``, where ``L`` is
        ``self.swiglu_limit``, the checkpoint's own bound resolved from the config
        when this object was built. Each of the three projections is a separate
        entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`, the
        dense blockwise route this fork uses for the dense MLP and the shared
        expert alike.

        The weights must arrive in the frame the kernel multiplies in -- gate and
        up ``[H, I]``, down ``[I, H]``, each with its ``128`` grid beside it --
        which is the transpose of the layout the loader delivers.
        :meth:`retile_checkpoint_scale_grids` bridges that once, on the load path,
        and this method refuses the loader's frame by name rather than transposing
        it per token.

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. ``T`` may be any
                positive count: the kernel tiles ``M`` over the PSUM partition
                axis and does not pad, so this method pads to a whole
                ``TILE_SIZE`` and slices the result back before returning, exactly
                as :meth:`Glm5NextSharedExperts.shared_expert_mm` does.
            quant_config: the resolved per-model quantisation policy, and the
                route selector. An argument rather than a field, because no module
                in this file holds a policy.

        Returns:
            ``[T, H]`` fp32 -- the kernel's own return dtype, not re-cast here.
            The layer forward decides the residual dtype.

        Raises:
            Glm5NextDenseMLPRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one this
                route consumes, when a scale grid the loader should have attached
                is absent, or when two operand extents contradict each other.
                Named rather than coerced, so a mis-wired call site fails where it
                is wrong instead of computing a different function.
        """
        from torch.nn.functional import silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm
        from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

        # ---- Route selection, the same two refusals the shared expert makes. --
        # There is no unquantised dense-MLP path at this site, and continuing
        # anyway is what would reach the substrate's ``QuantizationType.none``
        # default by omission.
        if not quant_config.is_block_quantized:
            raise Glm5NextDenseMLPRouteError(
                "Glm5NextDenseMLP.forward is the block-quant dense route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised dense-MLP path at "
                "this site."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextDenseMLPRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes the checkpoint's own ({TILE_SIZE}, {TILE_SIZE}) "
                f"blocks directly -- the dense kernel "
                f"indexes at that granularity and nothing is retiled -- and has "
                f"no path for any other checkpoint block shape."
            )

        # ---- The operands. The grid name is derived by the rule the prep loop
        # uses (``_sibling_scale_grid_name``) rather than spelled out here, so the
        # two cannot drift. The grids are plain attributes the weight loader
        # attaches and not declared parameters, which is why this is a ``getattr``.
        def scale_grid(leaf: str) -> torch.Tensor:
            name = f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_{FP8_SCALE_SUFFIX}"
            grid = getattr(self, name, None)
            if grid is None:
                raise Glm5NextDenseMLPRouteError(
                    f"{name} is not on this module. The block-scale grids are "
                    f"plain attributes the weight loader attaches beside each "
                    f"declared weight, and this route consumes the PUBLIC grid "
                    f"blockwise_fp8_mm declares. Refusing rather than running "
                    f"an unscaled matmul, which returns plausible numbers."
                )
            return grid

        # ---- Extents. Only the agreements the kernel cannot see: it reads its own
        # extents off each pair of operands separately, so nothing checks that gate
        # and up are the same shape or that down transposes them.
        if hidden_states.dim() != 2:
            raise Glm5NextDenseMLPRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        hidden = int(hidden_states.shape[1])
        gate_proj_weight = self.gate_proj_weight
        up_proj_weight = self.up_proj_weight
        down_proj_weight = self.down_proj_weight
        if tuple(gate_proj_weight.shape) != tuple(up_proj_weight.shape):
            raise Glm5NextDenseMLPRouteError(
                f"gate_proj_weight {tuple(gate_proj_weight.shape)} and "
                f"up_proj_weight {tuple(up_proj_weight.shape)} must have the "
                f"same [H, I] extents; they are multiplied elementwise after "
                f"the activation"
            )
        if gate_proj_weight.dim() != 2 or int(gate_proj_weight.shape[0]) != hidden:
            raise Glm5NextDenseMLPRouteError(
                f"gate_proj_weight must be [H={hidden}, I], got shape "
                f"{tuple(gate_proj_weight.shape)}"
            )
        intermediate = int(gate_proj_weight.shape[1])
        if tuple(down_proj_weight.shape) != (intermediate, hidden):
            raise Glm5NextDenseMLPRouteError(
                f"down_proj_weight must be [I={intermediate}, H={hidden}], got "
                f"shape {tuple(down_proj_weight.shape)}"
            )

        # ---- Pad to a whole tile, the same two lines as the shared expert's and
        # for the same reason: the kernel refuses a token count that is not a whole
        # tile and does not pad, so a one-token decode step pads here and is sliced
        # back at the single return below.
        hidden_states, tokens = _pad_tokens_to_tile(hidden_states, TILE_SIZE)

        # ---- The two parallel projections.
        gate = blockwise_fp8_mm(
            hidden_states, gate_proj_weight, scale_grid("gate_proj_weight")
        )
        up = blockwise_fp8_mm(
            hidden_states, up_proj_weight, scale_grid("up_proj_weight")
        )

        # ---- The clamp: the checkpoint's, not a guard this code invented. The
        # reference clamps gate from above only and up on both sides, and only then
        # multiplies them. The asymmetry is the reference's, and copying it as a
        # two-sided clamp on both would be a second wrong function.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # ---- The SwiGLU wiring: ``silu`` is torch's own, the product is
        # elementwise, and both run in the kernel's fp32 return dtype so no
        # precision is thrown away between the projections. ``silu`` is also why
        # ``down`` folds into neither predecessor.
        activated = silu(gate) * up

        # ---- The down projection re-enters the kernel, whose declared input dtype
        # is ``bfloat16``, so the fp32 intermediate is cast back to the caller's
        # activation dtype here. The slice back is the last thing that happens.
        return _unpad_rows(
            blockwise_fp8_mm(
                activated.to(hidden_states.dtype),
                down_proj_weight,
                scale_grid("down_proj_weight"),
            ),
            tokens,
        )


class Glm5NextDenseMLPRouteError(ValueError):
    """A dense-MLP call this route refuses, named rather than coerced.

    At module level rather than nested in the class that raises it, because an
    exception a caller catches belongs in the module namespace.

    Raised in preference to continuing, because each failure it closes returns
    plausible numbers rather than an error: an unquantised route reaches the
    substrate's ``QuantizationType.none`` default by omission, a missing scale
    grid runs an unscaled matmul, and contradicting extents multiply tensors the
    reference never multiplies.
    """


def _build_mlp(text_config: Glm5NextTextConfig, layer_idx: int) -> nn.Module:
    """Dense below ``first_k_dense_replace``, sparse at and above it.

    The same predicate the weight map uses, so the two sides cannot disagree
    about which layers carry experts.
    """
    if layer_idx < int(text_config.first_k_dense_replace):
        return Glm5NextDenseMLP(text_config)
    return Glm5NextMoEBlock(text_config)


# ---------------------------------------------------------------------------
# The KDA (``linear_attention``) half. ``Glm5NextKDALayer`` is the decoder layer
# and the gated-delta module it holds sits at the map's ``self_attn`` path.
# ``linear_attn`` survives below only as ``CACHE_NAME_SUFFIX``, which names the
# KV-cache entry and is not a module path.
# ---------------------------------------------------------------------------


def _per_request_entries(part):
    """One entry per request: a tuple's items, a stacked tensor's rows, or the whole.

    A one-row tensor stays whole, so the batch form and the pinned one-sequence form
    reach the kernels as the same shape.
    """
    if isinstance(part, (tuple, list)):
        return tuple(part)
    if torch.is_tensor(part) and part.dim() == 1 and int(part.shape[0]) > 1:
        return tuple(torch.unbind(part))
    return (part,)


def _per_request_row_entries(part, one_sequence_dim: int):
    """One entry per request for a row operand, or ``None`` when it was not passed.

    The per-request form carries one more axis than the one-sequence form --
    ``[requests, 1]`` beside ``[1]``, and ``[requests, T, 1]`` beside ``[T, 1]``
    -- so the split is read off the rank rather than guessed, and each entry this
    yields already has the shape one sequence's operand has.
    """
    if part is None:
        return None
    if torch.is_tensor(part) and part.dim() > one_sequence_dim:
        return tuple(torch.unbind(part, 0))
    return (part,)


def _int64_scalar(value: torch.Tensor | int, device: torch.device) -> torch.Tensor:
    """``value`` as a 0-d int64 tensor on ``device``; a number is factory-built so a trace keeps it fake."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.int64).reshape(())
    # A number handed to ``as_tensor`` on the meta device comes back as a real tensor under a
    # fake trace, and the next operator refuses it beside fake ones; ``full`` is dispatched.
    return torch.full((), int(value), dtype=torch.int64, device=device)


def _start_is_zero(start_position: torch.Tensor | int, device: torch.device):
    """``start_position == 0`` as a 0-d bool tensor, never as a python bool."""
    return _int64_scalar(start_position, device) == 0


class Glm5NextKDAAttention(nn.Module):
    """Gated-delta linear attention at ``self_attn``.

    Every parameter name here is the weight map's ``KDA_PROJECTIONS`` plus
    ``KDA_BARE_LEAVES``, as ``weight_loaders_fp8.py``'s ``_add_kda_attention``
    emits them: thirteen ``<leaf>_weight`` names and the two bare state tensors
    ``A_log`` and ``dt_bias``, fifteen in all and not one scale companion. There
    is no conv1d bias of any spelling in the checkpoint index.

    A linear-attention layer holds a *recurrent state*, not a key/value history.
    ``LayerSpec`` has no vocabulary for that state, so the spec this layer reports
    describes its per-head state extent in the existing fields and asserts nothing
    about recurrent-state layout.
    """

    #: The map's module path for this family, and therefore the suffix of the
    #: ``LayerSpec.name`` this layer reports.
    CACHE_NAME_SUFFIX = "linear_attn"

    def __init__(self, text_config: Glm5NextTextConfig, world_size: int) -> None:
        super().__init__()
        self.num_heads = _linear_attn_field(text_config, "num_heads")
        self.head_dim = _linear_attn_field(text_config, "head_dim")
        self.short_conv_kernel_size = _linear_attn_field(
            text_config, "short_conv_kernel_size"
        )

        # Head-count sharding follows the fork's KV-spec convention (per-rank
        # counts); the state width per head does not shard.
        self.num_kv_heads_per_rank = _per_rank(self.num_heads, world_size)
        self.head_size = self.head_dim
        self.cache_dtype = _resolve_kda_state_dtype(text_config)
        # The gate's lower bound is the checkpoint's, read through the same
        # checked accessor as the rest of ``linear_attn_config`` so a missing key
        # raises here instead of reaching the gate kernel as a default. The kernel
        # requires it and holds no copy of it (``gate_clamp.py``).
        self.gate_lower_bound = float(
            _linear_attn_field(text_config, "gate_lower_bound")
        )
        # The decoder's RMSNorm epsilon, from the config rather than from a local
        # default.
        self.rms_norm_eps = float(text_config.rms_norm_eps)
        # ``NeuronConfig.kda_state_chunk_size`` is declared a tuning dial with
        # "None = let the layer choose" (``neuron_config.py``), and
        # ``LayerSpec.chunk_size`` already exists, so the dial is
        # passed through rather than dropped. None by default.
        self.cache_chunk_size = getattr(
            text_config.neuron_config, "kda_state_chunk_size", None
        )

        # The four recurrent-state values ``get_kv_spec`` reports for this family.
        # They come from this layer's own state geometry.
        #
        # Both pairs are derived from vLLM's own calculators and neither is written
        # as a literal. The conv state's extent order is chosen by the environment
        # (``VLLM_SSM_CONV_STATE_LAYOUT``), so a hand-written pair would be right
        # under one layout and silently transposed under the other -- and a
        # transposition survives a byte reconciliation. The import is function-local
        # because this module holds no vLLM import at module level.
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
            get_conv_state_layout,
            is_conv_state_dim_first,
        )

        conv_state_shape, recurrent_state_shape = (
            MambaStateShapeCalculator.kda_state_shape(
                tp_world_size=world_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                conv_kernel_size=self.short_conv_kernel_size,
            )
        )
        self.kda_conv_state_shape = tuple(conv_state_shape)
        self.kda_recurrent_state_shape = tuple(recurrent_state_shape)
        # ``kda_state_dtype(model_dtype, "auto")`` returns
        # ``(state_dtype, torch.float32)`` and resolves ``"auto"`` to the dtype
        # passed in (``vllm/utils/torch_utils.py``), so passing
        # ``self.cache_dtype`` keeps ``NeuronConfig.kda_state_dtype``'s override
        # honoured on the conv carrier while the recurrent carrier's float32
        # still comes from the vendor rather than from a local constant.
        (
            self.kda_conv_state_dtype,
            self.kda_recurrent_state_dtype,
        ) = MambaStateDtypeCalculator.kda_state_dtype(self.cache_dtype, "auto")

        # Which axis of ``kda_conv_state_shape`` is the channel axis, read from
        # the same authority that ordered it. The forward slices the conv carrier,
        # so it must not infer the order from the extents:
        # ``is_conv_state_dim_first`` is the predicate the vendor's own orienting
        # helper branches on, so this is the same reading and not a second
        # inference from the extents. The layout string is kept beside it because a
        # transcript naming ``"sd"`` is checkable and a bare bool is not.
        self.kda_conv_state_layout = get_conv_state_layout()
        self.kda_conv_state_dim_first = is_conv_state_dim_first()

        # The map's fifteen, in the map's own order: ``KDA_PROJECTIONS`` as
        # ``<leaf>_weight``, then ``KDA_BARE_LEAVES`` with no suffix at all.
        _declare_parameters(
            self,
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "b_proj_weight",
            "f_a_proj_weight",
            "f_b_proj_weight",
            "g_a_proj_weight",
            "g_b_proj_weight",
            "q_conv1d_weight",
            "k_conv1d_weight",
            "v_conv1d_weight",
            "o_norm_weight",
            "o_proj_weight",
            "A_log",
            "dt_bias",
        )

    # ── the conv carrier's extent order, handled in one place ─────────────
    def _conv_history(self, conv_state: torch.Tensor) -> torch.Tensor:
        """The carried conv rows as ``[kernel_size - 1, channels]``, float32.

        The stored order is the environment's, so it is converted here once
        rather than assumed at the call site.
        """
        rows = conv_state.t() if self.kda_conv_state_dim_first else conv_state
        return rows.to(torch.float32)

    def _store_conv_history(
        self, conv_state: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Write ``rows`` (``[kernel_size - 1, channels]``) back in stored order."""
        value = rows.t() if self.kda_conv_state_dim_first else rows
        conv_state.copy_(value.to(conv_state.dtype))

    def _resolve_chunk_size(self, chunk_size: int | None) -> int:
        """The chunk width the chunked kernels are entered with.

        ``NeuronConfig.kda_state_chunk_size`` is a dial whose ``None`` means "let
        the layer choose", so this is where the layer chooses. The choice is
        derived from the two declared bounds it has to satisfy rather than picked:
        the intra-chunk kernel needs a power of two in ``[2, 128]``, and both
        chunked kernels refuse a chunk-local cumulative gate above
        ``GATE_CUMSUM_ABS_LIMIT``. The gate this layer produces lies in
        ``[gate_lower_bound, 0]``, so one chunk's cumulative gate cannot exceed
        ``chunk * |gate_lower_bound|`` -- and the largest power of two that keeps
        that product inside the limit is the widest chunk that cannot be refused
        for gate range at any input.
        """
        from vllm_neuron.functional.kda.chunked_recurrence import (
            GATE_CUMSUM_ABS_LIMIT,
            MAX_TILE,
        )

        requested = chunk_size if chunk_size is not None else self.cache_chunk_size
        if requested is not None:
            return int(requested)

        bound = abs(self.gate_lower_bound)
        widest = 2
        candidate = 2
        while candidate <= MAX_TILE:
            if bound * candidate <= GATE_CUMSUM_ABS_LIMIT:
                widest = candidate
            candidate *= 2
        return widest

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_state: torch.Tensor | tuple[torch.Tensor, ...],
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...],
        is_prefill: bool,
        start_position: torch.Tensor | int = 0,
        chunk_size: int | None = None,
        real_tokens: torch.Tensor | int | None = None,
        row_mask: torch.Tensor | None = None,
        _return_fp32_partial: bool = False,
    ) -> torch.Tensor:
        """One KDA layer's gated-delta linear attention over ``[T, hidden]``.

        The five kernels are entered in this order: the short convolution, the
        gate clamp, then either the two chunked recurrence kernels or the
        single-token decode kernel. This function composes them and implements
        none of their arithmetic.

        Args:
            hidden_states: ``[T, hidden]``. One rank's slice of one step's
                tokens.
            conv_state: this layer's conv carrier from the runner bank, shaped as
                ``get_kv_spec`` reports it. Read for the left context of the
                convolution and written in place with the new tail.
            recurrent_state: this layer's recurrent carrier from the runner bank,
                ``[heads, V, K]``. Index 1 is the value extent and index 2 the
                key extent -- the orientation the chunked kernel's
                ``final_state`` and the decode kernel's ``state`` both use.
                ``V == K == head_dim`` here, so that choice cannot be read off
                the shape and is stated rather than implied. Written in place.
            is_prefill: whether this call is a prefill. It selects the recurrence
                route and nothing else.
            start_position: how many tokens of this sequence are already
                computed. Zero means this call opens the sequence; above zero on
                a prefill means this call continues one, which is what decides
                whether the recurrence enters with the carrier's state.
            chunk_size: overrides the resolved chunk width.
            real_tokens: how many of the ``T`` rows carry a token of this
                sequence, as an ``int32`` tensor -- or ``[requests, 1]``, one row
                per request. ``None`` says every row does.
            row_mask: ``[T, 1]`` float, one for a row that carries a token and
                zero for a padding row -- or ``[requests, T, 1]``, one mask per
                request. Passed together with ``real_tokens``, or neither is
                passed.

        Concurrent decode keeps per-request arithmetic and combines FP32 output
        partials for one TP reduction. Concurrent prefill remains unsupported.
        ``_return_fp32_partial`` defers the reduction and cast to the outer call.

        Returns:
            ``[T, hidden]`` at the input dtype.

        A concurrent decode arrives with the five per-request arguments one per
        request, in the batch's order. The two state carriers arrive as tuples of
        views, because each request's states live at its own slot of the bank and
        stacking them would copy. The position arrives as one int32 tensor with a
        row per request, and each row operand as one tensor carrying a leading
        request axis, because a host number here becomes a constant of the
        captured graph. A row operand's per-request form is its one-sequence form
        with one more axis, which is what makes it unmistakable: a
        ``[requests, 1]`` mask would also read as one sequence of ``requests``
        rows. Each request is then served one at a time by this same method, on
        its own entry of all five, and a concurrent prefill is refused, because
        the carrier says nothing about where one request's tokens end.

        A prefill enters with a zero state only when it opens the sequence. A
        prompt longer than one batch of tokens is prefilled in segments, and the
        second segment continues the recurrence the first left, so the entering
        state is the carrier's whenever ``start_position`` is above zero. At
        position zero it is zero whatever the slot holds, because a fresh
        sequence must not inherit the last one's state. The position is enough to
        decide that because the runner refuses a step that does not continue the
        sequence its cursor names.

        Whole chunks go through the chunked pair and the remainder walks: a
        prefill of ``T = n * chunk + r`` tokens takes one intra-chunk and one
        inter-chunk call for the ``n`` whole chunks together -- their inputs carry
        the chunk axis -- and then ``r`` single-token decode calls. A decode call
        takes no chunked call at any token count.

        A padding row leaves the state where it found it. A step arrives padded up
        to its bucket, and every operand keeps that padded width, because that
        width is what a captured graph was compiled for. Two things keep the
        padding rows out of the sequence's state: the decay gate and the update
        rate are multiplied by ``row_mask``, which makes a padding row's decay one
        and its update zero -- the recurrence's own identity, exactly rather than
        approximately -- and the convolution's history is taken from the last real
        rows by ``index_select`` rather than from the bucket's tail. Neither reads
        a value off a tensor and neither changes a shape.
        """
        from vllm_neuron.functional.kda.chunked_recurrence import (
            kda_inter_chunk,
            kda_intra_chunk,
        )
        from vllm_neuron.functional.kda.decode_state import kda_decode_step
        from vllm_neuron.functional.kda.depthwise_conv1d import depthwise_conv1d
        from vllm_neuron.functional.kda.gate_clamp import (
            MAX_TILE as GATE_MAX_TILE,
            kda_gate_clamp,
        )

        if hidden_states.dim() != 2:
            raise ValueError(
                f"hidden_states must be [tokens, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        # Each request carries views of its own bank slot. Recursion writes
        # directly to those views and returns its unreduced FP32 output.
        # The position arrives as one int32 tensor with a row per request, and its
        # rows are taken with ``unbind``, a tensor operation: reading the value here
        # to split it would be a host read of tensor data inside a traced region,
        # which is what pins a captured graph to the position it was captured at.
        states = tuple(_per_request_entries(part)
                       for part in (conv_state, recurrent_state, start_position))
        counts = {len(part) for part in states}
        if len(counts) != 1:
            raise ValueError(
                f"the two state carriers and the position describe the same requests, "
                f"so they arrive in equal numbers; this call carries "
                f"{len(states[0])} conv, {len(states[1])} recurrent and "
                f"{len(states[2])} position entry(ies)"
            )
        requests = len(states[0])
        # The concurrent refusals come before the operands are read, so a call this
        # layer cannot serve at all is refused for what it is rather than for an
        # operand count derived from it.
        if requests > 1:
            if is_prefill:
                raise ValueError(
                    f"this call prefills {requests} requests together, and a prefill "
                    f"carries a different number of tokens for each of them; the "
                    f"carrier says nothing about where one request's tokens end and "
                    f"the next one's begin, so serving it would run every request "
                    f"over the whole batch's tokens. Concurrent DECODE is what this "
                    f"layer serves"
                )
            if int(hidden_states.shape[0]) != requests:
                raise ValueError(
                    f"a decode step advances each sequence by one token, so this call "
                    f"carries one token per request; it holds "
                    f"{int(hidden_states.shape[0])} token(s) for {requests} request(s)"
                )
        # The row operands are stacked rather than tupled, for the same reason the
        # states are tupled: a tuple carries what must not be copied, and these two
        # are built for the step rather than pointing into the bank. Their
        # per-request form is therefore one leading axis on the one-sequence form,
        # which no one-sequence operand can be mistaken for.
        rows = tuple(_per_request_row_entries(part, dim)
                     for part, dim in ((real_tokens, 1), (row_mask, 2)))
        for name, part in zip(("real_tokens", "row_mask"), rows):
            if part is not None and len(part) != requests:
                raise ValueError(
                    f"{name} arrives beside the states, one entry per request, and this "
                    f"call carries {requests} request(s) against {len(part)} {name} "
                    f"entry(ies); one request's padding is not another's"
                )
        reals = rows[0] if rows[0] is not None else (None,) * requests
        masks = rows[1] if rows[1] is not None else (None,) * requests
        if requests > 1:
            # Keep every request's arithmetic shape and state views unchanged.
            # Only the output collective shares a request dimension.
            partials = [
                self.forward(
                    hidden_states[index : index + 1],
                    conv_state=conv,
                    recurrent_state=recurrent,
                    is_prefill=False,
                    start_position=position,
                    chunk_size=chunk_size,
                    real_tokens=real,
                    row_mask=mask,
                    _return_fp32_partial=True,
                )
                for index, (conv, recurrent, position, real, mask) in enumerate(
                    zip(*states, reals, masks)
                )
            ]
            attn_out = torch.cat(partials, dim=0)
            if _return_fp32_partial:
                return attn_out
            return self._finish_output(
                attn_out, hidden_states.dtype, num_requests=requests
            )
        conv_state, recurrent_state, start_position = (part[0] for part in states)
        real_tokens, row_mask = reals[0], masks[0]
        tokens = int(hidden_states.shape[0])
        heads = int(self.num_kv_heads_per_rank)
        kdim = int(self.head_dim)
        width = heads * kdim
        state_rows = int(self.short_conv_kernel_size) - 1

        expected_recurrent = (heads, kdim, kdim)
        if tuple(recurrent_state.shape) != expected_recurrent:
            raise ValueError(
                f"recurrent_state {tuple(recurrent_state.shape)} must be "
                f"{expected_recurrent} for this rank's geometry"
            )
        if tuple(conv_state.shape) != tuple(self.kda_conv_state_shape):
            raise ValueError(
                f"conv_state {tuple(conv_state.shape)} must be the shape "
                f"get_kv_spec reports, {tuple(self.kda_conv_state_shape)}"
            )
        if (row_mask is None) != (real_tokens is None):
            raise ValueError(
                "real_tokens and row_mask are one fact in two operands -- which rows "
                "carry a token -- and this call passed one of them; a masked "
                "recurrence over an unmasked history would keep the padding rows"
            )
        if row_mask is not None and tuple(row_mask.shape) != (tokens, 1):
            raise ValueError(
                f"row_mask {tuple(row_mask.shape)} must be [{tokens}, 1], one row "
                f"for each row of hidden_states"
            )

        x = hidden_states.to(torch.float32)

        def project(weight: torch.Tensor) -> torch.Tensor:
            return x @ weight.to(torch.float32).t()

        q_in = project(self.q_proj_weight)
        k_in = project(self.k_proj_weight)
        v_in = project(self.v_proj_weight)
        raw_beta = project(self.b_proj_weight)
        # Both gates are low-rank: a bottleneck projection, then an expansion
        # back to the head width. The bottleneck width is read from the weights
        # rather than from the config, because the config declares no such field.
        raw_gate = (x @ self.f_a_proj_weight.to(torch.float32).t()) @ (
            self.f_b_proj_weight.to(torch.float32).t()
        )
        out_gate = (x @ self.g_a_proj_weight.to(torch.float32).t()) @ (
            self.g_b_proj_weight.to(torch.float32).t()
        )

        # A sequence that has computed nothing enters with a zero state, and this is
        # the one predicate that says so. A sequence at position 0 carries no
        # history, and the slot it was handed may still hold the bytes of whichever
        # request held it before -- nothing empties the recurrent banks, because an
        # eager write on a buffer whose storage is shared is refused by the runtime.
        # So both state reads below select zero here instead. It is computed once,
        # outside the head loop, and it is a tensor comparison: a python branch on
        # the position would compile the choice made at capture time into every
        # later step.
        #
        # It is read on every leg, because the position is what says whether a
        # sequence is opening and the number of tokens in the step is not. A
        # one-token prompt computes its whole prompt in a step whose query length
        # does not exceed the decode threshold, and a preempted request resumes at
        # position 0 however many tokens it is given; both are opening, and a
        # predicate the leg switched off would hand exactly those two the previous
        # owner's history.
        opening = _start_is_zero(start_position, conv_state.device)

        # --- 1: the short convolution, one call for q, k and v ---------------
        # The three streams are convolved together as one channel block, which is
        # the same channel extent the state calculator reports
        # (``conv_dim = proj + 2 * proj_k``). Padding is carried by the state rather
        # than by the kernel, which refuses non-zero width padding.
        conv_in = torch.cat((q_in, k_in, v_in), dim=-1)
        history = self._conv_history(conv_state)
        history = torch.where(opening, torch.zeros_like(history), history)
        padded = torch.cat((history, conv_in), dim=0)
        channels = 3 * width
        img = (
            padded.t().contiguous().reshape(1, channels, 1, state_rows + tokens)
        )
        filt = torch.cat(
            (
                self.q_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
                self.k_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
                self.v_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
            ),
            dim=0,
        ).contiguous()
        conv_out = depthwise_conv1d(img, filt)
        conv_out = conv_out.reshape(channels, tokens).t()
        conv_out = torch.nn.functional.silu(conv_out)
        q_conv, k_conv, v_conv = conv_out.split(width, dim=-1)
        # The history is the left context the next step convolves with, so it is the
        # last rows the sequence holds and not the last rows of the bucket. The
        # index is derived from the real length rather than read from it, and a step
        # whose real length is below the history width keeps the older rows it still
        # needs. On an opening chunk -- where the history rows are zeroed -- a chunk
        # of fewer than ``state_rows`` rows leaves a window whose leading rows are
        # zero, which is the padding a sequence with no history has, rather than the
        # previous owner's rows or this chunk's rows repeated.
        real_length = _int64_scalar(
            tokens if real_tokens is None else real_tokens, padded.device
        )
        history_index = torch.arange(state_rows, device=padded.device) + real_length
        self._store_conv_history(conv_state, padded.index_select(0, history_index))

        # --- 2: the gate clamp, one call per head per token tile --------------
        # The kernel takes one head per call: it refuses an ``a_log`` holding more
        # than one value, because the decay rate is per head while the bias and the
        # gate are per key channel.
        #
        # It also refuses more than ``GATE_MAX_TILE`` tokens in one call, because
        # both of its axes pass through a transpose that serves that width. So a
        # prompt longer than one tile is handed over in tiles and reassembled here,
        # in the caller, rather than the kernel growing a quiet fallback.
        #
        # Tiling is exact here, not approximate. The gate applies a per-channel
        # bias, one scalar decay rate and a sigmoid, with no reduction along the
        # token axis, so tile boundaries cannot move a value: a tiled call and a
        # whole call agree bit for bit.
        a_log = self.A_log.to(torch.float32).reshape(-1)
        dt_bias = self.dt_bias.to(torch.float32).reshape(-1)

        def clamp_one_head(h: int) -> torch.Tensor:
            """One head's gate over the whole prompt, in tiles the kernel accepts."""
            span = slice(h * kdim, (h + 1) * kdim)
            head_bias = dt_bias[span]
            head_decay = a_log[h]
            tiles = [
                kda_gate_clamp(
                    raw_gate[start : start + GATE_MAX_TILE, span].contiguous(),
                    head_decay,
                    bias=head_bias,
                    lower=self.gate_lower_bound,
                )
                for start in range(0, tokens, GATE_MAX_TILE)
            ]
            # A prompt that already fits is returned as the single tile it is, so it
            # still costs exactly one call and no concatenation.
            return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)

        gate_parts = [clamp_one_head(h) for h in range(heads)]
        beta = torch.sigmoid(raw_beta)
        # The mask goes on after the clamp and after the sigmoid, on the two values
        # the recurrence reads per row: a zero log-decay is a decay of one and a zero
        # rate is no update, so a padding row hands the state straight through.
        # Before either non-linearity a zero would mean the clamp's own floor and a
        # rate of a half instead.
        if row_mask is not None:
            gate_parts = [part * row_mask for part in gate_parts]
            beta = beta * row_mask

        # --- 3 to 5: the recurrence, per head --------------------------------
        chunk = self._resolve_chunk_size(chunk_size)
        n_chunks = tokens // chunk if is_prefill else 0
        chunked = n_chunks * chunk
        # An allocation on the traced path follows the activation it is combined
        # with. A bare factory call takes the default device, so under a capture that
        # holds this module and its inputs on ``meta`` this buffer would land on the
        # host and the first arithmetic against a parameter would meet two devices.
        # ``q_conv`` is the convolution's own output, which every value written into
        # this buffer is derived from.
        core = torch.empty(tokens, width, dtype=torch.float32, device=q_conv.device)
        for h in range(heads):
            span = slice(h * kdim, (h + 1) * kdim)
            q_h = q_conv[:, span].contiguous()
            k_h = k_conv[:, span].contiguous()
            v_h = v_conv[:, span].contiguous()
            gk_h = gate_parts[h]
            beta_h = beta[:, h].contiguous()

            # The entering state, chosen without reading the position. A sequence
            # starting at 0 enters with a zero state and a continuation enters with
            # its carried one, under the one predicate computed above -- the same one
            # the convolution's history is selected with, because both states belong
            # to one request and a second signal for one fact is a place for the two
            # to disagree. ``torch.where`` makes the choice on device, so one graph
            # serves position 0 and position N.
            carried = recurrent_state[h].to(torch.float32)
            state = torch.where(opening, torch.zeros_like(carried), carried)

            if chunked:
                shape = (n_chunks, chunk, kdim)
                intra = kda_intra_chunk(
                    q_h[:chunked].reshape(shape).contiguous(),
                    k_h[:chunked].reshape(shape).contiguous(),
                    v_h[:chunked].reshape(shape).contiguous(),
                    beta_h[:chunked].reshape(n_chunks, chunk).contiguous(),
                    gk_h[:chunked].reshape(shape).contiguous(),
                )
                inter = kda_inter_chunk(
                    intra.kg,
                    intra.w,
                    intra.u,
                    gk_h[:chunked].reshape(shape).contiguous(),
                    q_h[:chunked].reshape(shape).contiguous(),
                    intra.aqk,
                    state=state,
                )
                core[:chunked, span] = inter.o.reshape(chunked, kdim)
                state = inter.final_state

            for t in range(chunked, tokens):
                step = kda_decode_step(
                    state,
                    q_h[t : t + 1],
                    k_h[t : t + 1],
                    v_h[t : t + 1],
                    beta_h[t].reshape(1, 1),
                    gk_h[t : t + 1],
                )
                # Both sides stay rank 2, like the chunked write above. The step
                # kernel already returns ``[1, V]``, and flattening it made the
                # source rank 1 against a rank-2 target slice: eager torch
                # broadcasts that, and the graph compiler refuses it.
                core[t : t + 1, span] = step.o.reshape(1, kdim)
                state = step.state

            recurrent_state[h] = state.to(recurrent_state.dtype)

        # --- gated output norm, then the output projection -------------------
        # ``rmsnorm(core) * sigmoid(out_gate)``, normalised over the head extent
        # because the norm gain is one value per key channel. The reference
        # builds this half as a gated RMSNorm whose activation is sigmoid
        # (``kimi_gdn_linear_attn.py``), which is why the raw gate is passed
        # through a sigmoid here and not through a silu.
        shaped = core.reshape(tokens, heads, kdim)
        variance = shaped.pow(2).mean(dim=-1, keepdim=True)
        shaped = shaped * torch.rsqrt(variance + self.rms_norm_eps)
        shaped = shaped * self.o_norm_weight.to(torch.float32).reshape(1, 1, kdim)
        shaped = shaped * torch.sigmoid(
            out_gate.reshape(tokens, heads, kdim)
        )
        attn_out = shaped.reshape(tokens, width) @ (
            self.o_proj_weight.to(torch.float32).t()
        )
        if _return_fp32_partial:
            return attn_out
        return self._finish_output(attn_out, hidden_states.dtype)

    def _finish_output(
        self, attn_out: torch.Tensor, output_dtype: torch.dtype, *, num_requests: int = 1
    ) -> torch.Tensor:
        """Reduce fresh row-parallel partials in FP32, then restore the input dtype."""
        attn_out = _reduce_tp_rows(attn_out, num_requests=num_requests)
        return attn_out.to(output_dtype)


class Glm5NextKDALayer(nn.Module):
    """Decoder layer on the ``linear_attention`` half of the hybrid stack."""

    #: Attribute the family's attention module is bound to -- the weight map's own
    #: module path, built as ``f"{param_prefix}.self_attn"``, because
    #: ``declared_parameter_names`` builds its paths from ``named_modules()``. Not
    #: the same string as ``Glm5NextKDAAttention.CACHE_NAME_SUFFIX``, which stays
    #: ``linear_attn`` and names a KV-cache entry rather than a module.
    ATTENTION_ATTR = "self_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = KDA_LAYER_TYPE
        _declare_parameters(
            self,
            "input_layernorm_weight",
            "post_attention_layernorm_weight",
            # The six mHC weights sit flat on the layer because that is where the
            # weight map puts them: ``MHC_LEAVES``, emitted for every layer by an
            # unconditional ``_add_mhc``, with no ``.weight`` leaf, no scale
            # companion and no submodule.
            #
            # :meth:`bind_hyper_connection_sites` hands the six loaded tensors to
            # two ``Glm5NextHyperConnection`` instances after the load and holds
            # those instances in a plain dict, so no leaf moves under a submodule
            # attribute and the map equality is untouched.
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        )
        self.self_attn = Glm5NextKDAAttention(text_config, world_size)
        self.mlp = _build_mlp(text_config, layer_idx)
        self.rms_norm_eps = float(text_config.rms_norm_eps)

    def bind_hyper_connection_sites(
        self, text_config: Glm5NextTextConfig, device: torch.device
    ) -> int:
        """Give this layer's two mHC sites the six tensors the load brought.

        One body for both layer families, in
        :func:`_bind_hyper_connection_sites`, so the linear-attention and
        sparse-attention halves cannot drift apart in a rule neither of them owns.
        ``_run_load_time_preps`` is the single production caller and reaches this
        method through the same ``hasattr`` gate it uses on the load-time preps.
        """
        return _bind_hyper_connection_sites(self, text_config, device)

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-attention RMSNorm: ``x / sqrt(mean(x**2) + eps) * gain``.

        The epsilon is the checkpoint's ``rms_norm_eps``, resolved at
        construction.
        """
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        normed = normed * self.input_layernorm_weight.to(torch.float32)
        return normed.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_state: torch.Tensor | tuple[torch.Tensor, ...],
        recurrent_state: torch.Tensor | tuple[torch.Tensor, ...],
        is_prefill: bool,
        start_position: torch.Tensor | int = 0,
        chunk_size: int | None = None,
        real_tokens: torch.Tensor | int | None = None,
        row_mask: torch.Tensor | None = None,
        streams: torch.Tensor | None = None,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The linear-attention half, mixed either by mHC or by a plain add.

        The ``streams`` keyword picks the route.

        * ``streams`` absent: pre-norm, attention, plain residual add. The draft
          head needs this route, and so does every direct caller.
        * ``streams`` present: the four-stream mHC pair runs around the same
          attention half -- collapse the streams, norm, attend, then re-mix --
          which is what the target model does.

        The branch refuses both ways, in :func:`_mhc_attention_site`, so the
        optional keyword cannot silently reinstate the one-stream network. The
        attention half is written once, as ``attention_half`` below, and both
        routes run that one closure: a second copy of the attention call is how
        the two routes come to disagree about what they wrap.

        ``self.mlp`` is not called here. The feed-forward half is
        ``Glm5NextModel._ffn_half``'s, and the FFN mHC site is composed there; the
        six mHC weights sit flat on this layer and the two sites are bound to it
        after the load, by :meth:`bind_hyper_connection_sites`.

        Args:
            streams: ``[T, S, H]`` residual streams, or ``None`` for the
                one-stream route. Every other argument is the attention module's,
                passed through unchanged -- including the per-request tuple form
                of the two carriers and the position, which this method neither
                reads nor splits; see :meth:`Glm5NextKDAAttention.forward` for
                what they mean.

        Returns:
            ``[T, H]`` on the one-stream route -- the input dtype, unchanged. On
            the streams route, ``[T, S, H]`` in the streams' dtype: the kernels
            compute in fp32 and :meth:`Glm5NextHyperConnection.mhc_post` casts the
            mix back, following the reference. So a bfloat16 carrier stays
            bfloat16 through this layer, which is what the dense and MoE kernels
            downstream ask for.

        Raises:
            Glm5NextHyperConnectionError: from :func:`_mhc_attention_site` when
                the route and this layer's weights disagree, or from the kernels
                on a geometry they cannot serve.
        """

        num_requests = (
            len(conv_state) if not is_prefill and isinstance(conv_state, tuple) else 1
        )

        def attention_half(single_stream: torch.Tensor) -> torch.Tensor:
            if num_requests > 1 and single_stream.dtype == torch.bfloat16:
                if single_stream.dim() != 2 or single_stream.shape[0] != num_requests:
                    raise ValueError(
                        "concurrent BF16 KDA input norm requires "
                        "[num_requests, hidden] rows"
                    )
                # Per-row norm and the FP32 join preserve the singleton BF16
                # boundary in compiled decode, as the captured-state replay verified.
                normed = torch.cat(
                    [
                        self._input_norm(single_stream[row : row + 1]).to(torch.float32)
                        for row in range(num_requests)
                    ],
                    dim=0,
                ).to(single_stream.dtype)
            else:
                normed = self._input_norm(single_stream)
            attended = self.attention(
                normed,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                is_prefill=is_prefill,
                start_position=start_position,
                chunk_size=chunk_size,
                real_tokens=real_tokens,
                row_mask=row_mask,
            )
            if collector is not None:
                collector.append(attended)
            return attended

        site = _mhc_attention_site(self, streams)
        if site is None:
            return hidden_states + attention_half(hidden_states)
        if num_requests > 1:
            return site.forward(streams, attention_half, num_requests=num_requests)
        return site.forward(streams, attention_half)


# ---------------------------------------------------------------------------
# The DSA (``deepseek_sparse_attention``) half. ``Glm5NextDSALayer`` is the
# decoder layer, ``Glm5NextMLAAttention`` sits at ``self_attn``, and the sparse
# indexer sits under it at ``self_attn.indexer``.
# ---------------------------------------------------------------------------


class Glm5NextDSAIndexerError(ValueError):
    """Raised when the DSA indexer is asked for something it cannot serve.

    Declared beside the class that raises it, the convention every sibling route
    error in this file follows. A distinct type rather than a bare ``ValueError``,
    so a refusal this class owns can be told apart from one torch raised on its
    way through: the indexer's refusals are geometry and materialisation checks on
    operands it did not load.
    """


class Glm5NextDSAIndexer(nn.Module):
    """The DSA sparse indexer at ``self_attn.indexer``.

    Sized from the seven ``index_*`` fields on :class:`Glm5NextTextConfig`. It
    owns the four projections of the indexer chain, its key normalisation and its
    forward. The four projections run on the ``mla_projection`` kernel and their
    weights are prepared once per instance, which is what the reference's own
    indexer does -- it caches its transposed weights-projection half on the
    indexer module rather than transposing per call.
    """

    #: Attribute the prepared weights are cached on. A plain attribute and not a
    #: buffer: a buffer enters ``state_dict()``, which would double every prepared
    #: weight in a saved checkpoint.
    PREPARED_WEIGHTS_ATTR = "_prepared_indexer_weights"

    #: Site name -> the parameter attribute that site's weight arrives on.
    #:
    #: Three of the four follow the ``f"{name}_weight"`` convention and the fourth
    #: does not: ``index_kpool_compress_gate`` is a bare checkpoint tensor with no
    #: ``.weight`` leaf, so a name-plus-suffix rule would look for a parameter that
    #: does not exist. Stating all four here makes the exception visible instead of
    #: hiding it in a fallback.
    #:
    #: One consequence, disclosed: the model's prep walk pre-flights each operand
    #: by building ``name + _WEIGHT_LEAF_SUFFIX`` and continues past a ``None``, so
    #: it covers three of these four sites and silently skips the fourth. It does
    #: not fail. :meth:`prepare_projection_weights` therefore performs that site's
    #: own absent and placeholder checks, so the skip does not become a gap.
    PROJECTION_PARAMETERS: dict[str, str] = {
        "wq_b": "wq_b_weight",
        "wk": "wk_weight",
        "weights_proj": "weights_proj_weight",
        "index_kpool_compress_gate": "index_kpool_compress_gate",
    }

    #: The checkpoint tensors the forward stops reading once
    #: :meth:`prepare_projection_weights` has transposed them: the four
    #: projection weights, which carry no scale grid. The load releases these.
    RELEASED_AFTER_PREP: tuple[str, ...] = tuple(PROJECTION_PARAMETERS.values())

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        # The dials this class computes with, each read from the config rather than
        # transcribed.
        self.hidden_size = int(text_config.hidden_size)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.index_n_heads = int(text_config.index_n_heads)
        self.index_head_dim = int(text_config.index_head_dim)
        self.index_kpool = int(text_config.index_kpool)
        self.index_topk = int(text_config.index_topk)
        # Both compress dials are read, and neither is a switch: the forward refuses
        # any value but True for either (:meth:`require_dials`). They are
        # preconditions, so no branch in this class serves False.
        self.index_kpool_compress = bool(text_config.index_kpool_compress)
        self.index_kpool_always_select_tail = bool(
            text_config.index_kpool_always_select_tail
        )

        # The weight map's seven, in its own order: the four scaled projections,
        # then ``k_norm_bias``, then the two bare compress tensors.
        _declare_parameters(
            self,
            "wq_b_weight",
            "wk_weight",
            "k_norm_weight",
            "weights_proj_weight",
            "k_norm_bias",
            "index_kpool_compress_ape",
            "index_kpool_compress_gate",
        )

    def projection_widths(self) -> tuple[tuple[str, int, int], ...]:
        """The four sites as ``(name, in_features, out_features)``, closed form.

        Every width is computed from a config dial and named where it comes from,
        so a test's expectation is not a transcription of the same literal this
        code used.

        ``index_n_heads * index_head_dim`` and ``hidden_size`` are both 4096 on
        this checkpoint and they are not the same quantity: the first is the
        indexer's total head width, the second the model's residual width. Each is
        written from its own dial below, so the coincidence cannot be mistaken for
        an identity by a later reader or by a checkpoint that separates them.

        ``index_kpool_compress_gate`` is a projection here because that is what it
        is: the reference computes the per-token pool gate as
        ``F.linear(hidden_states, index_kpool_compress_gate)`` with the weight
        shaped ``[index_head_dim, hidden_size]``, so it contracts the hidden width
        exactly as the other three do.
        """
        return (
            ("wq_b", self.q_lora_rank, self.index_n_heads * self.index_head_dim),
            ("wk", self.hidden_size, self.index_head_dim),
            ("weights_proj", self.hidden_size, self.index_n_heads),
            ("index_kpool_compress_gate", self.hidden_size, self.index_head_dim),
        )

    def prepare_projection_weights(self) -> int:
        """Transpose the four indexer weights once. Returns how many.

        ``mla_projection`` needs each weight contraction-major,
        ``[in_features, out_features]``, while a checkpoint stores the torch
        orientation, ``[out_features, in_features]``. A projection weight never
        changes, so it is transposed here once rather than on every call.

        The method is found by duck type: the model's prep walk calls
        ``prepare_projection_weights`` on any module whose type declares it, so
        binding this class under the attention module is all the wiring it needs
        and no load-path code is touched.

        It carries its own absent and placeholder checks, which is not
        duplication: the walk's pre-flight builds each attribute name by appending
        ``_weight``, so it cannot see this class's bare
        ``index_kpool_compress_gate`` leaf at all.

        Each weight is checked against the closed-form widths before it is
        transposed, so a checkpoint whose indexer geometry differs fails here with
        the site named, rather than reaching a kernel that would accept the
        geometry and quietly compute the wrong thing. That check is what lets the
        widths above come from named dials instead of from the weights themselves.
        """
        prepared: dict[str, torch.Tensor] = {}
        for name, idim, odim in self.projection_widths():
            attribute = self.PROJECTION_PARAMETERS[name]
            weight = getattr(self, attribute, None)
            if weight is None:
                raise Glm5NextDSAIndexerError(
                    f"{attribute} is declared but not materialised; load the "
                    f"checkpoint before preparing the indexer's weights"
                )
            if torch.nn.parameter.is_lazy(weight):
                raise Glm5NextDSAIndexerError(
                    f"{attribute} is still a shape-free placeholder, so preparing "
                    f"it would transpose a parameter no checkpoint has filled"
                )
            if tuple(weight.shape) != (odim, idim):
                raise Glm5NextDSAIndexerError(
                    f"{attribute} is {tuple(weight.shape)}; this config's closed "
                    f"form for the {name} site is [out_features, in_features] = "
                    f"{(odim, idim)}"
                )
            # ``.t()`` alone is a view and the kernel loads from memory, so the copy
            # is forced here, once. It is forced on a host copy, because this operand
            # is device-resident by the caller's pre-flight and a Neuron tensor
            # refuses ``.contiguous()`` on a transposed view. The upcast rides along
            # on the host, so no fp32 copy of the weight -- four times the fp8 size
            # -- is ever built on the device.
            prepared[name] = _relaid_out_on_the_host(
                weight, lambda host: host.to(torch.float32).t()
            )
        setattr(self, self.PREPARED_WEIGHTS_ATTR, prepared)
        return len(prepared)

    def _prepared_weight(self, name: str) -> torch.Tensor:
        """One prepared weight, or a refusal naming what was not done.

        Refusing is what makes "never per call" checkable: a fallback that
        transposed on demand would bring back the per-call copy this preparation
        exists to remove, with nothing to report it.
        """
        prepared = getattr(self, self.PREPARED_WEIGHTS_ATTR, None)
        if not prepared:
            raise Glm5NextDSAIndexerError(
                "prepare_projection_weights() has not run; the indexer's "
                "projection weights are transposed once at load time, never per "
                "call"
            )
        return prepared[name]

    #: Epsilon of the indexer's key normalisation. A module constant and not a
    #: config read, because this fork's config carries no field for it: the adapter
    #: keeps only declared dataclass fields, and no indexer dial survives that
    #: filter. The value is the reference's own ``LayerNorm(head_dim, eps=1e-6)``,
    #: which its DeepSeek-V3.2 ancestor states identically. Not ``rms_norm_eps``:
    #: that is this checkpoint's 1e-05 for the RMS norms, a different constant on a
    #: different norm, and reusing it would be a silent substitution.
    KEY_NORM_EPS = 1e-6

    def _key_norm(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """LayerNorm on the indexer key: ``[tokens, index_head_dim]`` in, same out.

        A LayerNorm and not an RMSNorm, and the checkpoint is what says so: the
        weight map binds ``k_norm_bias`` beside ``k_norm_weight``, and an RMS norm
        has no bias to bind. So the mean is subtracted here; treating this as the
        sibling RMS norm would drop that subtraction and compute a different
        function at exactly the right shape.

        The fp32 cast is the reference's, not a local preference: upstream
        normalises in float32 and casts back to the input dtype in one fused step.
        The cast back matters because the value returned here feeds kernels whose
        gates admit bf16 only, and a float32 key would take a torch oracle instead
        of the NKI route while every shape still checked out.
        """
        if x.ndim != 2:
            raise Glm5NextDSAIndexerError(
                f"the indexer key must be [tokens, index_head_dim]; got shape "
                f"{tuple(x.shape)}"
            )
        width = int(x.shape[1])
        for name, operand in (("k_norm_weight", weight), ("k_norm_bias", bias)):
            if operand is None:
                raise Glm5NextDSAIndexerError(
                    f"{name} is declared but not materialised; load the "
                    f"checkpoint before the indexer runs"
                )
            if operand.ndim != 1 or int(operand.shape[0]) != width:
                raise Glm5NextDSAIndexerError(
                    f"{name} must be [{width}] to normalise a key of width "
                    f"{width}; got {tuple(operand.shape)}"
                )
        normed = torch.nn.functional.layer_norm(
            x.float(),
            (width,),
            weight.float(),
            bias.float(),
            self.KEY_NORM_EPS,
        )
        return normed.to(x.dtype)

    def projection_scale(self) -> float:
        """The single constant folded into ``weights`` before the score gemm.

        Upstream folds two factors into one constant -- ``softmax_scale``
        (``head_dim ** -0.5``) and ``n_head ** -0.5`` -- and both come from this
        class's own dials, so nothing is minted here and no caller passes a scale
        in.

        Not the layer's ``softmax_scale``, and the two must not be confused: that
        one is the MLA attention's and reaches ``attend()`` as a caller argument,
        while this one is the indexer's.

        Upstream has a third factor, the fp8 quantisation scale from
        ``fwht128_quant_fp8``. This fork's rotation kernel ``dsa_hadamard128``
        returns a plain tensor and carries no fp8 or ue8m0 scale, so a reference
        for this chain folds two factors and not three.
        """
        return float(self.index_head_dim**-0.5) * float(self.index_n_heads**-0.5)

    def project_stage(
        self, hidden_states: torch.Tensor, q_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The indexer's four projections, its key norm and its query rotation.

        Returns ``(query, key, weights, gate_score)``:

        * ``query`` ``[tokens, index_n_heads, index_head_dim]`` bf16, rotated.
        * ``key`` ``[tokens, index_head_dim]`` bf16, normalised.
        * ``weights`` ``[tokens, index_n_heads]`` float32, scale already folded.
        * ``gate_score`` ``[tokens, index_head_dim]`` bf16.

        ``q_latent`` is the normalised query latent, which upstream's indexer also
        takes as an argument rather than computing. Here it comes from
        ``Glm5NextMLAAttention.project_query_latent``.

        Five kernel calls: four to ``mla_projection``, one per site in
        :meth:`projection_widths`, and one to ``dsa_hadamard128`` for the
        rotation.

        The dtype ladder is forced by the kernels' own gates, not chosen.
        ``mla_projection`` returns float32 by contract. ``dsa_hadamard128`` and
        the pooling kernels take the NKI route on bf16 only, and
        ``dsa_score_gemm`` wants bf16 queries with float32 weights. So the key,
        the query and the gate are cast to bf16 and the weights stay float32 -- a
        float32 key would pass every shape check and silently take a torch oracle
        instead of the kernel.

        RoPE and head zero-padding are both absent because this checkpoint does
        not reach them: upstream skips the split, the rotary and the cat entirely
        when ``rope_dim == 0`` and this checkpoint's ``qk_rope_head_dim`` is 0,
        and it pads heads only when ``n_head < 32`` while this checkpoint has
        exactly 32.
        """
        # Fully-qualified module imports, which is this file's own form for a
        # ``functional/`` kernel and not a package-level one: the DSA package's
        # ``__init__.py`` is empty, so a package-level import of a kernel name would
        # raise ``ImportError`` at the first call.
        from vllm_neuron.functional.attention.mla_projections import mla_projection
        from vllm_neuron.functional.dsa.kpool_hadamard import dsa_hadamard128

        if hidden_states.ndim != 2 or int(hidden_states.shape[1]) != self.hidden_size:
            raise Glm5NextDSAIndexerError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(hidden_states.shape[0])
        if q_latent.ndim != 2 or tuple(q_latent.shape) != (tokens, self.q_lora_rank):
            raise Glm5NextDSAIndexerError(
                f"q_latent must be [tokens, q_lora_rank] = "
                f"{(tokens, self.q_lora_rank)} to pair with these hidden "
                f"states; got {tuple(q_latent.shape)}"
            )

        hidden_f32 = hidden_states.to(torch.float32)

        # 1. The query. Upstream views the projection to ``[-1, n_head, head_dim]``;
        #    the rotation kernel takes a 2-D ``[rows, 128]``, so the view is flat
        #    for the call and restored after it.
        query = mla_projection(q_latent.to(torch.float32), self._prepared_weight("wq_b"))
        query = dsa_hadamard128(
            query.reshape(tokens * self.index_n_heads, self.index_head_dim).to(
                torch.bfloat16
            )
        ).reshape(tokens, self.index_n_heads, self.index_head_dim)

        # 2. The key, then its LayerNorm. The cast to bf16 happens before the norm
        #    so the norm's own cast-back lands on bf16, which is upstream's order:
        #    its ``k`` leaves a bf16 linear and its fused key norm normalises in
        #    float32 and casts back.
        key = self._key_norm(
            mla_projection(hidden_f32, self._prepared_weight("wk")).to(torch.bfloat16),
            self.k_norm_weight,
            self.k_norm_bias,
        )

        # 3. The weights, with the single constant folded in. float32 all the way:
        #    ``dsa_score_gemm`` requires it and applies no scale of its own.
        weights = mla_projection(
            hidden_f32, self._prepared_weight("weights_proj")
        ) * self.projection_scale()

        # 4. The kpool gate: upstream's
        #    ``F.linear(hidden_states, index_kpool_compress_gate)``, which is why
        #    ``projection_widths`` carries this site as a projection.
        gate_score = mla_projection(
            hidden_f32, self._prepared_weight("index_kpool_compress_gate")
        ).to(torch.bfloat16)

        return query, key, weights, gate_score

    def pool_window(
        self,
        key: torch.Tensor,
        gate_score: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool complete blocks of ``index_kpool`` keys. The prefill leg.

        Returns ``(pooled, write_mask)``: ``pooled`` is
        ``[tokens, index_head_dim]`` in ``key``'s dtype, one candidate per
        position, and ``write_mask`` is ``[tokens]`` bool marking which of those
        positions actually completed a pool. The caller writes only the masked
        rows, at ``slot_mapping``'s slots.

        The reference's shape is reproduced rather than improved on: every
        position is treated as a pool-completion candidate and the non-completions
        are masked, because -- in the reference's own words -- compacting the valid
        rows first "costs two device syncs on the eager prefill path and buys
        nothing numerically".

        ``slot_mapping`` is pool-granular, which is the whole reason the mask
        exists: only the last token of each complete pool carries a non-negative
        slot and intra-pool positions carry ``-1``. The second term,
        ``pos >= index_kpool - 1``, drops a pool whose start falls before the
        batch -- leading padding, whose gate and key data the reference calls
        undefined.

        ``dsa_kpool_hadamard`` asserts complete pools, so this method masks every
        non-completion and pools nothing partial. The remainder belongs to the
        decode ring, and :meth:`seed_tail` writes it there on this same leg: no
        caller could do it, because the key and the gate for those positions exist
        only inside this class's forward.

        One kernel call, on ``dsa_kpool_hadamard``. The sliding window is index
        arithmetic and carries no call of its own.
        """
        from vllm_neuron.functional.dsa.kpool_hadamard import dsa_kpool_hadamard

        if key.ndim != 2 or int(key.shape[1]) != self.index_head_dim:
            raise Glm5NextDSAIndexerError(
                f"key must be [tokens, {self.index_head_dim}]; got "
                f"{tuple(key.shape)}"
            )
        if tuple(gate_score.shape) != tuple(key.shape):
            raise Glm5NextDSAIndexerError(
                f"gate_score must match key; got {tuple(gate_score.shape)} "
                f"against {tuple(key.shape)}"
            )
        tokens = int(key.shape[0])
        if slot_mapping.ndim != 1 or int(slot_mapping.shape[0]) != tokens:
            raise Glm5NextDSAIndexerError(
                f"slot_mapping must be [tokens] = {(tokens,)}, one pool-granular "
                f"slot per position; got {tuple(slot_mapping.shape)}"
            )
        pool = self.index_kpool
        if tokens < pool:
            raise Glm5NextDSAIndexerError(
                f"no pool can complete in {tokens} token(s) at a pool size of "
                f"{pool}; the reference returns early here "
                f"(sparse_attn_indexer_kpool.py) and this caller refuses "
                f"instead, so a silent no-op cannot look like a pooled prefill"
            )

        ape = self.index_kpool_compress_ape
        if ape is None:
            raise Glm5NextDSAIndexerError(
                "index_kpool_compress_ape is declared but not materialised; "
                "load the checkpoint before the indexer runs"
            )
        if ape.ndim != 2 or tuple(ape.shape) != (pool, self.index_head_dim):
            raise Glm5NextDSAIndexerError(
                f"index_kpool_compress_ape must be [index_kpool, "
                f"index_head_dim] = {(pool, self.index_head_dim)}; got "
                f"{tuple(ape.shape)}"
            )

        pos = torch.arange(tokens, device=key.device)
        offsets = torch.arange(pool, device=key.device)
        window = (pos - (pool - 1)).clamp_min(0)[:, None] + offsets[None, :]
        write_mask = (slot_mapping >= 0) & (pos >= pool - 1)

        # The ape is float32 for the kernel's inner contract even though the
        # checkpoint leaf is bf16; it is [4, 128], so the cast is free and is
        # done here rather than cached, unlike the projection weights.
        pooled = dsa_kpool_hadamard(
            key[window], gate_score[window], ape.to(torch.float32)
        )
        return pooled, write_mask

    def tail_step(
        self,
        tail: torch.Tensor,
        key: torch.Tensor,
        gate_score: torch.Tensor,
        position: torch.Tensor | int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Advance the decode ring by one token. Returns ``(pooled, new_tail)``.

        ``new_tail`` always has ``tail``'s shape. ``pooled`` is
        ``[1, index_head_dim]`` when this token completed a pool and ``None`` when
        it did not -- but only when the position arrives as a python int. A tensor
        position always answers a row, because whether a pool ended is then a value
        on device.

        The ring is state and this method does not own it: the returned ring goes
        back to the caller, exactly as ``attend()`` takes its latent cache as an
        argument rather than holding one. This method allocates nothing.

        This is the decode counterpart of :meth:`pool_window` rather than a special
        case of it. A prefill sees whole pools and pools them in one call; a decode
        step sees one token and must remember the pool it is part-way through,
        which is what the ring holds -- half 0 keys, half 1 gate scores, upstream's
        own layout.

        ``position`` is the token's absolute position, and it decides which ring row
        is written and whether the pool completes. A tensor is the route this method
        is for: a decode graph is captured once and replayed at every step, so a
        ring row derived from a host int is the row the capture happened at, written
        again at every later position. The int route is kept beside it as the eager
        reference the tensor route is compared against, bit for bit.

        One kernel call either way, on the pair that shares one kernel, and it is
        the only one in this chain that fires on the decode leg and not the prefill
        leg.
        """
        from vllm_neuron.functional.dsa.decode_tail_update import (
            dsa_decode_tail_update,
            dsa_decode_tail_update_at,
        )

        pool = self.index_kpool
        want_tail = (2, pool, self.index_head_dim)
        if tail.ndim != 3 or tuple(tail.shape) != want_tail:
            raise Glm5NextDSAIndexerError(
                f"tail must be [2, index_kpool, index_head_dim] = {want_tail} "
                f"-- half 0 keys, half 1 gate scores; got {tuple(tail.shape)}"
            )
        want_row = (1, self.index_head_dim)
        for name, operand in (("key", key), ("gate_score", gate_score)):
            if operand.ndim != 2 or tuple(operand.shape) != want_row:
                raise Glm5NextDSAIndexerError(
                    f"{name} must be [1, index_head_dim] = {want_row} for one "
                    f"decode token; got {tuple(operand.shape)}"
                )
        # The refusal is the int route's alone, and its place in the order is
        # unchanged. Reading a tensor position to compare it with zero is the host
        # read this method exists to remove, so the tensor route carries no such
        # refusal: the runner computes the position and can refuse before the trace.
        if not torch.is_tensor(position) and int(position) < 0:
            raise Glm5NextDSAIndexerError(
                f"position must be the token's absolute position in the "
                f"request; got {position}"
            )

        ape = self.index_kpool_compress_ape
        if ape is None or tuple(ape.shape) != (pool, self.index_head_dim):
            raise Glm5NextDSAIndexerError(
                f"index_kpool_compress_ape must be materialised and shaped "
                f"{(pool, self.index_head_dim)}; got "
                f"{None if ape is None else tuple(ape.shape)}"
            )

        if torch.is_tensor(position):
            return dsa_decode_tail_update_at(
                tail, key, gate_score, ape.to(torch.float32), position
            )
        return dsa_decode_tail_update(
            tail, key, gate_score, ape.to(torch.float32), int(position)
        )

    def seed_tail(
        self,
        tail: torch.Tensor,
        key: torch.Tensor,
        gate_score: torch.Tensor,
        end_position: torch.Tensor | int,
        start_position: torch.Tensor | int | None = None,
    ) -> torch.Tensor | int:
        """Stash this chunk's remainder in the decode ring. Returns the rows written.

        The count comes back as a python int when ``end_position`` is one and as a
        0-d int32 tensor when it is a tensor, because that count is then a value on
        device and no caller may have it as a number without reading one.

        A prefill pools only complete blocks of ``index_kpool`` keys
        (:meth:`pool_window`); the positions after its last complete pool belong to
        a pool that has not completed yet. The decode leg pools from the ring and
        every real token stashes there, so those positions must be in the ring
        before the first decode step -- otherwise the next completion pools zeros in
        their place and no check notices.

        The ring slot of an absolute position is ``position % index_kpool``. With
        ``end_position`` the sequence length after this chunk, the open pool holds
        positions ``end_position - r`` through ``end_position - 1`` for
        ``r = end_position % index_kpool``, and those are slots ``0`` through
        ``r - 1``. Half 0 is keys and half 1 is gate scores, the halves
        :meth:`tail_step` declares and refuses on.

        A chunked prefill writes only its own share: when the open pool started in
        an earlier chunk, only its last ``min(r, tokens)`` positions are in this
        chunk's ``key``, so only the matching high slots are written and the earlier
        chunk's rows stay. Chunks therefore compose without clobbering each other.

        Nothing is written when the sequence divides evenly, and that is correct
        rather than a shortcut: there is no open pool, the next decode step starts
        at slot 0, and the ring it starts from was emptied for the sequence by the
        runner.

        ``end_position`` decides the slots, so a host int here would compile this
        chunk's remainder length into the graph. The tensor route below writes the
        same slots without reading the value, and the int route stays as the eager
        reference it is compared against.

        The remainder comes from the chunk's real rows. A padded chunk's trailing
        rows carry no token of the sequence, so ``start_position`` says where the
        chunk began: its real length is ``end_position - start_position`` and the
        remainder is read from the last real rows of ``key``. Without it the length
        falls back to the operand's width, which is what an unpadded caller hands
        anyway.
        """
        pool = self.index_kpool
        want_tail = (2, pool, self.index_head_dim)
        if tail.ndim != 3 or tuple(tail.shape) != want_tail:
            raise Glm5NextDSAIndexerError(
                f"prefill_tail must be [2, index_kpool, index_head_dim] = "
                f"{want_tail} -- half 0 keys, half 1 gate scores; got "
                f"{tuple(tail.shape)}"
            )
        if key.ndim != 2 or int(key.shape[1]) != self.index_head_dim:
            raise Glm5NextDSAIndexerError(
                f"key must be [tokens, {self.index_head_dim}]; got "
                f"{tuple(key.shape)}"
            )
        if tuple(gate_score.shape) != tuple(key.shape):
            raise Glm5NextDSAIndexerError(
                f"gate_score must match key; got {tuple(gate_score.shape)} "
                f"against {tuple(key.shape)}"
            )
        tokens = int(key.shape[0])
        # Either position being a tensor takes the tensor route. Both describe the
        # same chunk, so a caller that has one as a value has both, and the int route
        # below may then read neither as a number.
        if torch.is_tensor(end_position) or torch.is_tensor(start_position):
            return self._seed_tail_at(
                tail, key, gate_score, end_position, tokens, start_position
            )
        real = (
            tokens
            if start_position is None
            else int(end_position) - int(start_position)
        )
        if int(end_position) < real:
            raise Glm5NextDSAIndexerError(
                f"end_position is the sequence length AFTER this chunk and "
                f"{int(end_position)} is shorter than the chunk's own {real} "
                f"token(s)"
            )
        rows = int(end_position) % pool
        take = min(rows, real)
        if take <= 0:
            return 0
        # ``take`` and ``rows`` are python ints, so these are trace-time addresses --
        # the same reason ``tail_step``'s slot is a python int and not a tensor. Both
        # sides stay rank 3: slicing the half rather than indexing it keeps the
        # target's rank, so no source can reach it by broadcast, which eager torch
        # allows and the graph compiler refuses. The slice ends at the chunk's real
        # last row, which is its width when nothing was padded.
        tail[0:1, rows - take:rows, :] = (
            key[real - take:real].to(tail.dtype).reshape(1, take, -1)
        )
        tail[1:2, rows - take:rows, :] = (
            gate_score[real - take:real].to(tail.dtype).reshape(1, take, -1)
        )
        return take

    def _seed_tail_at(
        self,
        tail: torch.Tensor,
        key: torch.Tensor,
        gate_score: torch.Tensor,
        end_position: torch.Tensor,
        tokens: int,
        start_position: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """:meth:`seed_tail`'s write with the chunk's end as a tensor. The same slots.

        The same three numbers, none of them read: ``r = end_position %
        index_kpool`` is the open pool's row count, the written slots are
        ``r - take`` through ``r - 1`` for ``take = min(r, real)``, and slot ``s``
        takes the key at row ``real - r + s``. All three are arithmetic on a 0-d
        tensor here, so the write's addresses are values rather than trace-time
        constants.

        ``real`` is the chunk's own length, ``end_position - start_position``, and
        it is the operand's width only when nothing was padded. The remainder
        belongs to the sequence's last real rows; taken from the width, a padded
        chunk would seed the ring from padding rows and the next completion would
        pool them.

        The whole ring is copied to write part of it, because a slice needs its
        bounds as host ints and a boolean index produces a data-dependent shape --
        the graph break in another costume. ``torch.where`` over all
        ``index_kpool`` rows has one shape at every position, and the rows outside
        the window take their own old value, which is what "an earlier chunk's rows
        stay" means as an op. Nothing is written when the sequence divides evenly,
        because the mask is then empty everywhere.

        The source is clamped, not masked, and the clamp is not a correction: rows
        the mask discards still have to name a legal source row, or the gather
        would read out of bounds to produce values nothing uses.
        """
        pool = self.index_kpool
        device = tail.device
        rows = torch.remainder(_int64_scalar(end_position, device), pool)
        slots = torch.arange(pool, device=device)
        real = _int64_scalar(tokens, device)
        if start_position is not None:
            began = _int64_scalar(start_position, device)
            real = _int64_scalar(end_position, device) - began
        write = ((slots >= (rows - real).clamp_min(0)) & (slots < rows))[:, None]
        source = (slots - rows + real).clamp_min(0).minimum(real - 1)
        tail[0].copy_(
            torch.where(write, key.index_select(0, source).to(tail.dtype), tail[0])
        )
        tail[1].copy_(
            torch.where(
                write, gate_score.index_select(0, source).to(tail.dtype), tail[1]
            )
        )
        return write.sum().to(torch.int32)

    def score_pools(
        self,
        query: torch.Tensor,
        candidate_keys: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate pool against every token. ``[tokens, cands]`` fp32.

        ``query`` is ``[tokens, index_n_heads, index_head_dim]`` bf16 and
        ``candidate_keys`` is ``[cands, index_head_dim]`` -- one key per candidate,
        shared across heads, which is this checkpoint's mqa shape.

        The candidate axis is pool-granular rather than token-granular, which is
        why the next stage selects pools and why an expansion is needed afterwards
        to reach tokens.

        ``weights`` arrives already scaled and this method adds nothing: the kernel
        applies no scale of its own by contract, and the constant is
        :meth:`projection_scale`, folded in by :meth:`project_stage`. Folding it
        twice would square it, so the fold has one site.

        One kernel call, on ``dsa_score_gemm``.
        """
        from vllm_neuron.functional.dsa.score_gemm import dsa_score_gemm

        if query.ndim != 3 or tuple(query.shape[1:]) != (
            self.index_n_heads,
            self.index_head_dim,
        ):
            raise Glm5NextDSAIndexerError(
                f"query must be [tokens, index_n_heads, index_head_dim] = "
                f"[tokens, {self.index_n_heads}, {self.index_head_dim}]; got "
                f"{tuple(query.shape)}"
            )
        tokens = int(query.shape[0])
        if candidate_keys.ndim != 2 or int(candidate_keys.shape[1]) != self.index_head_dim:
            raise Glm5NextDSAIndexerError(
                f"candidate_keys must be [cands, {self.index_head_dim}], one "
                f"key per candidate pool; got {tuple(candidate_keys.shape)}"
            )
        if tuple(weights.shape) != (tokens, self.index_n_heads):
            raise Glm5NextDSAIndexerError(
                f"weights must be [tokens, index_n_heads] = "
                f"{(tokens, self.index_n_heads)} to pair with this query; got "
                f"{tuple(weights.shape)}"
            )
        return dsa_score_gemm(query, candidate_keys, weights)

    def select_k(self) -> int:
        """How many pools the selection keeps. Derived in one place.

        ``index_topk`` counts tokens, while the selection on this checkpoint is
        pool-granular, so the token budget divides by the pool width before it
        reaches ``dsa_topk_select``. Neither number is spelled here; both are
        config dials.
        """
        return self.index_topk // self.index_kpool

    def require_dials(self) -> None:
        """Refuse the two compress dials this class does not serve.

        A refusal rather than a branch. The ``dsa_index_expand`` kernel is
        upstream's fused expand-plus-tail: it appends the tail itself and derives
        ``tail_start`` internally, so there is no force-include step for this class
        to author, and a ``False`` would need a different expansion no kernel here
        provides.

        What the tail actually is. The appended entities are raw token indices and
        not a pool id: the kernel adds ``pool_size - 1`` columns holding
        ``tail_start + t`` with ``tail_start = seq_len // pool_size * pool_size``,
        one column per possible position in the incomplete final pool. Every other
        column of the row carries pool ids expanded to tokens; these carry
        individual tokens directly. The append is empty whenever
        ``seq_len % pool_size == 0``, because ``tail_count`` is then zero and every
        tail column masks to ``-1``. So the dial selects an expansion *shape* that
        reserves those columns, not a guarantee that they are populated -- and at a
        pool-aligned length no tail token is contributed at all.

        Upstream refuses the same two values, so this is a transcription and not a
        local policy. It tests ``is not True`` rather than falsiness; both dials are
        stored through ``bool()`` at construction, so the two tests cannot diverge
        here.
        """
        if not self.index_kpool_compress:
            raise Glm5NextDSAIndexerError(
                "index_kpool_compress must be True; this indexer serves only "
                "the compressed, pool-granular path, because the "
                "dsa_index_expand seam expands pool ids and no seam "
                "selects token-granular candidates. Upstream refuses the same "
                "value (transformers_utils/configs/glm5_next.py)"
            )
        if not self.index_kpool_always_select_tail:
            raise Glm5NextDSAIndexerError(
                "index_kpool_always_select_tail must be True; the "
                "dsa_index_expand seam appends the tail itself -- pool_size - 1 "
                "columns carrying the raw token indices of the incomplete final "
                "pool, empty when seq_len % pool_size == 0 -- and derives "
                "tail_start internally, so False would need an expansion no "
                "seam provides. Upstream refuses the same value "
                "(transformers_utils/configs/glm5_next.py)"
            )

    def select_pools(self, scores: torch.Tensor) -> torch.Tensor:
        """Keep the ``select_k`` highest-scoring pools. ``[rows, select_k]`` int32.

        The one cast between two kernels that disagree: ``dsa_topk_select`` returns
        int64 indices to match ``torch.topk``, while ``dsa_index_expand`` admits
        int32 only, because int64 indices would double the SBUF traffic for a range
        no sequence length reaches.

        Nothing in production calls this method any more: both entry points route
        the selecting regime through :meth:`select_bounded_pools`, which calls the
        selector itself, because the sentinel needs the ``values`` this one
        discards. So the same one-line cast appears in both places.

        This method does not re-check ``dsa_topk_select``'s strict
        ``0 < k < width`` bound. At ``k == width`` the kernel's gate returns false
        rather than raising, which takes the torch route silently, so both entry
        points route that regime away before a score exists:
        ``_require_serviceable`` reports ``selects=False`` and they return the
        causal bypass. Checking the bound in two places would let the two disagree.

        One kernel call, on ``dsa_topk_select``.
        """
        from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

        if scores.ndim != 2:
            raise Glm5NextDSAIndexerError(
                f"scores must be [rows, cands] from score_pools; got "
                f"{tuple(scores.shape)}"
            )
        _values, indices = dsa_topk_select(scores, self.select_k())
        return indices.to(torch.int32)

    def select_bounded_pools(
        self, scores: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Bound each row to its own position, select, sentinelise.

        The one site for the causal bound, and both entry points route through it
        for the reason :meth:`_require_serviceable` gives for itself: ``forward``
        and ``forward_ragged`` must agree about what a row may see, and two copies
        of that rule are two things that can disagree.

        ``causal_len`` is ``seq_lens``, the same column :meth:`expand_indices`
        already receives. One producer, two consumers, nothing minted, which is
        what makes the bound and the expansion incapable of disagreeing about a
        row's length.

        The sentinel is fed the selector's own ``values`` rather than a score
        gathered back at the returned index, because the two are the same number
        only where the selector leaves its input alone -- and it often does not.
        ``dsa_topk_select`` wraps the vendored ``rotational_topk``, whose core
        strikes each value it takes by writing ``-inf`` over it in the input buffer
        whenever the fold's ``k`` is a whole number of stages; only a last fold with
        ``k % 8 != 0`` takes the ``max8`` plus ``nc_find_index8`` path that never
        writes. A later pass's ``max8`` then matches one of those struck positions,
        so the index returned beside a struck value can be a column that was
        originally finite and legal -- gathering from the unstruck scores at that
        index would read the original finite score, write no ``-1``, and let the row
        carry a legal pool id twice, which ``dsa_index_expand`` would then expand
        twice. Production ``select_k`` is ``index_topk // index_kpool`` (512), above
        the factory's small-``k`` threshold, so production always runs the striking
        branch.

        The value alone is still not enough. The same selector pads its own input
        when the fold is uneven, with a finite ``-9948.0`` at column positions that
        keep counting past the real width, and a finite pad outranks a bounded
        column -- so on exactly the rows this bound acts on the pads would win slots
        and come back with an out-of-range pool id and an ordinary-looking value. It
        also moves selected values across partitions with a 0/1 permutation matmul,
        where ``0 * -inf`` is NaN, so an ``-inf`` marker need not survive the trip
        at all. The bound therefore writes a finite ``BOUND_FILL``, and the marker
        takes ``width`` and fires on the fill or on an index at or past ``width``.
        ``width`` is read off the bounded tensor rather than from a config field, so
        it cannot disagree with what the selector was actually given.
        """
        from vllm_neuron.functional.dsa.causal_bound import (
            dsa_causal_bound,
            dsa_causal_sentinel,
        )
        from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

        bounded = dsa_causal_bound(
            scores, seq_lens.to(torch.int32).reshape(-1, 1), self.index_kpool
        )
        values, indices = dsa_topk_select(bounded, self.select_k())
        pool_ids = indices.to(torch.int32)
        sentinelised = dsa_causal_sentinel(values, pool_ids, int(bounded.shape[1]))
        return self._canonical_sentinel_order(sentinelised)

    @staticmethod
    def _canonical_sentinel_order(pool_ids: torch.Tensor) -> torch.Tensor:
        """Sentinels to the trailing columns; real ids keep their relative order.

        Without this, the composition in :meth:`select_bounded_pools` is not a
        function of its inputs alone. ``dsa_topk_select`` promises its results
        highest first and pins nothing about the order among equal values, and the
        causal bound manufactures ties on purpose: every column a row may not see
        becomes the same ``BOUND_FILL``. Which masked column the selector returns,
        and in which position, is therefore unspecified -- and the choice varies
        with the batch rather than with the row, because the kernel's gate consults
        ``n_rows`` and ``width``, both properties of the pack. So one and the same
        query row can take the NKI route packed and the torch route alone, and the
        two routes need not break a tie the same way.

        The values were never wrong, only their places. Every masked column the
        selector picks is turned into ``-1`` by :func:`dsa_causal_sentinel`, so the
        set a row reports is already deterministic, and the real ids are pinned
        because their scores are distinct. What leaked was the position of the
        sentinels among the ``select_k`` columns, and the expansion is positional,
        so a row whose ``-1`` moved re-lays its whole span.

        Keying the sentinel on the selector's own returned value fixes the set on
        every branch but not the places: "sentinels trail" would then rest on the
        selector returning its values in descending order, and its contract says
        that about one pass and nothing about the order across the striking passes
        or across the rotational stages that feed them. So the permutation stays and
        the property holds by construction rather than by an unstated promise. It is
        a no-op whenever the values already arrive descending.

        After this method the output is a function of
        ``(scores, seq_lens, pool_size, select_k)`` and of nothing about the batch:
        real ids first in the selector's own descending order, sentinels after them.

        This is not the "no compaction" clause being broken. That clause belongs to
        :meth:`expand_indices` and governs the expanded tensor handed to
        ``attend()``. This method orders pool ids before the expansion runs and
        neither adds nor removes one: it is a permutation within each row, and every
        row's multiset is unchanged.

        Selecting on the unbounded scores would remove the ties at the source and
        would also be deterministic, and it is refused, because it reverses the
        declared order -- the bound runs between ``dsa_score_gemm`` and
        ``dsa_topk_select``, and the sentinel runs on the selector's output. A row
        would then spend its ``k`` slots on pools it may not see and blank them
        afterwards, instead of choosing among the pools it may see.

        The ordering runs on chip, in
        :func:`vllm_neuron.functional.dsa.sentinel_order.dsa_sentinel_order`,
        because a stable partition is countable rather than comparable: a real id's
        destination is the number of reals before it, a sentinel's is the real total
        plus the number of sentinels before it. The torch spelling of that -- two
        prefix sums through the fork's own ``cumsum``, a ``where`` and an
        out-of-place ``scatter`` -- was correct, but it handed the compiler an hlo
        scatter whose source layout the compiler chose for itself, moving the
        128-wide axis last at the cost of one DMA transpose per DSA layer on the
        prefill path. The kernel does the same two prefix sums as scans, uses the
        destinations as a search key, ``nc_find_index8`` for the inverse permutation
        and ``nc_n_gather`` for the ids, keeps rows on partitions as stored, and
        leaves no scatter to lay out. The torch spelling survives as that module's
        oracle and serves any call the gate refuses. The result is exact for integer
        ids: every count is below 2**24, a bound ``can_run_dsa_sentinel_order``
        enforces by admitting at most ``SEARCH_MAX_FREE`` columns -- the widest row
        whose tiles fit one SBUF partition -- and sending a wider row to the oracle.

        No sort, because the target has none: an argsort spelled this ordering until
        the compiler refused it by name -- ``Operation sort is not supported on
        trn2`` -- in both the prefill and the decode graph.
        """
        if pool_ids.ndim != 2:
            raise Glm5NextDSAIndexerError(
                f"pool_ids must be [rows, select_k] from the sentinel; got "
                f"{tuple(pool_ids.shape)}"
            )
        from vllm_neuron.functional.dsa.sentinel_order import dsa_sentinel_order

        return dsa_sentinel_order(pool_ids)

    def expand_indices(
        self, pool_ids: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Selected pools expanded to token indices, tail appended. One kernel call.

        Returns ``[rows, select_k * index_kpool + index_kpool - 1]`` int32 token
        indices in upstream's column order, where ``-1`` means "this column selects
        no token" and is a value rather than an out-of-bounds index.

        The ``-1`` leaves this class untouched: the mask goes inside
        ``mla_sparse_attention``, which masks ``-1`` and narrows its gate to
        ``lo < -1``. So this class applies no filler, no compaction, no clamp and no
        mask, and hands the sentinel-bearing tensor to ``attend()`` unchanged.

        That is needed because the fork's ``mla_sparse_attention`` refuses any
        negative index by name, while upstream bounds its kernel with a separate
        per-row valid count instead -- a parameter the fork's kernel does not have.
        The sentinel is upstream's design and not an accident: it initialises its
        whole index buffer to ``-1`` and writes the ``-1``-bearing expansion
        straight into it.

        No filler would have been free either. Softmax normalises over the columns
        it is given, so any in-range filler duplicates a real cache row and takes
        real probability mass. Upstream can point its empty rows at slot 0 only
        because it also clamps their length to 1 and zeroes the output afterwards.
        """
        from vllm_neuron.functional.dsa.index_expand import dsa_index_expand

        if pool_ids.ndim != 2:
            raise Glm5NextDSAIndexerError(
                f"pool_ids must be [rows, select_k] from select_pools; got "
                f"{tuple(pool_ids.shape)}"
            )
        if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != int(pool_ids.shape[0]):
            raise Glm5NextDSAIndexerError(
                f"seq_lens must be [rows] = {(int(pool_ids.shape[0]),)}, one "
                f"length per selected row; got {tuple(seq_lens.shape)}"
            )
        return dsa_index_expand(pool_ids, seq_lens, self.index_kpool)

    def _require_serviceable(
        self, max_seq_len: int, page_size: int, pool_cache: torch.Tensor
    ) -> tuple[int, int, bool]:
        """Read the regime. Returns ``(candidates, trash_row, selects)``.

        One implementation for both entry points: :meth:`forward` and
        :meth:`forward_ragged` must agree about what is serviceable, and two copies
        of a refusal are two things that can drift apart. ``selects`` is the one
        decision, and the two entry points each act on it.

        ``selects`` is ``candidates > select_k()``, strictly, because
        ``dsa_topk_select`` needs ``0 < k < width`` strictly: at ``k == width`` its
        gate returns false rather than raising, which takes the torch route
        silently, unlike the loud raise at ``k > width``. ``width`` is the widest
        row's complete-pool count, because that row sizes the shared candidate axis.

        Below the strict bound there is nothing to select from, and upstream does
        not clamp ``k`` there -- it bypasses selection entirely and attends
        causally. So the regime is served rather than refused, and this method
        reports which of the two answers the caller owes.

        The bypass cannot live in this method, which is the whole shape of it. Both
        entry points call this before they write the pooled-key store and advance
        the decode ring, so returning an answer here would skip the write and
        corrupt every later step. The decision is here, once, and the two dispatches
        are at the two entry points, each after its own write stage.
        """
        pool = self.index_kpool
        if int(max_seq_len) <= 0:
            raise Glm5NextDSAIndexerError(
                f"max_seq_len must be the batch's longest sequence as a python "
                f"int; got {max_seq_len!r}"
            )
        if int(page_size) <= 0:
            raise Glm5NextDSAIndexerError(
                f"page_size must be positive; got {page_size!r}"
            )
        if pool_cache.ndim != 2 or int(pool_cache.shape[1]) != self.index_head_dim:
            raise Glm5NextDSAIndexerError(
                f"pool_cache must be [slots, index_head_dim] with "
                f"index_head_dim={self.index_head_dim}; got "
                f"{tuple(pool_cache.shape)}"
            )

        candidates = int(max_seq_len) // pool
        selects = candidates > self.select_k()
        trash = int(pool_cache.shape[0]) - 1
        if candidates > trash:
            raise Glm5NextDSAIndexerError(
                f"pool_cache has {int(pool_cache.shape[0])} row(s), which "
                f"leaves no trash row above the {candidates} addressable "
                f"candidate pool(s); allocate at least {candidates + 1}"
            )
        return candidates, trash, selects

    def bypass_width(self) -> int:
        """The width the short-sequence bypass emits: selection's own width, derived.

        The bypass and the selecting path must hand the sparse attention kernel the
        same shape, or the emitted width would vary by regime and every consumer
        would have to branch. ``dsa_index_expand`` allocates
        ``index_expand_width(n_groups, pool_size)``, so that is read from the
        kernel's own helper here rather than recomputed.
        """
        from vllm_neuron.functional.dsa.index_expand import index_expand_width

        return int(index_expand_width(self.select_k(), self.index_kpool))

    def _bypass_indices(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """The short regime's answer: each row's plain causal prefix, at :meth:`bypass_width`.

        One kernel call, from both entry points, so the two cannot answer the short
        regime differently. Row ``i`` holds ``0 .. seq_lens[i] - 1`` and then
        ``-1``, which is what "attend causally, select nothing" means as an index
        tensor; the ``-1`` is the sentinel the sparse kernel masks.

        Every causal column fits, with no clamp, and that is arithmetic rather than
        luck: the bypass holds only while ``max_seq_len // pool <= select_k``, so
        the largest position is ``select_k * pool + pool - 2``, one below the raw
        expansion width and therefore below the rounded-up emitted width.
        """
        from vllm_neuron.functional.dsa.causal_fill import dsa_causal_fill

        return dsa_causal_fill(seq_lens.to(torch.int32) - 1, self.bypass_width())

    def _gather_candidates(
        self, pool_cache: torch.Tensor, candidates: int, page_size: int
    ) -> torch.Tensor:
        """The candidate pooled keys. One call on ``dsa_paged_gather``.

        The candidate axis is every complete pool of the widest row, shared across
        the batch, which is upstream's own shape: a ``[tokens, max_seq_len_pooled]``
        logits grid rather than a per-row candidate set. Per-row differences enter
        later, at the expansion, where ``dsa_index_expand`` derives each row's tail
        from that row's own ``seq_len``.

        The page table is arithmetic rather than a lookup, and only because of the
        serving constraint: at one sequence per call, pool ``j`` lives at a fixed
        page and slot, which is the same contract ``attend()`` states. A batched
        page table would be a real lookup.
        """
        from vllm_neuron.functional.dsa.paged_gather import dsa_paged_gather

        flat = torch.arange(candidates, device=pool_cache.device, dtype=torch.int32)
        return dsa_paged_gather(
            pool_cache,
            torch.div(flat, int(page_size), rounding_mode="floor"),
            torch.remainder(flat, int(page_size)),
            int(page_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        pool_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        max_seq_len: int,
        page_size: int,
        slot_mapping: torch.Tensor | None = None,
        tail: torch.Tensor | None = None,
        position: torch.Tensor | int | None = None,
        prefill_tail: torch.Tensor | None = None,
        prefill_end_position: torch.Tensor | int | None = None,
        prefill_start_position: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """The whole indexer chain. Returns ``topk_indices`` and nothing else.

        Indices alone is deliberate: the ``-1`` mask stays inside the sparse
        kernel, so no per-row valid length leaves this class -- returning one would
        give a caller a second, contradictory way to bound the same kernel.
        :meth:`expand_indices` states why.

        Args:
            hidden_states: ``[tokens, hidden_size]``.
            q_latent: ``[tokens, q_lora_rank]`` fp32, from the sibling attention's
                ``project_query_latent``.
            pool_cache: ``[slots, index_head_dim]`` bf16, the pooled-key store,
                written in place for this call's completed pools and then read back
                as the candidate set. Its last row is reserved as write trash.
            seq_lens: ``[rows]`` int32, upstream's shape, consumed on device by the
                expansion kernel.
            max_seq_len: the batch's longest sequence, as a python int.
            page_size: rows per page in ``pool_cache``.
            slot_mapping: ``[tokens]`` int32, pool-granular. Prefill only.
            tail: ``[2, index_kpool, index_head_dim]`` bf16 ring, written in place.
                Passing it selects the decode leg.
            position: the decode token's absolute position, as a tensor on the
                traced path. A python int is admitted and takes the eager route; see
                :meth:`tail_step` for which is which and why both exist.
            prefill_tail: the same ring, on the prefill leg, written in place for
                this chunk's remainder by :meth:`seed_tail`. It does not select a
                leg -- ``tail`` alone does that -- and passing it on a decode step
                refuses.
            prefill_end_position: the sequence length after this prefill chunk,
                required with ``prefill_tail``, as a tensor or a python int on the
                same terms as ``position``. It is not ``max_seq_len``: that one is
                the batch's longest sequence, equal to this sequence's end only
                while the batch is one request.
            prefill_start_position: where this chunk began, so the remainder is read
                from the chunk's real rows and not from a padded operand's trailing
                ones. Optional: without it the chunk's length is taken to be the
                operand's width, which is what an unpadded caller hands.

        ``max_seq_len`` is a python int rather than read off ``seq_lens`` because
        ``int(seq_lens.max())`` is a host read of tensor data inside a region the
        runner compiles with ``fullgraph=True``, which is a graph break. The
        candidate count also sizes an ``arange``, so it must be a trace-time
        constant regardless. The cost is that consistency with ``seq_lens`` is
        declared by the caller rather than measured here, and the tests assert it
        host-side, outside the traced region.

        The ring and the pool store are written in place, following this file's own
        contract for carried state, and that is what lets this method return indices
        alone. ``dsa_decode_tail_update`` is functional by design, so its new ring
        is copied into the caller's buffer here, at one site. The prefill leg writes
        the same ring at one site too, :meth:`seed_tail`, for its remainder alone.

        The pool write uses a trash row rather than a boolean mask, because indexing
        with a bool mask produces a data-dependent shape -- the same graph break in
        another costume. So every position writes, and the positions that completed
        no pool are redirected to ``pool_cache``'s last row. Duplicate trash
        destinations resolve in unspecified order, which is harmless: the row is
        never addressed as a candidate. Both legs use it, because the decode leg's
        completion is a value once the position is a tensor, so it steers the write
        instead of deciding whether one happens.

        What is torch here is orchestration, and it is named so a reader can check
        it: shape validation, index arithmetic for the gather and for the two write
        addresses, one ``index_copy_`` per leg, one ring copy on the decode leg, and
        one ring copy on the prefill leg -- masked over the whole ring when the
        chunk's end arrives as a tensor. No torch path computes an indexer value.
        """
        from vllm_neuron.functional.dsa.decode_tail_update import decode_pool_address

        self.require_dials()

        pool = self.index_kpool
        candidates, trash, selects = self._require_serviceable(
            int(max_seq_len), int(page_size), pool_cache
        )
        is_decode = tail is not None
        if is_decode and position is None:
            raise Glm5NextDSAIndexerError(
                "the decode leg needs both tail and position; got a tail with "
                "no position"
            )
        if not is_decode and slot_mapping is None:
            raise Glm5NextDSAIndexerError(
                "the prefill leg needs slot_mapping, the pool-granular slot per "
                "position; pass tail and position instead for a decode step"
            )
        if is_decode and prefill_tail is not None:
            raise Glm5NextDSAIndexerError(
                "prefill_tail is the PREFILL leg's ring and this is a decode "
                "step, whose ring is `tail`; one call advances the ring or seeds "
                "it, never both"
            )
        if prefill_tail is not None and prefill_end_position is None:
            raise Glm5NextDSAIndexerError(
                "seeding the ring needs the sequence length after this chunk; got "
                "a prefill_tail with no prefill_end_position"
            )
        query, key, weights, gate_score = self.project_stage(hidden_states, q_latent)

        if is_decode:
            pooled, new_tail = self.tail_step(tail, key, gate_score, position)
            # The kernel does not mutate its argument, so the ring is threaded
            # back into the caller's buffer here.
            tail.copy_(new_tail)
            if torch.is_tensor(position):
                # The pool write, addressed on device. Whether this token ended a
                # pool is a value now, so it cannot choose whether a write happens
                # -- it chooses where the write lands, and a step that ended no pool
                # lands on the trash row. That is the prefill leg's own pattern two
                # branches below, for the same reason: a write the graph performs at
                # every position beats a write whose existence was decided when the
                # graph was captured.
                pool_index, completes = decode_pool_address(
                    position, pool, pool_cache.device
                )
                # ``new_full`` and not ``torch.tensor``: a tensor built from python data
                # stays real while the rest of the trace is fake, and graph extraction
                # refuses that mix. Both pool writes below take the same form.
                destination = torch.where(
                    completes,
                    pool_index,
                    pool_cache.new_full((), trash, dtype=torch.int64),
                ).reshape(1)
                pool_cache.index_copy_(0, destination, pooled.to(pool_cache.dtype))
            elif pooled is not None:
                # `pooled is not None` is decided from `position`, a python int,
                # so this branch is a trace-time choice and not a data read.
                pool_slot = torch.full(
                    (1,), position // pool, dtype=torch.int64,
                    device=pool_cache.device,
                )
                pool_cache.index_copy_(0, pool_slot, pooled.to(pool_cache.dtype))
        else:
            pooled, write_mask = self.pool_window(key, gate_score, slot_mapping)
            destination = torch.where(
                write_mask,
                slot_mapping.to(torch.int64),
                pool_cache.new_full((), trash, dtype=torch.int64),
            )
            pool_cache.index_copy_(0, destination, pooled.to(pool_cache.dtype))
            if prefill_tail is not None:
                # The remainder is this leg's to persist: the pooled store took the
                # complete pools above, and the open pool's rows exist only here.
                self.seed_tail(
                    prefill_tail,
                    key,
                    gate_score,
                    prefill_end_position,
                    prefill_start_position,
                )

        if not selects:
            # The write stage above has already landed -- the pool row is stored
            # and, on the decode leg, the ring has advanced -- so returning here
            # loses nothing. Below the strict bound there is nothing to select from.
            return self._bypass_indices(seq_lens)

        candidate_keys = self._gather_candidates(pool_cache, candidates, int(page_size))
        scores = self.score_pools(query, candidate_keys, weights)
        # The causal bound and the sentinel, at one site.
        pool_ids = self.select_bounded_pools(scores, seq_lens)
        return self.expand_indices(pool_ids, seq_lens)

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        pool_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        lengths: Sequence[int],
        *,
        max_seq_len: int,
        page_size: int,
    ) -> torch.Tensor:
        """The non-uniform decode arm. Returns dense ``[tokens, width]`` int32.

        The ragged-pack path is unreachable at batch one, because upstream guards
        its whole pack region with ``if decode_metadata.requires_padding:`` and a
        batch of one is uniform by construction. This arm is called on the indexer
        directly, so no layer and no ``attend()`` is involved.

        Only the query is packed, and only because ``dsa_score_gemm`` needs its rows
        dense and aligned with the gate. A weights pack is impossible rather than
        merely unnecessary: the gate weights are fp32 by ``dsa_score_gemm``'s
        contract and ``dsa_ragged_pack`` admits bfloat16 alone, so a weights pack
        would take the torch route. The weights are moved by ``index_select``
        instead -- torch glue on a narrow fp32 gate, which preserves every bit of
        the fold that a bf16 round trip would lose.

        Nothing is unpacked. The arm's result is int32 by ``dsa_index_expand``'s
        contract, the pack module admits bf16 only, and the consumer wants the dense
        form anyway: ``mla_sparse_attention`` gates ``topk_indices.ndim != 2`` and
        raises by name, so a padded rank-3 tensor would be refused. There is no
        dense-to-padded step to author.

        Mind the names, because upstream's are inverted. Upstream's
        ``pack_seq_triton`` is dense-to-padded and ``unpack_seq_triton`` is
        padded-to-dense; the fork's ``dsa_ragged_pack`` is padded-to-dense and
        ``dsa_ragged_unpack`` is dense-to-padded. This method is written to the
        fork's contract, so reading either name by upstream habit inverts the arm
        silently.

        Args:
            hidden_states: ``[batch, max_len, hidden_size]`` -- the padded batch.
            q_latent: ``[batch, max_len, q_lora_rank]`` fp32, padded to match.
            pool_cache: ``[slots, index_head_dim]`` bf16, read only here.
            seq_lens: ``[tokens]`` int32, one per packed row.
            lengths: valid rows per request, as python ints. That is
                ``dsa_ragged_pack``'s own contract: the packed length is derived
                from them, and deriving it from tensor data would be a host read
                inside a ``fullgraph=True`` region.
            max_seq_len: the batch's longest sequence, a python int.
            page_size: rows per page in ``pool_cache``.

        This arm writes no cache and advances no ring, declared rather than
        omitted: its job is the selection chain on a non-uniform batch, the
        pooled-key store arrives already populated, and the ring is the uniform
        case's business.

        One cost, disclosed: :meth:`project_stage` computes the key and the gate
        score that this arm never uses, because it is one unit whose four
        ``mla_projection`` calls per indexer call are the same on both entry points.
        Two unused projections is the price of that staying stable, and it is stated
        rather than hidden.
        """
        from vllm_neuron.functional.dsa.ragged_pack import dsa_ragged_pack

        self.require_dials()
        candidates, _trash, selects = self._require_serviceable(
            int(max_seq_len), int(page_size), pool_cache
        )

        if hidden_states.ndim != 3 or q_latent.ndim != 3:
            raise Glm5NextDSAIndexerError(
                f"the ragged arm takes a PADDED batch: hidden_states "
                f"[batch, max_len, hidden_size] and q_latent "
                f"[batch, max_len, q_lora_rank]; got "
                f"{tuple(hidden_states.shape)} and {tuple(q_latent.shape)}"
            )
        batch, max_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if tuple(q_latent.shape[:2]) != (batch, max_len):
            raise Glm5NextDSAIndexerError(
                f"q_latent's leading dims must match hidden_states' "
                f"{(batch, max_len)}; got {tuple(q_latent.shape[:2])}"
            )
        checked = [int(n) for n in lengths]
        if len(checked) != batch:
            raise Glm5NextDSAIndexerError(
                f"lengths must carry one valid-row count per request; got "
                f"{len(checked)} for a batch of {batch}"
            )
        if any(n < 0 or n > max_len for n in checked):
            raise Glm5NextDSAIndexerError(
                f"every length must lie in [0, max_len={max_len}]; got {checked}"
            )
        tokens = sum(checked)
        if tokens <= 0:
            raise Glm5NextDSAIndexerError(
                "every request is empty; there is nothing to index"
            )
        if len(set(checked)) == 1:
            raise Glm5NextDSAIndexerError(
                f"this arm exists for a NON-UNIFORM batch and every request here "
                f"has {checked[0]} row(s). A uniform batch needs no pack at all "
                f"-- upstream guards its pack region with "
                f"requires_padding (sparse_attn_indexer_kpool.py) -- so "
                f"serving it here would move the pack counter on a case that "
                f"does not need one. Use forward() instead"
            )
        if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != tokens:
            raise Glm5NextDSAIndexerError(
                f"seq_lens must be [tokens] = {(tokens,)}, one per PACKED row, "
                f"which is what the lengths sum to; got {tuple(seq_lens.shape)}"
            )

        # The projections are row-wise, so they commute exactly with row selection:
        # projecting the padded grid and then packing gives bit-for-bit what packing
        # and then projecting would. Padding rows compute values that the pack drops
        # and nothing downstream ever sees. Flattening is a reshape, not a move.
        heads, head_dim = self.index_n_heads, self.index_head_dim
        query, _key, weights, _gate_score = self.project_stage(
            hidden_states.reshape(batch * max_len, -1),
            q_latent.reshape(batch * max_len, -1),
        )

        # The one pack. The query carries ``heads * head_dim`` per row, and the
        # kernel takes a 3-D ``[batch, max_len, width]``, so the head axis folds into
        # the width for the move and unfolds after -- upstream reshapes around its own
        # pack for the same reason.
        packed_query = dsa_ragged_pack(
            query.reshape(batch, max_len, heads * head_dim), checked
        ).reshape(tokens, heads, head_dim)

        # The fp32 gate moves by ``index_select`` rather than through the pack
        # kernel, which admits bf16 alone. The row index is built from the lengths,
        # which are python ints, so its shape is a trace-time constant and no tensor
        # data is read on the host. It comes from ``arange`` and a slice rather than
        # from a python list, because a tensor built from python data stays real while
        # the rest of the trace is fake, and graph extraction refuses that mix.
        grid = torch.arange(max_len, dtype=torch.int64, device=weights.device)
        keep = torch.cat([grid[:n] + b * max_len for b, n in enumerate(checked)])
        packed_weights = weights.index_select(0, keep)

        if not selects:
            # The bypass is placed after the pack rather than before it,
            # deliberately. This arm has no write stage to protect, and the pack is
            # this arm's whole reason for existing, so bypassing before it would zero
            # that reading on the short regime. The packed query it computes is then
            # unused, which is the same disclosed cost as the two unused projections
            # above.
            return self._bypass_indices(seq_lens)

        candidate_keys = self._gather_candidates(pool_cache, candidates, int(page_size))
        scores = self.score_pools(packed_query, candidate_keys, packed_weights)
        # The causal bound and the sentinel, at one site.
        pool_ids = self.select_bounded_pools(scores, seq_lens)
        return self.expand_indices(pool_ids, seq_lens)


class Glm5NextMLADecodeError(ValueError):
    """Raised when the MLA decode path is asked for something it cannot serve.

    Declared beside the class that raises it, the convention every sibling route
    error in this file follows. A distinct type rather than a bare ``ValueError``,
    so the ``B == 1`` serving constraint is distinguishable from a shape typo: a
    test that caught ``ValueError`` would pass on either, and the constraint would
    be asserted without being measured. The kernels this path calls raise their own
    named errors for the same reason.
    """


class Glm5NextMLAAttention(nn.Module):
    """Multi-head latent attention at ``self_attn``, NoPE on this checkpoint.

    Parameter names are the weight map's. No ``*_rope_*`` projection exists:
    ``mla_use_nope`` with ``qk_rope_head_dim == 0`` means there is no rotary head
    slice at all.

    MLA caches one compressed latent vector per token per layer, not one entry per
    attention head -- the map's own projection name says so:
    ``kv_a_proj_with_mqa``, multi-query, a single KV head. So ``num_kv_heads`` is 1
    and is not tensor-parallel sharded, since a single latent is replicated across
    ranks rather than split. The width is ``kv_lora_rank + qk_rope_head_dim``,
    which is 512 on this checkpoint.
    """

    #: The weight map's module path for this family.
    CACHE_NAME_SUFFIX = "self_attn"

    #: MLA compresses KV to one latent per token; the latent is replicated
    #: across tensor-parallel ranks, so this is 1 at every world size.
    NUM_LATENT_KV_HEADS = 1

    #: One latent vector per token, and no value half to cache. The runner reads
    #: this to size the page for one buffer instead of a key/value pair.
    LATENT_KV_CACHE = True

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_attention_heads = int(text_config.num_attention_heads)
        self.kv_lora_rank = int(text_config.kv_lora_rank)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.qk_nope_head_dim = int(text_config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(text_config.qk_rope_head_dim)
        self.v_head_dim = int(text_config.v_head_dim)
        self.mla_use_nope = bool(text_config.mla_use_nope)
        # The hidden size is the input width of two of the five projection sites
        # and the output width of a third; the epsilon is the checkpoint's, read
        # through the config rather than defaulted locally.
        self.hidden_size = int(text_config.hidden_size)
        self.rms_norm_eps = float(text_config.rms_norm_eps)

        self.num_kv_heads_per_rank = self.NUM_LATENT_KV_HEADS
        self.head_size = _resolve_mla_head_size(text_config)
        self.cache_dtype = _resolve_model_dtype(text_config)
        self.cache_chunk_size = None

        _declare_parameters(
            self,
            "q_a_proj_weight",
            "q_b_proj_weight",
            "kv_a_proj_with_mqa_weight",
            "kv_b_proj_weight",
            "o_proj_weight",
            "q_a_layernorm_weight",
            "kv_a_layernorm_weight",
        )
        # The four blockwise-FP8 scale parameters, in their own call so the seven
        # names above stay as they are.
        #
        # Derived from the weight map's own list and never a second literal: the
        # leaves come from ``DSA_SCALED_PROJECTIONS`` and the suffix from
        # ``FP8_SCALE_SUFFIX``. That is what keeps the declared-name set and the
        # weight map's parameter set from drifting apart.
        #
        # ``kv_b_proj`` is absent because it is absent from that list: this
        # checkpoint keeps it in bf16, so it carries no scale companion and takes no
        # dequant.
        _declare_parameters(
            self,
            *(f"{leaf}_{FP8_SCALE_SUFFIX}" for leaf in DSA_SCALED_PROJECTIONS),
        )
        self.indexer = Glm5NextDSAIndexer(text_config)

    # The one place this class turns heads into its heads. It sits in the class
    # preamble because both halves below read it: the projections' widths and the
    # absorb operands' widths are the same head count seen from two sides.

    def _heads_per_rank(self) -> int:
        """This rank's share of the attention heads. Floored at 1.

        A method and not a constructor value. ``self.num_attention_heads`` stays
        the model's count for the life of the module -- it is what the checkpoint,
        the config and every closed-form comment mean by "heads" -- and this is the
        per-rank view of it. Overwriting the attribute instead would make every
        remaining reader of it silently per-rank, and would need the world size in
        ``__init__``, which this class is not given: it is constructed with the text
        config alone.

        The world size comes from :func:`_resolve_world_size`, which is also what
        the root binds ``self.world_size`` to and therefore what the shard table's
        width callbacks are handed. One source, so the width a rank's weight is
        sliced to and the width this class expects cannot come from two different
        answers. It is read per call rather than cached because tests inject the
        world size by monkeypatching that function, and a value cached in
        ``__init__`` would freeze the answer before the injection.

        Floored at 1 by ``_per_rank``, reused rather than re-derived so this class
        and the shard table floor identically. ``Glm5NextKDAAttention`` divides
        through the same function once, in its ``__init__``, which is why the KDA
        width functions divide nowhere: their attribute is already per-rank. MLA
        needs the division here because its attribute is not.
        """
        return _per_rank(self.num_attention_heads, _resolve_world_size())

    # The five low-rank projections run on the ``mla_projection`` NKI kernel,
    # because both substrate candidates refuse this checkpoint's widths. Every
    # number below is computed from the config and not one is read from a weight's
    # shape, so a mis-shaped checkpoint is caught rather than adopted.
    #
    # There is no torch matmul anywhere below. The sibling linear-attention class
    # projects with ``x @ w.t()`` and is right to -- its widths are small and no
    # kernel refuses them -- but here a torch matmul would be a fallback for work
    # the kernel does. That is also why the weights are transposed once, below,
    # rather than per call.
    #
    # No rotary parameter is allocated and no rotary slice is computed, because
    # ``qk_rope_head_dim`` is 0 on this checkpoint and that 0 is a value rather than
    # a placeholder.

    #: Attribute the transposed projection weights are cached on. A plain
    #: attribute and not a buffer: a buffer enters ``state_dict()``, which would
    #: double every projection weight in a saved checkpoint.
    PREPARED_WEIGHTS_ATTR = "_prepared_projection_weights"

    #: The checkpoint tensors the forward stops reading once
    #: :meth:`prepare_projection_weights` has transposed them: the five
    #: projection weights and the four scale grids of the quantised ones
    #: (``kv_b_proj`` is bf16 and has none). The load releases these.
    RELEASED_AFTER_PREP: tuple[str, ...] = (
        "q_a_proj_weight",
        "q_b_proj_weight",
        "kv_a_proj_with_mqa_weight",
        "kv_b_proj_weight",
        "o_proj_weight",
        *(f"{leaf}_{FP8_SCALE_SUFFIX}" for leaf in DSA_SCALED_PROJECTIONS),
    )

    def projection_widths(self) -> tuple[tuple[str, int, int], ...]:
        """The five sites as ``(name, in_features, out_features)``, closed form.

        Each width is derived from the config and named where it comes from, so the
        expectation a test compares against is not a transcription of the same
        literal the code used.

        On this checkpoint the rotary head width is 0, so the query head width is
        the nope width alone and the latent width is the rank alone. Both sums are
        written out anyway: on a config that had a rotary slice the bare value would
        be short by exactly that slice, which is the same reason
        ``_resolve_mla_head_size`` sums rather than takes the rank.

        The head count is this rank's, so these are the five widths of the weights
        this rank holds: a checkpoint's projection is sliced to the rank's heads at
        load time, and this method is what ``prepare_projection_weights`` checks the
        loaded weight against. The three widths that carry the head count narrow
        with the world size and the other two do not, because the latent projections
        are replicated.
        """
        heads = self._heads_per_rank()
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        return (
            ("q_a_proj", self.hidden_size, self.q_lora_rank),
            ("q_b_proj", self.q_lora_rank, heads * qk_head_dim),
            (
                "kv_a_proj_with_mqa",
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            (
                "kv_b_proj",
                self.kv_lora_rank,
                heads * (self.qk_nope_head_dim + self.v_head_dim),
            ),
            ("o_proj", heads * self.v_head_dim, self.hidden_size),
        )

    def prepare_projection_weights(self) -> int:
        """Transpose the five projection weights once. Returns how many.

        A matmul contracts the partition axis, so the kernel needs each weight
        contraction-major, ``[in_features, out_features]``, while a checkpoint
        stores the torch orientation, ``[out_features, in_features]``. Transposing
        per call would copy up to 64 MB every time; a projection weight never
        changes, so it is transposed here, once, when the weights are loaded.

        Each weight is checked against the closed-form widths before it is
        transposed. A checkpoint whose projection is the wrong shape fails here,
        with the site named, instead of reaching the kernel as a geometry it would
        accept and quietly compute the wrong thing with.
        """
        prepared: dict[str, torch.Tensor] = {}
        for name, idim, odim in self.projection_widths():
            weight = getattr(self, f"{name}_weight", None)
            if weight is None:
                raise ValueError(
                    f"{name}_weight is declared but not materialised; load the "
                    f"checkpoint before preparing the projection weights"
                )
            if tuple(weight.shape) != (odim, idim):
                raise ValueError(
                    f"{name}_weight is {tuple(weight.shape)}; this config's "
                    f"closed form is [out_features, in_features] = "
                    f"{(odim, idim)}"
                )
            # The dequant runs in the checkpoint's own
            # ``[out_features, in_features]`` orientation, where the scale grid
            # matches by construction, and only then is the weight transposed by the
            # line below.
            #
            # The hop to the host comes first, before the dequant and not after it.
            # The dequant upcasts the weight to fp32 and multiplies it by a scale grid
            # broadcast to the weight's own shape, and that broadcast is
            # ``repeat_interleave`` twice, which torch implements as an expand
            # followed by a contiguous copy. So a device-resident weight put through
            # it builds an fp32 copy and materialises a zero-stride view there -- the
            # second being the refusal this order exists to remove. Four of the five
            # MLA projections arrive as fp8 bytes in the published checkpoint, so this
            # is the ordinary path and not an edge.
            #
            # Everything below therefore runs on host memory: the dequant, the upcast
            # and the transpose whose copy ``.contiguous()`` forces, because ``.t()``
            # alone is a view and the kernel loads from memory. One move puts the
            # finished operand back.
            device = weight.device
            (weight,) = _on_the_host(weight)
            weight = self._dequantised_projection_weight(name, weight)
            (operand,) = _on_the_device(
                device, weight.to(torch.float32).t().contiguous()
            )
            prepared[name] = operand
        setattr(self, self.PREPARED_WEIGHTS_ATTR, prepared)
        return len(prepared)

    def _dequantised_projection_weight(
        self, name: str, weight: torch.Tensor
    ) -> torch.Tensor:
        """One projection weight in a real dtype, dequantised if it arrived as fp8.

        The published checkpoint stores four of the five MLA projections as
        blockwise-FP8 bytes with a ``weight_scale_inv`` companion per 128x128 tile.
        Using those bytes as if they were real numbers computes the wrong function
        at exactly the right shapes, which no shape check can see.

        The test is the dtype and never a config flag, because a flag can be wrong,
        absent or stale while a weight that is fp8 bytes is fp8 bytes. So the three
        cases are decided by what actually arrived:

        * a real dtype is returned unchanged;
        * fp8 bytes with their scale materialised are dequantised;
        * fp8 bytes with no scale materialised raise, naming the site and the
          parameter that is missing.

        The third case is the point: the one thing this method must never do is
        continue quietly when the scale is absent, and a refusal that names the site
        is also distinguishable from a skipped test.

        The dequant itself is the fork's own ``dequantise_blockwise``; this method
        chooses no arithmetic of its own. It returns fp32, so the caller's following
        ``.to(torch.float32)`` stays a no-op rather than a second conversion.

        Both operands are on the host when the dequant runs. The caller hands a host
        copy of the weight and this method takes one of the scale grid, because the
        dequant broadcasts that grid with ``repeat_interleave`` and the copy behind
        it is refused on a Neuron tensor.
        """
        if not _is_fp8_dtype(weight.dtype):
            return weight
        scale_name = f"{name}_{FP8_SCALE_SUFFIX}"
        scale = getattr(self, scale_name, None)
        if scale is None:
            raise ValueError(
                f"{name}_weight arrived as {weight.dtype} blockwise-FP8 bytes "
                f"but {scale_name} is not materialised, so the bytes cannot be "
                f"dequantised; load the checkpoint's {FP8_SCALE_SUFFIX} "
                f"companion for {name} before preparing the projection weights"
            )
        (scale,) = _on_the_host(scale)
        return dequantise_blockwise(weight, scale)

    def _prepared_weight(self, name: str) -> torch.Tensor:
        """One prepared weight, or a refusal naming what was not done.

        Refusing is what makes "never per call" checkable. If this fell back to
        transposing on demand, the per-call copy the section exists to avoid
        would come back silently and nothing would report it.
        """
        prepared = getattr(self, self.PREPARED_WEIGHTS_ATTR, None)
        if not prepared:
            raise ValueError(
                "prepare_projection_weights() has not run; the projection "
                "weights are transposed once at load time, never per call"
            )
        return prepared[name]

    # The absorb operands. The projections above work in the head width 256 while
    # the sparse attention kernel works in the latent rank 512, and two per-head
    # matmuls cross between them. This part prepares that kernel's two weight
    # operands once, at load time, and does nothing else.
    #
    # Both are halves of one weight this class already prepares, ``kv_b_proj``: it
    # expands a latent into a per-head key and value pair, and the projections above
    # split its output on that boundary. This splits the weight on the same
    # boundary, which is what lets the two matmuls be absorbed into the query and
    # into the attention output. No second projection is read, no new parameter is
    # declared, and no checkpoint tensor is touched.
    #
    # It runs once rather than per call: the split reshapes, slices and permutes a
    # 512 x 32,768 weight, and it never changes. The accessor refuses instead of
    # building on demand, because refusing is what makes "never per call"
    # checkable.
    #
    # The two orientations are not symmetric, and that asymmetry is the design.
    # ``nc_matmul`` contracts the partition axis, so each operand must present its
    # own contraction extent there:
    #   absorb-in  turns ``query [S, H, 256]`` into ``[S, H, 512]``, contracting
    #              the head width, so ``W_UK`` is ``[H, 256, 512]``;
    #   absorb-out turns the kernel's ``[S, H, 512]`` into ``[S, H, 256]``,
    #              contracting the latent, so ``W_UV`` is ``[H, 512, 256]``.
    # ``W_UV`` is therefore the value half as it already sits in the prepared
    # weight, and ``W_UK`` is the key half transposed. Storing ``W_UK`` the other
    # way round would put a transpose back on the per-forward path, which is the one
    # cost this preparation exists to avoid.

    #: Attribute the two absorb operands are cached on. A plain attribute for the
    #: same reason ``PREPARED_WEIGHTS_ATTR`` is one: a buffer enters
    #: ``state_dict()``, which would duplicate a 32 MB weight in every saved
    #: checkpoint.
    ABSORB_WEIGHTS_ATTR = "_prepared_absorb_weights"

    def absorb_widths(self) -> tuple[tuple[str, int, int, int], ...]:
        """The two absorb operands as ``(name, heads, contraction, out_features)``.

        Closed form from the config and never read off a weight's shape, so a
        mis-shaped checkpoint is caught rather than adopted -- the rule
        ``projection_widths`` states for the five projections, applied here.

        The head count is this rank's, and it has to be: both operands are split
        from the prepared ``kv_b_proj``, which under tensor parallelism holds only
        this rank's heads, so an expectation written in the model's full head count
        would refuse a correctly loaded weight. The two contraction extents --
        ``qk_nope_head_dim`` and ``kv_lora_rank`` -- are not touched, because
        neither is per-head: the latent is replicated across ranks and the head
        width is a property of one head.
        """
        heads = self._heads_per_rank()
        latent = self.kv_lora_rank
        return (
            ("W_UK", heads, self.qk_nope_head_dim, latent),
            ("W_UV", heads, latent, self.v_head_dim),
        )

    def prepare_absorb_weights(self) -> int:
        """Split the prepared ``kv_b_proj`` into the two absorb operands, once.

        Returns how many operands were built, so a caller can tell this ran from a
        call that had nothing to do.

        The input is the prepared weight and not the checkpoint parameter, and the
        difference is load-bearing. ``_prepared_weight`` returns ``kv_b_proj`` after
        the dequantisation and the one-time transpose, so it is
        ``[kv_lora_rank, heads * (nope + v)]`` in a real dtype. Splitting the raw
        parameter instead would, on a checkpoint that stored this weight as fp8
        bytes, split bytes rather than numbers and compute the wrong function at
        exactly the right shapes.

        The split is a rearrangement and nothing else. Every element of both
        operands is an element of the prepared weight, in the prepared weight's own
        dtype: no cast, no scale, no arithmetic.

        The head count below is this rank's, and every line after it -- the closed
        form, the reshape, the two permutes, the width comparison -- is already
        per-rank arithmetic once "heads" means this rank's heads. The refusal it
        raises is the one that matters under tensor parallelism, because a
        full-count expectation against a rank's weight would stop a correct load.
        """
        prepared = self._prepared_weight("kv_b_proj")
        heads = self._heads_per_rank()
        nope = self.qk_nope_head_dim
        vdim = self.v_head_dim
        latent = self.kv_lora_rank
        closed_form = (latent, heads * (nope + vdim))
        if tuple(prepared.shape) != closed_form:
            raise Glm5NextMLADecodeError(
                f"the prepared kv_b_proj is {tuple(prepared.shape)}; this "
                f"config's closed form is [kv_lora_rank, heads * (nope + v)] = "
                f"{closed_form}, so the absorb split would silently mix heads"
            )
        per_head = prepared.reshape(latent, heads, nope + vdim)
        # Both halves relayout on a host copy. ``per_head`` is a dense view of a
        # device-resident prepared weight, and each half is a strided slice before its
        # permute even runs, so a Neuron tensor refuses the copy. The slice and the
        # permute both go into the callback, where they run on the host copy. Two
        # copies of ``per_head`` cross the boundary rather than one, which is a
        # load-time cost that keeps every strided read inside the one helper.
        operands = {
            # [latent, heads, nope] -> [heads, nope, latent]: the key half,
            # transposed, because absorb-in contracts the head width.
            "W_UK": _relaid_out_on_the_host(
                per_head, lambda host: host[:, :, :nope].permute(1, 2, 0)
            ),
            # [latent, heads, v] -> [heads, latent, v]: the value half as it
            # already stands, because absorb-out contracts the latent.
            "W_UV": _relaid_out_on_the_host(
                per_head, lambda host: host[:, :, nope:].permute(1, 0, 2)
            ),
        }
        for name, exp_heads, contraction, out_features in self.absorb_widths():
            got = tuple(operands[name].shape)
            want = (exp_heads, contraction, out_features)
            if got != want:
                raise Glm5NextMLADecodeError(
                    f"{name} came out {got}; this config's closed form is {want}"
                )
        setattr(self, self.ABSORB_WEIGHTS_ATTR, operands)
        return len(operands)

    def _absorb_weight(self, name: str) -> torch.Tensor:
        """One absorb operand, or a refusal naming what was not done.

        Refusing instead of building on demand is what makes "once, at load time"
        checkable, and the failure it prevents is a silent per-call rebuild that
        nothing reports.
        """
        operands = getattr(self, self.ABSORB_WEIGHTS_ATTR, None)
        if not operands:
            raise Glm5NextMLADecodeError(
                "prepare_absorb_weights() has not run; the absorb operands are "
                "split from the prepared kv_b_proj once at load time, never per "
                "call"
            )
        return operands[name]

    def _latent_norm(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """RMSNorm on a low-rank latent: ``x / sqrt(mean(x**2) + eps) * gain``.

        The two latent norms sit between the projections, so they are applied here:
        a projection chain that emitted un-normalised latents would be numerically
        wrong.
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        return normed * gain.to(torch.float32)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, no-rotary key and value from hidden states. Four kernel calls.

        ``hidden_states`` is ``[tokens, hidden_size]``. The three returns are
        ``[tokens, heads, qk_nope_head_dim]``, ``[tokens, heads,
        qk_nope_head_dim]`` and ``[tokens, heads, v_head_dim]``.

        The chain is the low-rank one this checkpoint declares: compress to a rank,
        normalise the latent, expand to the head width. The key and value come out
        of one expansion and are split, which is why this is four calls and not
        five.

        The head count is this rank's. This method is the dense path, kept as the
        oracle the absorbed decode path is compared against, so it needs no
        reduction of its own -- but it reads ``projection_widths`` below, and the
        key-value reshape would refuse a correctly loaded weight under tensor
        parallelism if the count here were the model's.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        widths = {name: (idim, odim) for name, idim, odim in self.projection_widths()}
        x = hidden_states.to(torch.float32)
        if x.ndim != 2 or int(x.shape[1]) != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(x.shape[0])
        heads = self._heads_per_rank()

        q_latent = mla_projection(x, self._prepared_weight("q_a_proj"))
        q_latent = self._latent_norm(q_latent, self.q_a_layernorm_weight)
        query = mla_projection(q_latent, self._prepared_weight("q_b_proj"))
        query = query.reshape(tokens, heads, widths["q_b_proj"][1] // heads)

        kv_latent = mla_projection(x, self._prepared_weight("kv_a_proj_with_mqa"))
        kv_latent = self._latent_norm(kv_latent, self.kv_a_layernorm_weight)
        key_value = mla_projection(kv_latent, self._prepared_weight("kv_b_proj"))
        key_value = key_value.reshape(
            tokens, heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key_nope = key_value[..., : self.qk_nope_head_dim]
        value = key_value[..., self.qk_nope_head_dim :]

        out_dtype = hidden_states.dtype
        return (
            query.to(out_dtype),
            key_nope.contiguous().to(out_dtype),
            value.contiguous().to(out_dtype),
        )

    # The absorbed decode path needs the normalised latent that ``project_qkv``
    # computes as a local and then consumes. A ``return_latent`` keyword on
    # ``project_qkv`` would hand that back while still paying for its fourth call,
    # the ``kv_b_proj`` expansion from 512 to 32,768 -- and the entire point of
    # absorbing ``W_UK`` and ``W_UV`` is that the decode path never performs that
    # expansion, so the keyword would make the absorbed path slower than the path
    # it replaces.
    #
    # So the latent has a sibling method of its own, below, whose three lines repeat
    # ``project_qkv``'s first three rather than being factored out of them. Visible
    # repetition beats an invisible rewrite of a method other code stands on, so
    # ``project_qkv`` above is untouched.

    def project_query_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The normalised query latent: one kernel call, ``[tokens, q_lora_rank]``
        in float32.

        The DSA indexer's ``wq_b`` projection contracts ``q_lora_rank``, so the
        indexer's input is this latent, exactly as the reference passes it in as an
        argument. The DSA layer calls this once per phase and hands the result to
        the indexer; :meth:`project_query_and_latent` calls it too, so the value has
        one producer rather than two copies.

        float32 on purpose, and the caller is told rather than left to infer: the
        kernel returns float32 by contract and the norm runs in float32, while the
        sibling casts back to the model dtype at its own return because that is what
        its callers consume. A caller that needs another dtype casts once, itself.

        Recorded cost: ``attend()`` recomputes this latent inside
        ``project_query_and_latent``, so a DSA layer that calls the indexer and then
        ``attend()`` computes it twice -- one call of a
        ``[tokens, 4096] x [4096, 1536]`` projection per layer per phase. Threading
        the latent into ``attend()`` would change a signature with its own callers.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        x = hidden_states.to(torch.float32)
        if x.ndim != 2 or int(x.shape[1]) != self.hidden_size:
            raise Glm5NextMLADecodeError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        q_latent = mla_projection(x, self._prepared_weight("q_a_proj"))
        return self._latent_norm(q_latent, self.q_a_layernorm_weight)

    def project_query_and_latent(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query and the normalised KV latent. Three kernel calls, not four.

        ``hidden_states`` is ``[tokens, hidden_size]``. The returns are
        ``query [tokens, heads, qk_nope_head_dim]`` and
        ``kv_latent [tokens, kv_lora_rank]``.

        This is the absorbed path's entry into the projections: the query at the
        head width, and the compressed KV latent before any expansion. The expansion
        is what ``W_UK`` and ``W_UV`` absorb, so performing it here would be paying
        for the work the absorb removes.

        The latent is returned normalised because that is what the cache stores and
        what the kernel contracts against; returning the un-normalised compression
        would put a norm on every reader of the cache instead of one norm on the
        writer.

        The head count is this rank's, so the returned query is
        ``[tokens, heads_on_this_rank, qk_nope_head_dim]``. This is the one site in
        the class where a full-count head reshape would not have raised: the reshape
        below divides the projection's own output width by the head count, so on two
        ranks of a 64-head model a rank's ``[tokens, 8192]`` query would reshape
        cleanly to ``[tokens, 64, 128]`` instead of ``[tokens, 32, 256]`` -- right
        element count, wrong heads, no error, and a head width of 128 the config
        never mentions.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        widths = {name: (idim, odim) for name, idim, odim in self.projection_widths()}
        x = hidden_states.to(torch.float32)
        if x.ndim != 2 or int(x.shape[1]) != self.hidden_size:
            raise Glm5NextMLADecodeError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(x.shape[0])
        heads = self._heads_per_rank()

        q_latent = self.project_query_latent(hidden_states)
        query = mla_projection(q_latent, self._prepared_weight("q_b_proj"))
        query = query.reshape(tokens, heads, widths["q_b_proj"][1] // heads)

        kv_latent = mla_projection(x, self._prepared_weight("kv_a_proj_with_mqa"))
        kv_latent = self._latent_norm(kv_latent, self.kv_a_layernorm_weight)

        out_dtype = hidden_states.dtype
        return query.to(out_dtype), kv_latent.to(out_dtype)

    def project_output(
        self,
        attn_out: torch.Tensor,
        collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The output projection. One kernel call.

        ``attn_out`` is ``[tokens, heads, v_head_dim]`` or the same flattened to
        ``[tokens, heads * v_head_dim]``; the return is ``[tokens, hidden_size]``.
        Both input forms are accepted because the decode path's layout is its own
        choice.

        This is the row-parallel site, and it reduces. Under tensor parallelism the
        head width is this projection's input, so each rank holds a slice of the
        weight's columns, contracts it against its own heads, and produces a partial
        sum at the full output width -- every rank's result is the same shape and
        none of them is the answer. Without the collective the model returns one
        rank's fraction of every attention output, at exactly the right shape, which
        is the failure mode that does not announce itself.

        The group and the statement form are the fork's own convention, used at
        every shipped row-parallel site. :func:`_resolve_tp_group` is where this file
        resolves it, and it returns ``None`` at world size 1, so a single-rank run
        performs no collective and imports no vllm symbol.

        Two readings worth stating, because both could be wrong in a way tests would
        not catch. The reduction is in place: the statement form discards a return
        value, so it assumes ``all_reduce`` writes through its argument -- which is
        what every shipped row-parallel site assumes, and which is safe against
        aliasing here because ``mla_projection`` returns a fresh tensor rather than a
        view of a cached weight. And the sum happens before the cast back to the
        input dtype, because rounding each rank's fraction to bfloat16 first and
        adding after would round the parts instead of the whole.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        heads = self._heads_per_rank()
        expected_width = heads * self.v_head_dim
        x = attn_out.to(torch.float32)
        if x.ndim == 3:
            x = x.reshape(int(x.shape[0]), -1)
        if x.ndim != 2 or int(x.shape[1]) != expected_width:
            raise ValueError(
                f"attn_out must be [tokens, {heads}, "
                f"{self.v_head_dim}] or [tokens, {expected_width}]; got "
                f"{tuple(attn_out.shape)}"
            )
        projected = mla_projection(x.contiguous(), self._prepared_weight("o_proj"))
        if collector is not None:
            # The partial is cloned and the input is not. The reduction below writes
            # through its argument, so a tap holding ``projected`` would come back
            # holding the sum instead of this rank's share.
            collector += [x, projected.clone()]
        # Sum this rank's partial with every other rank's. ``None`` means one
        # rank, where the partial already is the whole sum.
        group = _resolve_tp_group()
        if group is not None:
            group.all_reduce(projected)
        whole = projected.to(attn_out.dtype)
        if collector is not None:
            collector.append(whole)
        return whole

    # The decode chain turns hidden states into attention output for this layer,
    # and it is one method rather than a prefill method and a decode method: a
    # decode step is compared against the matching slice of a prefill run, and that
    # comparison is only worth taking if both sides are the same code at different
    # token counts.
    #
    # The chain is ``project_query_and_latent``, the cache write, the cache read,
    # absorb-in through ``W_UK``, sparse attention, absorb-out through ``W_UV``,
    # and ``project_output``.
    #
    # The expansion never happens here. ``kv_b_proj`` expands a 512 latent to
    # 32,768 per token; the absorbed chain multiplies the query by ``W_UK`` and the
    # attention output by ``W_UV`` instead, so the expansion is never computed on
    # the per-token path at all. That is why this is not simply ``project_qkv``
    # followed by dense attention.
    #
    # ``topk_indices`` and ``softmax_scale`` are both caller arguments: the indices
    # come from the DSA indexer and the scale is the layer's, so neither is derived
    # here.

    def attend(
        self,
        hidden_states: torch.Tensor,
        latent_cache: torch.Tensor,
        start_position: torch.Tensor | int,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        batch_size: int = 1,
        prefill_end_position: torch.Tensor | int | None = None,
        collector: list[torch.Tensor] | None = None,
        *,
        active_mla_query_rows: int | None = None,
        block_table_row: torch.Tensor,
        latent_slots: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """One layer's MLA attention. Returns ``[tokens, hidden_size]``.

        ``hidden_states`` is ``[tokens, hidden_size]``: a prefill passes all its
        tokens at once and a decode step passes one. ``latent_cache`` is the whole
        latent bank in the model's own declared spec layout,
        ``[slots, NUM_LATENT_KV_HEADS, head_size]``, one latent per token, bf16, and
        ``block_table_row`` names which of its ``page_size``-row blocks this request
        holds, as a ``[pages, 1]`` int32 column with ``-1`` in unused entries.
        ``start_position`` is the row this step's first token occupies in the window
        those blocks make, and ``latent_slots`` is ``[tokens]`` int64: the physical
        bank row each of this step's tokens is written to.

        The bank travels whole and the pages are named, which is what lets a request
        hold blocks that are not one ascending run. The kernel assembles the window
        from the table, so nothing here slices the bank, nothing needs spare blocks
        past a request's last one, and the write goes to the physical rows the runner
        computed from the same table.

        ``active_mla_query_rows`` is a static prefix selected by the runner. Only
        the sparse attention call uses this shorter width; projections and cache
        writes keep all rows, and zero padding restores the attention output to the
        ordinary operator width before absorb-out.

        The window's length is a constant and the position is a tensor. A graph is
        captured once and replayed at every position, so a length derived from a
        position would compile a context of exactly the captured length and match no
        other step. The table's width is fixed for the bucket, so the window it names
        is too, and the position arrives as a tensor so no host read of it can
        specialise the graph either.

        What bounds the read, now that the length does not: rows of the window past
        this sequence's context hold other pages, so they must never be attended.
        Two things prevent it and neither is a length -- the indexer builds
        candidates only out of ``seq_lens``, so no index beyond the context exists to
        select, and the sparse kernel masks its ``-1`` rows itself. Nothing between
        the indexer and this method may clamp or refill those indices, because a
        clamp would turn an out-of-context index into an in-context one and attend a
        row this bound exists to exclude. A caller that hands indices outside the
        context is the one error this method cannot catch.

        ``batch_size`` is a parameter rather than inferred because
        ``block_table_row`` is one request's row and the kernel assembles one window
        per call, so a caller serving more than one sequence would have to loop. The
        named refusal below says so rather than attending one request's window with
        another request's queries.
        """
        if int(batch_size) != 1:
            raise Glm5NextMLADecodeError(
                f"the MLA decode path serves one sequence at a time "
                f"(batch_size == 1); got batch_size={batch_size}. One block "
                f"table names one request's window, and the kernel assembles one "
                f"window per dispatch, so a larger batch would attend one "
                f"request's rows with another request's queries"
            )
        if hidden_states.ndim != 2 or int(hidden_states.shape[1]) != self.hidden_size:
            raise Glm5NextMLADecodeError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(hidden_states.shape[0])
        if active_mla_query_rows is not None:
            if type(active_mla_query_rows) is not int or not (
                1 <= active_mla_query_rows <= tokens
            ):
                raise Glm5NextMLADecodeError(
                    f"active_mla_query_rows must be a static integer in [1, {tokens}]; "
                    f"got {active_mla_query_rows!r}"
                )
            if topk_indices.ndim != 2 or topk_indices.shape[0] != tokens:
                raise Glm5NextMLADecodeError(
                    f"topk_indices must have {tokens} query rows before the active "
                    f"prefix is selected; got {tuple(topk_indices.shape)}"
                )
        latent = self.kv_lora_rank
        want_cache = (self.NUM_LATENT_KV_HEADS, self.head_size)
        if latent_cache.ndim != 3 or tuple(latent_cache.shape[1:]) != want_cache:
            raise Glm5NextMLADecodeError(
                f"latent_cache must be [slots, {self.NUM_LATENT_KV_HEADS}, "
                f"{self.head_size}] -- this layer's own declared cache spec; got "
                f"{tuple(latent_cache.shape)}"
            )
        slots = int(latent_cache.shape[0])
        page = int(page_size)
        if page < 1 or slots % page:
            raise Glm5NextMLADecodeError(
                f"the latent bank is paged, so its {slots} slot(s) must be whole "
                f"blocks of page_size; got page_size={page_size!r}"
            )
        if block_table_row.ndim != 2 or int(block_table_row.shape[1]) != 1:
            raise Glm5NextMLADecodeError(
                f"block_table_row must be [pages, 1] -- the column the kernel reads "
                f"the page number out of; got {tuple(block_table_row.shape)}"
            )
        window = int(block_table_row.shape[0]) * page
        # A shape check and not a position check. The window the table names has to
        # be long enough to hold this step's own tokens, and that reads off the
        # shapes, which are real even on a meta tensor. Whether the position plus
        # these tokens stays inside the request's own pages is arithmetic on values,
        # and checking it here would need a host read of a tensor in a traced region.
        # The runner owns that check: it builds the row and it computes the position,
        # so it can refuse before the trace.
        if tokens > window:
            raise Glm5NextMLADecodeError(
                f"these {tokens} token(s) cannot fit the window this block table "
                f"names, {int(block_table_row.shape[0])} block(s) of {page} = "
                f"{window} slot(s); the table's width is fixed for the bucket and "
                f"the runner sizes it, so a window this short is the caller's error"
            )
        if latent_slots.ndim != 1 or int(latent_slots.shape[0]) != tokens:
            raise Glm5NextMLADecodeError(
                f"latent_slots must carry one physical bank row per token, [{tokens}]; "
                f"got {tuple(latent_slots.shape)}"
            )
        start = _int64_scalar(start_position, latent_cache.device)

        query, kv_latent = self.project_query_and_latent(hidden_states)

        # The cache write. One latent per token, into this layer's single KV head.
        # Done before the read below, so a decode step attends to its own token as
        # well as its context -- the same set a prefill of the same tokens would see,
        # which is what makes the two comparable.
        #
        # An indexed write rather than a slice, because the rows this step writes are
        # its request's physical bank rows, wherever the allocator put them.
        # ``index_copy_`` takes them as a tensor and no arithmetic on the position
        # happens here. The row count is ``tokens``, a shape, so the write's own
        # shape is constant too.
        offsets = torch.arange(tokens, device=latent_cache.device)
        # A padded chunk carries rows that are not the sequence's, and they must not
        # reach slots the request does not hold: the chunk arrives padded up to its
        # bucket while the pages it was allocated cover the real length. Clamping the
        # offset at the last real row collapses those rows onto it, index and value
        # together -- the gather takes each row's own latent and the clamped rows
        # take the last real one -- so every write lands inside the request's own
        # pages and writes that slot's own latent. The repeated indices are therefore
        # idempotent: no order of the writes can change the result. An unpadded chunk
        # clamps nothing, so its write is unchanged.
        #
        # A clamp and a gather rather than two ``where``s, deliberately: ``where``
        # promotes when its arms differ and the eager Neuron backend refuses a
        # dtype-converting copy of a tensor already on the device, so a selection
        # that cannot promote at all is the one that stays safe as the arms change.
        if prefill_end_position is not None:
            end = _int64_scalar(prefill_end_position, latent_cache.device)
            offsets = torch.minimum(offsets, end - start - 1)
            kv_latent = kv_latent.index_select(0, offsets)
        # The write's rows are the runner's, computed from the same block table that
        # travels beside them, and they already carry the clamp above: a padded
        # chunk's trailing rows repeat the last real slot, so the repeated writes are
        # idempotent for the same reason the clamped gather is.
        rows = latent_slots
        if collector is not None:
            # Taken before the write and after the clamp above, so these hold the
            # very values and slots the write carries -- physical bank rows, which is
            # what a dump reader has to index the bank by. Narrowed for the file
            # only; the write keeps the int64 index it needs.
            collector.extend([kv_latent, rows.to(torch.int32)])
        # One cast, used twice, so the value that persists and the value this step
        # reads back cannot differ.
        written = kv_latent.to(latent_cache.dtype)
        # Keep the update shape equal to the input so the compiler chains all
        # request writes to this bank. Recreated 2D views lose earlier updates.
        latent_cache.index_copy_(0, rows, written.unsqueeze(1))

        # The cache read. The whole bank, as a view, with the block table beside it.
        # The kernel assembles the window the table names and no copy of it is made
        # here: a window-sized copy per layer per step is exactly the cost the paging
        # removes, and the table is what makes the pages it gathers the request's own.
        #
        # This step's rows travel beside the bank rather than inside it. Reading them
        # back out of the bank would leave the kernel depending on when the write
        # above becomes visible inside one graph, and on device it observed the rows
        # from before the write. ``written`` and the position hand the kernel the
        # same values the write carries, and it overlays them on the window it
        # assembles, so what it attends is right whether or not the write has landed.
        c_kv = latent_cache[:, 0, :]
        at = start.reshape(1, 1).to(torch.int32)
        if collector is not None:
            # One entry, as before: the dump's names are positional, so a second tap
            # here would rename every tensor after it. This is the tensor the kernel
            # gathers from, which is the bank rather than a window of it. A copy,
            # because the bank is the caller's and a later step writes into it: a view
            # would make the dump report the bank as it is when the files are written.
            collector.append(c_kv.detach().clone())

        from vllm_neuron.functional.attention.mla_absorb import mla_absorb
        from vllm_neuron.functional.attention.mla_sparse import mla_sparse_attention

        # Absorb-in. ``[S, H, 256] x [H, 256, 512] -> [S, H, 512]``: the query moves
        # into the latent space the kernel works in.
        q_lift = mla_absorb(query, self._absorb_weight("W_UK"))
        if int(q_lift.shape[2]) != latent:
            raise Glm5NextMLADecodeError(
                f"absorb-in produced a width of {int(q_lift.shape[2])}; the "
                f"sparse seam contracts {latent}"
            )

        if active_mla_query_rows is None or active_mla_query_rows == tokens:
            attended = mla_sparse_attention(
                q_lift,
                c_kv,
                topk_indices,
                softmax_scale,
                block_table_row=block_table_row,
                written=written,
                write_offset=at,
                page_size=page,
            )
        else:
            attended = mla_sparse_attention(
                q_lift[:active_mla_query_rows],
                c_kv,
                topk_indices[:active_mla_query_rows],
                softmax_scale,
                block_table_row=block_table_row,
                written=written,
                write_offset=at,
                page_size=page,
            )
            # The kernel returns float32, so the padding is restored in that dtype
            # before the model-dtype cast and absorb-out below.
            attended = torch.cat(
                (
                    attended,
                    attended.new_zeros(
                        (tokens - active_mla_query_rows, *attended.shape[1:])
                    ),
                ),
                dim=0,
            )
        if collector is not None:
            # The attention output, including restored padding, before absorb-out,
            # beside the full-width query used to select the sparse prefix.
            collector.extend([attended, q_lift])

        # Absorb-out. ``[S, H, 512] x [H, 512, 256] -> [S, H, 256]``: back to the
        # head width ``project_output`` consumes. The cast is here rather than inside
        # the kernel because the kernel returns float32 by contract while this
        # layer's chain carries the model dtype.
        reduced = mla_absorb(
            attended.to(hidden_states.dtype), self._absorb_weight("W_UV")
        )
        return self.project_output(reduced, collector)

    def forward(
        self,
        normed_hidden_states: torch.Tensor,
        *,
        latent_cache: torch.Tensor,
        pool_cache: torch.Tensor | tuple[torch.Tensor, ...],
        seq_lens: torch.Tensor | tuple[torch.Tensor, ...],
        start_position: torch.Tensor | int | tuple[torch.Tensor | int, ...],
        softmax_scale: float,
        max_seq_len: int,
        page_size: int,
        block_table_row: torch.Tensor | tuple[torch.Tensor, ...],
        latent_slots: torch.Tensor | tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor | None = None,
        tail: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        position: torch.Tensor | int | tuple[torch.Tensor | int, ...] | None = None,
        prefill_tail: torch.Tensor | None = None,
        prefill_end_position: torch.Tensor | int | None = None,
        collector: list[torch.Tensor] | None = None,
        active_mla_query_rows: int | None = None,
    ) -> torch.Tensor:
        """This module's whole contribution to one layer: select, then attend.

        Returns ``[tokens, hidden_size]`` -- the attention half's output, with no
        residual added and nothing normalised, because both belong to the layer.

        The input arrives normalised and the parameter name says so. The layer owns
        its pre-attention norm and its residual; this module owns the three calls
        between them. All three consume the same normalised tensor, which is why one
        argument carries it rather than three.

        Singleton execution composes the query latent, the indexer, and
        :meth:`attend`. Concurrent decode dispatches one singleton call per
        request. The latent bank is shared; page tables and side-cache views
        belong to each request. Concurrent prefill is refused before writes.

        The indices pass through unchanged. The ``-1`` mask lives inside the sparse
        kernel, so no filler, compaction, clamp or mask is applied between the
        indexer and ``attend()`` -- this hands over exactly what the indexer
        returned.

        Recorded cost, not paid here: ``attend()`` recomputes the query latent
        inside :meth:`project_query_and_latent`, so the latent this method computes
        for the indexer is computed a second time -- one
        ``[tokens, hidden_size] x [hidden_size, q_lora_rank]`` dispatch per layer
        per phase. Paying it means threading the latent into ``attend()``, which
        changes a signature its callers use positionally.

        The two carriers are the caller's and both are written in place:
        ``latent_cache`` is ``attend()``'s own contract and ``pool_cache``, ``tail``
        and ``prefill_tail`` are the indexer's -- the last of those is the same ring
        on the prefill leg, where the indexer seeds this chunk's remainder. See
        :meth:`Glm5NextDSAIndexer.forward` for what each means and why
        ``max_seq_len`` is a python int.

        ``block_table_row`` and ``latent_slots`` pass straight through to
        :meth:`attend`, which owns both contracts: the latent bank travels whole,
        the row names the blocks this request's window is built from, and the slots
        are where this step's own rows are persisted. ``page_size`` is the same
        number for both readers -- the indexer's pool pages and the latent bank's
        blocks are one page size, which the runner refuses to let disagree.
        """
        request_operands = {
            "pool_cache": pool_cache,
            "seq_lens": seq_lens,
            "start_position": start_position,
            "block_table_row": block_table_row,
            "latent_slots": latent_slots,
            "tail": tail,
            "position": position,
        }
        if any(isinstance(value, tuple) for value in request_operands.values()):
            if (
                slot_mapping is not None
                or prefill_tail is not None
                or prefill_end_position is not None
            ):
                raise ValueError("concurrent sparse prefill is not supported")
            if not all(isinstance(value, tuple) for value in request_operands.values()):
                raise ValueError("concurrent sparse decode needs tuples for all request operands")
            requests = len(pool_cache)
            if requests == 0 or any(
                len(value) != requests for value in request_operands.values()
            ):
                raise ValueError("concurrent sparse decode request operand counts must agree")
            if (
                normed_hidden_states.ndim != 2
                or normed_hidden_states.shape[0] != requests
            ):
                raise ValueError("concurrent sparse decode needs exactly one hidden row per request")
            if any(value is None for value in tail) or any(
                value is None for value in position
            ):
                raise ValueError("concurrent sparse decode needs each request's tail and position")
            if active_mla_query_rows is not None:
                raise ValueError("concurrent sparse decode does not accept an active prefill prefix")
            # The sparse/indexer kernels remain singleton kernels. Each call sees
            # its own pages, physical write slots, and views into its side caches.
            # Validate the whole batch above before the first in-place write.
            # Join in FP32 to preserve the singleton cast path before MHC post.
            return torch.cat(
                [
                    self.forward(
                        normed_hidden_states[index:index + 1],
                        latent_cache=latent_cache,
                        softmax_scale=softmax_scale,
                        max_seq_len=max_seq_len,
                        page_size=page_size,
                        collector=collector,
                        **{key: value[index] for key, value in request_operands.items()},
                    ).to(torch.float32)
                    for index in range(requests)
                ],
                dim=0,
            ).to(normed_hidden_states.dtype)
        q_latent = self.project_query_latent(normed_hidden_states)
        topk_indices = self.indexer(
            normed_hidden_states,
            q_latent,
            pool_cache,
            seq_lens,
            max_seq_len=int(max_seq_len),
            page_size=int(page_size),
            slot_mapping=slot_mapping,
            tail=tail,
            position=position,
            prefill_tail=prefill_tail,
            prefill_end_position=prefill_end_position,
            prefill_start_position=(
                None if prefill_end_position is None else start_position
            ),
        )
        if collector is not None:
                # The first five rows only, and as the indexer emitted them: a static
                # slice inside the trace, no cast, so a -1 stays the sentinel it is
                # rather than becoming a float. A shorter batch yields the rows it has.
            collector.append(topk_indices[:5])
        return self.attend(
            normed_hidden_states,
            latent_cache,
            start_position,
            topk_indices,
            float(softmax_scale),
            prefill_end_position=prefill_end_position,
            collector=collector,
            block_table_row=block_table_row,
            latent_slots=latent_slots,
            page_size=int(page_size),
            **(
                {"active_mla_query_rows": active_mla_query_rows}
                if active_mla_query_rows is not None else {}
            ),
        )


class Glm5NextDSALayer(nn.Module):
    """Decoder layer on the ``deepseek_sparse_attention`` half."""

    #: Attribute the family's attention module is bound to -- the weight map's own
    #: module path, which ``weight_loaders_fp8.py`` builds as
    #: ``f"{param_prefix}.self_attn"``.
    ATTENTION_ATTR = "self_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = DSA_LAYER_TYPE
        _declare_parameters(
            self,
            "input_layernorm_weight",
            "post_attention_layernorm_weight",
            # The same six mHC weights, flat on the layer for the same reason as
            # ``Glm5NextKDALayer``: the mHC mapping runs for every layer of the
            # stack, not only the linear-attention half.
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        )
        self.self_attn = Glm5NextMLAAttention(text_config)
        self.mlp = _build_mlp(text_config, layer_idx)
        # Resolved at construction, on the same ground the KDA sibling states.
        self.rms_norm_eps = float(text_config.rms_norm_eps)

    def bind_hyper_connection_sites(
        self, text_config: Glm5NextTextConfig, device: torch.device
    ) -> int:
        """Give this layer's two mHC sites the six tensors the load brought.

        The same one-line delegation as the KDA sibling: the rule itself lives once
        in :func:`_bind_hyper_connection_sites`, so the two halves cannot drift
        apart.
        """
        return _bind_hyper_connection_sites(self, text_config, device)

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-attention RMSNorm: ``x / sqrt(mean(x**2) + eps) * gain``.

        The same body as ``Glm5NextKDALayer._input_norm``. Sharing it would need
        either a module-level helper or a common base class, and both would move a
        class this file's other half owns, so the five lines are repeated instead.
        """
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        normed = normed * self.input_layernorm_weight.to(torch.float32)
        return normed.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        latent_cache: torch.Tensor,
        pool_cache: torch.Tensor | tuple[torch.Tensor, ...],
        seq_lens: torch.Tensor | tuple[torch.Tensor, ...],
        start_position: torch.Tensor | int | tuple[torch.Tensor | int, ...],
        softmax_scale: float,
        max_seq_len: int,
        page_size: int,
        block_table_row: torch.Tensor | tuple[torch.Tensor, ...],
        latent_slots: torch.Tensor | tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor | None = None,
        tail: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        position: torch.Tensor | int | tuple[torch.Tensor | int, ...] | None = None,
        prefill_tail: torch.Tensor | None = None,
        prefill_end_position: torch.Tensor | int | None = None,
        streams: torch.Tensor | None = None,
        collector: list[torch.Tensor] | None = None,
        active_mla_query_rows: int | None = None,
    ) -> torch.Tensor:
        """The sparse-attention half, mixed either by mHC or by a plain add.

        The attention half is one call: the query latent, the indexer and
        ``attend()`` are :meth:`Glm5NextMLAAttention.forward`'s body, and this layer
        calls it. The composition therefore has one definition -- a second copy here
        is how the two come to disagree about which tensor each callee consumes.

        The indices pass through unchanged, which is the whole point of the shape.
        The ``-1`` mask lives inside the sparse kernel, so no filler, compaction,
        clamp or mask is applied between the indexer and ``attend()``.

        Two routes, and the keyword picks one: ``streams`` absent is the plain
        residual add, and ``streams`` present runs the four-stream mHC pair around
        the same sparse attention half. One rule refuses both ways, in
        :func:`_mhc_attention_site`, and the attention call is written once as
        ``attention_half`` so the two routes cannot disagree about what they wrap.

        ``self.mlp`` is not called here: the feed-forward half is
        ``Glm5NextModel._ffn_half``'s and its mHC site is composed there. The six
        mHC weights sit flat on this layer and its two sites are bound after the
        load, by :meth:`bind_hyper_connection_sites`.

        The two carriers are the caller's, both written in place: ``latent_cache``
        is ``attend()``'s own contract and ``pool_cache``, ``tail`` and
        ``prefill_tail`` are the indexer's. See :meth:`Glm5NextDSAIndexer.forward`
        for what each means, which leg each belongs to, and why ``max_seq_len`` is a
        python int. ``block_table_row`` and ``latent_slots`` belong to the paged
        latent bank and are handed on unread: the attention half's callee owns both
        contracts, and a second reading here is a second authority on one extent.

        ``streams`` is ``[T, S, H]`` or ``None``; the return is ``[T, H]`` in the
        input dtype on the one-stream route and ``[T, S, H]`` in the streams' own
        dtype on the streams route, because
        :meth:`Glm5NextHyperConnection.mhc_post` computes the mix in fp32 and casts
        back to the residual's dtype before this forward hands the value on
        unchanged.
        """

        num_requests = len(pool_cache) if isinstance(pool_cache, tuple) else 1

        def attention_half(single_stream: torch.Tensor) -> torch.Tensor:
            if num_requests > 1 and single_stream.dtype == torch.bfloat16:
                if single_stream.dim() != 2 or single_stream.shape[0] != num_requests:
                    raise ValueError("DSA decode rows must match request carriers")
                normed = torch.cat(
                    [
                        self._input_norm(single_stream[row : row + 1]).to(torch.float32)
                        for row in range(num_requests)
                    ],
                    dim=0,
                ).to(single_stream.dtype)
            else:
                normed = self._input_norm(single_stream)
            if collector is not None:
                # What the attention half was actually handed: the stream the
                # hyper-connection site collapsed, and that stream normalised. The
                # norm is hoisted to a local so the tap and the call read one tensor.
                # ``extend`` rather than ``+=``: this is a closure over the caller's
                # name, and an augmented assignment would bind it locally and make
                # every read above unbound, on the untapped path too.
                collector.extend([single_stream, normed])
            attended = self.attention(
                normed,
                latent_cache=latent_cache,
                pool_cache=pool_cache,
                seq_lens=seq_lens,
                start_position=start_position,
                softmax_scale=float(softmax_scale),
                max_seq_len=int(max_seq_len),
                page_size=int(page_size),
                block_table_row=block_table_row,
                latent_slots=latent_slots,
                slot_mapping=slot_mapping,
                tail=tail,
                position=position,
                prefill_tail=prefill_tail,
                prefill_end_position=prefill_end_position,
                **({"collector": collector} if collector is not None else {}),
                **(
                    {"active_mla_query_rows": active_mla_query_rows}
                    if active_mla_query_rows is not None else {}
                ),
            )
            if collector is not None:
                collector.append(attended)
            return attended

        site = _mhc_attention_site(self, streams)
        if site is None:
            return hidden_states + attention_half(hidden_states)
        if num_requests > 1:
            return site.forward(streams, attention_half, num_requests=num_requests)
        return site.forward(streams, attention_half)


def _build_layer(
    text_config: Glm5NextTextConfig, layer_idx: int, layer_type: str, world_size: int
) -> nn.Module:
    """One decoder layer, family chosen by equality on ``layer_types``.

    Never by substring: ``"attention"`` is a substring of both family names, so a
    substring test would silently mis-partition the stack.
    """
    if layer_type == DSA_LAYER_TYPE:
        return Glm5NextDSALayer(text_config, layer_idx, world_size)
    if layer_type == KDA_LAYER_TYPE:
        return Glm5NextKDALayer(text_config, layer_idx, world_size)
    raise ValueError(
        f"layer {layer_idx} declares unrecognised attention family "
        f"{layer_type!r}; expected {KDA_LAYER_TYPE!r} or {DSA_LAYER_TYPE!r}"
    )


# ---------------------------------------------------------------------------
# The decoder stack.
# ---------------------------------------------------------------------------


#: How many layers' output streams the dump keeps. The residual error is localised in the
#: first few layers, and every further layer costs one graph output per rank.
DUMP_STREAM_LAYERS = 6


#: The attention-side taps of one layer, in the order its forward appends them, and
#: before ``attention_output`` because each is taken further inside the same call.
#: They exist only where the attention module carries the sparse indexer: the KDA
#: half emits no index rows and projects its output through no row-parallel matmul,
#: so there is nothing at those names to read there.
DUMP_DSA_TAP_NAMES = (
    "attn_hc_collapsed",
    "attn_input_normed",
    "index_rows",
    "latent_written",
    "write_rows",
    "cache_rows",
    "attended_latent",
    "q_lift",
    "o_proj_input",
    "o_proj_partial",
    "o_proj_reduced",
)

#: The layer's remaining taps, in the order its forward appends them.
#: ``routed_output`` and ``shared_output`` are each half's own result before the
#: layer reduces them, so they read as partials on a sharded run; ``mlp_output``
#: and ``attention_output`` are taken after the reduction and are whole.
DUMP_TAP_NAMES = (
    "attention_output",
    "mlp_input1",
    "router_logits",
    "router_topk_indices",
    "router_topk_weights",
    "router_affinities",
    "token_position_to_id",
    "routed_output",
    "shared_output",
    "mlp_output",
)


def dump_tap_layers(layers) -> tuple[int, ...]:
    """The layers tapped inside: the first layer holding experts, and the next one that does."""
    holding = [
        index for index, layer in enumerate(layers)
        if isinstance(layer.mlp, Glm5NextMoEBlock)
    ]
    if not holding:
        return ()
    first = holding[0]
    return (first, first + 1) if first + 1 in holding else (first,)


def dump_tap_names(layer: nn.Module) -> tuple[str, ...]:
    """The tap names one layer contributes, in the order its forward appends them."""
    indexer = getattr(getattr(layer, "attention", None), "indexer", None)
    names = DUMP_DSA_TAP_NAMES if indexer is not None else ()
    if getattr(layer.mlp, "shared_experts", None) is None:
        return names + tuple(name for name in DUMP_TAP_NAMES if name != "shared_output")
    return names + DUMP_TAP_NAMES


def _dump_layers(model: nn.Module):
    """The decoder layers, whether handed the stack, the root or a wrapped root."""
    holder = model
    for _ in range(3):
        layers = getattr(holder, "layers", None)
        if layers is not None:
            return layers
        holder = getattr(holder, "_model", None) or getattr(holder, "model", None)
        if holder is None:
            return None
    return None


def layer_dump_names(model: nn.Module) -> tuple[str, ...]:
    """The dump's extra outputs, named in the order the stack returns them."""
    layers = _dump_layers(model)
    if layers is None:
        return ()
    names = [f"after_layer_{index}" for index in range(min(DUMP_STREAM_LAYERS, len(layers)))]
    for tap in dump_tap_layers(layers):
        names += [f"layer{tap}_{suffix}" for suffix in dump_tap_names(layers[tap])]
    return tuple(names)


class Glm5NextModel(nn.Module):
    """The decoder stack, named ``model`` because the weight map's paths say so.

    Every mapped parameter outside the layer stack hangs here or on the root.
    """

    def __init__(self, config: Glm5NextConfig, world_size: int) -> None:
        super().__init__()
        text_config = config.text_config
        self.config = config
        self.text_config = text_config
        self.vocab_size = int(text_config.vocab_size)
        self.hidden_size = int(text_config.hidden_size)

        _declare_parameters(self, "embed_tokens_weight", "norm_weight")

        layer_types = list(text_config.layer_types or ())
        self.layers = nn.ModuleList(
            [
                _build_layer(text_config, layer_idx, layer_type, world_size)
                for layer_idx, layer_type in enumerate(layer_types)
            ]
        )

    def _rms_norm(self, hidden_states: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """``x / sqrt(mean(x**2) + eps) * gain``, computed in fp32 and cast back.

        One body for the two norms this class applies: each layer's post-attention
        (feed-forward) norm and the stack's final norm.

        The epsilon is the checkpoint's ``rms_norm_eps``, read off the config on
        every call rather than cached, so a fixture that edits the config between
        calls cannot be normalised with a stale value.
        """
        eps = float(self.text_config.rms_norm_eps)
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        normed = normed * gain.to(torch.float32)
        return normed.to(hidden_states.dtype)

    def _ffn_half(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        quant_config: Glm5NextQuantConfig,
        block_size: int | None,
        moe_group: object | None,
        tp_degree: int,
        expert_parallel_rank: int | torch.Tensor,
        collector: list[torch.Tensor] | None = None,
        num_requests: int = 1,
    ) -> torch.Tensor:
        """One layer's feed-forward contribution, without its residual add.

        The residual add is the caller's, so this method is only the sublayer:
        normalise with that layer's own post-attention gain, then run whichever MLP
        ``_build_mlp`` gave the layer.

        It lives here rather than in the layer forwards because both layer forwards
        end at the attention half, and joining the halves here leaves both of their
        signatures unchanged -- callers that pass the attention carriers alone keep
        working. The cost is that the two layer classes stay attention-half-only.

        The MoE branch takes the activations twice: this fork's router fuses the
        feed-forward RMSNorm inside the kernel, so it consumes the pre-norm tensor
        together with the norm's gain, while both expert halves consume the
        normalised one. The gain the fused router needs is this layer's
        ``post_attention_layernorm_weight``, and this method is where that layer's
        own gain is in scope.

        The branch is on the MLP class, not on the attention family. ``_build_mlp``
        is the single authority for which layers carry experts (dense below
        ``first_k_dense_replace``, sparse at and above it), and the two families are
        orthogonal to it -- a linear-attention layer can hold either MLP. An
        unrecognised third type refuses by name rather than falling through to one
        of the two.

        Returns:
            ``[T, H]`` in ``hidden_states``' dtype. Both routes return their own
            kernels' dtype and this method casts, because both callees put that
            choice on their caller.
        """
        gain = layer.post_attention_layernorm_weight
        if gain is None:
            raise ValueError(
                f"layer {getattr(layer, 'layer_idx', '?')} has no "
                f"post_attention_layernorm_weight; the FFN norm's gain is a "
                f"mapped checkpoint tensor and nothing was loaded onto it"
            )
        normed = self._rms_norm(hidden_states, gain)
        if collector is not None:
            collector.append(normed)
        mlp = layer.mlp
        if isinstance(mlp, Glm5NextMoEBlock):
            out = mlp(
                hidden_states,
                normed,
                **({"collector": collector} if collector is not None else {}),
                router_gamma=gain,
                text_config=self.text_config,
                quant_config=quant_config,
                block_size=block_size,
                moe_group=moe_group,
                tp_degree=tp_degree,
                expert_parallel_rank=expert_parallel_rank,
            )
        elif isinstance(mlp, Glm5NextDenseMLP):
            out = mlp(normed, quant_config=quant_config)
        else:
            raise ValueError(
                f"layer {getattr(layer, 'layer_idx', '?')} holds an MLP of type "
                f"{type(mlp).__name__}; _build_mlp builds "
                f"{Glm5NextDenseMLP.__name__} or {Glm5NextMoEBlock.__name__} and "
                f"this forward has no route for anything else"
            )

        # The one row-parallel reduction at the feed-forward site: the routed bank's,
        # the shared expert's and the dense MLP's partial sums combine in one call.
        #
        # All three meet here because every feed-forward weight family that is
        # row-parallel is sharded on its intermediate width (``_SHARD_GEOMETRY``
        # above: ``down_proj_weight`` for the dense MLP, the shared expert and the
        # routed bank), so on every route each rank returns a partial sum at the full
        # output width. Both branches above return through the same ``out``, and the
        # sparse branch's value is already routed-plus-shared, so one reduction owns
        # all three. A rank's routed contribution covers only its own experts and only
        # its slice of their intermediate width, and the expert-parallel groups
        # partition the experts, so summing across the tensor-parallel world sums each
        # token's contributions exactly once rather than twice.
        #
        # Nothing on this path reduced before it. The file's only other collective is
        # ``project_output``'s reduction; ``moe_group`` is forwarded to the MoE branch
        # and read only by the metadata builder, and the blockwise MoE functional
        # performs no collective at all. What crosses the wire on the routed path is
        # per-expert token counts, not values.
        #
        # ``_resolve_tp_group`` returns ``None`` at world size 1, so a single-rank run
        # performs no collective and imports no vllm symbol, and ``all_reduce`` is
        # called as a statement whose return is discarded -- the form every shipped
        # row-parallel site uses.
        #
        # Before the cast, and that ordering is the load-bearing part. ``out`` is the
        # kernel's own dtype here (fp32 on the dense route), so the partial sums are
        # added at the width they were computed in; reducing after the cast below
        # would round each rank's fraction to the caller's dtype and add the rounded
        # parts instead of rounding the whole.
        #
        # In place is safe against aliasing for the same reason it is at
        # ``project_output``: both branches return a freshly allocated tensor, so
        # neither is a view of a cached weight or of the residual the caller holds.
        out = _reduce_tp_rows(out, num_requests=num_requests)
        mixed = out.to(hidden_states.dtype)
        if collector is not None:
            collector.append(mixed)
        return mixed

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        layer_carriers: Sequence[dict],
        quant_config: Glm5NextQuantConfig,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        collect_layer_streams: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The whole decoder stack: embed, expand to streams, every layer in config
        order, collapse, final norm.

        The inter-layer carrier is ``[T, hc_mult, H]`` -- the checkpoint's four
        parallel residual streams, not one. Each of the three steps is the reference
        model's own: the embedding is expanded across the stream axis, every stream a
        view of the same token vector; every layer is handed the streams and hands
        streams back, with the attention half's mHC site inside the layer and the
        feed-forward half's site here, around ``_ffn_half``'s return, because that
        call is this class's; and the streams are collapsed by an unweighted mean
        before the final norm, the one collapse in this model that carries no learned
        weight at all, which is why that line is a ``mean`` and not another mHC site.

        The stack passes streams unconditionally. The branch in :func:`_mhc_site`
        refuses a streams call on a layer that carries no mHC weight and refuses a
        no-streams call on a layer that carries them, so a conditional here would be
        the one place a silent one-stream stack could come back. A caller whose
        layers hold no mHC weights is refused by name instead of served a different
        network. The draft head is unaffected: it calls the layer class directly.

        The stack is family-blind, which is why this signature takes carriers as a
        sequence of mappings rather than named cache arguments. The two families need
        different state -- the linear-attention layers take
        ``conv_state``/``recurrent_state``/``is_prefill``, the sparse-attention
        layers take ``latent_cache``/``pool_cache``/``seq_lens`` and the rest -- and
        each layer is handed its own mapping with ``**``, so this method holds no
        per-family branch at all and a new family costs it nothing.

        The softmax scale is the caller's: each sparse-attention layer's
        ``softmax_scale`` rides in that layer's own carrier mapping. Putting the
        constant in this loop would give the tree two authorities for one value.

        Args:
            input_ids: ``[T]`` integer token ids. Embedded by indexing
                ``embed_tokens_weight``; this tree holds no ``nn.Embedding`` module,
                so the lookup is the index.
            layer_carriers: one mapping per layer, in stack order, each holding that
                layer's own forward keywords. A length that disagrees with the stack
                refuses by name.
            quant_config: the resolved quantisation policy, threaded down to each
                MLP. An argument rather than a field, the convention every compute
                method in this file follows; the root resolves it once.
            block_size: tokens per block, forwarded to the expert bank unread.
            moe_group: the MoE ``GroupCoordinator``, forwarded unread.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select, as an int64
                device tensor (one graph for every rank) or a python int. The caller
                supplies both, so no degree is fixed at this site.
            collect_layer_streams: hand the per-layer carrier back beside the hidden
                state. Default False, which is the production route and leaves this
                method returning exactly the tensor it always returned. The caller
                that sets it writes the tensors to disk after the forward returns:
                this method performs no I/O, holds no file path and reads no
                environment, because a save call inside a traced forward is a graph
                break rather than a dump.

        Returns:
            ``[T, H]`` after the final RMSNorm, in the embedding table's dtype -- or,
            under ``collect_layer_streams``, that tensor and the dump's tensors: one
            ``[T, hc_mult, H]`` carrier for each of the first ``DUMP_STREAM_LAYERS``
            layers, each the tensor the next layer was handed, then the tapped
            layer's own inner tensors in the order :func:`layer_dump_names` names
            them. The streams live only inside this method and carry the table's
            dtype the whole way -- the expand copies it, both mHC sites cast their
            mixes back to it, the mean keeps it and the norm returns it -- so the
            ``.to()`` on the collapse is a no-op on the production path and a guard
            on any fixture that feeds the stack something else. Logits are the
            root's, which is where ``lm_head_weight`` lives.

        Raises:
            ValueError: when the carrier count disagrees with the layer count, when
                a mapped parameter this forward reads was never loaded, or when the
                checkpoint's ``hc_mult`` is not a positive stream count.
            Glm5NextHyperConnectionError: from :func:`_mhc_site` when a layer's mHC
                weights disagree with the streams route this method takes.
        """
        layers = list(self.layers)
        carriers = list(layer_carriers)
        if len(carriers) != len(layers):
            raise ValueError(
                f"Glm5NextModel.forward received {len(carriers)} per-layer carrier "
                f"mappings for {len(layers)} layers; one mapping per layer is "
                f"required, in stack order. A short sequence would silently run a "
                f"prefix of the stack"
            )
        table = self.embed_tokens_weight
        if table is None:
            raise ValueError(
                "Glm5NextModel.forward has no embed_tokens_weight; the embedding "
                "table is a mapped checkpoint tensor and nothing was loaded onto it"
            )
        embedded = table[input_ids]
        hc_mult = int(self.text_config.hc_mult)
        if hc_mult <= 0:
            raise ValueError(
                f"Glm5NextModel.forward cannot build a stream carrier from "
                f"hc_mult={hc_mult}; the checkpoint's config declares how many "
                f"parallel residual streams this model carries and the count has to "
                f"be positive"
            )
        # The expand. Every stream starts as the same token vector; ``contiguous``
        # because the streams are written independently from here on and a view would
        # alias them.
        streams = embedded.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
        # The collection is a list of the loop's own rebindings, appended after each
        # layer's feed-forward site, so the entry for layer i is the tensor layer i+1
        # is handed. Both lists stay empty on the production route, so the loop below
        # runs the same operations either way and nothing but a python append is
        # added. The streams stop at ``DUMP_STREAM_LAYERS``; which layer the taps come
        # from is read off the stack rather than from a constant, so a three-layer
        # stack taps the layer it actually has.
        collected: list[torch.Tensor] = []
        taps: list[torch.Tensor] = []
        tap_layers = dump_tap_layers(layers) if collect_layer_streams else ()
        for index, (layer, carrier) in enumerate(zip(layers, carriers)):
            # One tensor, passed as both arguments on purpose. The streams are this
            # layer's input; the keyword is this tree's route selector and the
            # positional is the one-stream route's operand, which a bound layer
            # refuses to take.
            streams = layer(
                streams,
                **carrier,
                streams=streams,
                **({"collector": taps} if index in tap_layers else {}),
            )
            # The feed-forward site. ``_ffn_half`` is called, never edited: the site
            # collapses the streams, hands it the single ``[T, H]`` stream it has
            # always taken, and mixes its unchanged return back. ``streams`` is never
            # ``None`` here, so this is a site or a refusal, never the plain add.
            site = _mhc_ffn_site(layer, streams)
            request_states = carrier.get("conv_state", carrier.get("pool_cache"))
            num_requests = (
                len(request_states)
                if isinstance(request_states, tuple)
                and not carrier.get("is_prefill", False)
                else 1
            )
            streams = site.forward(
                streams,
                lambda single_stream, layer=layer, num_requests=num_requests, collector=(
                    taps if index in tap_layers else None
                ): self._ffn_half(
                    layer,
                    single_stream,
                    quant_config=quant_config,
                    block_size=block_size,
                    moe_group=moe_group,
                    tp_degree=tp_degree,
                    expert_parallel_rank=expert_parallel_rank,
                    collector=collector,
                    **({"num_requests": num_requests} if num_requests > 1 else {}),
                ),
                **({"num_requests": num_requests} if num_requests > 1 else {}),
            )
            if collect_layer_streams and index < DUMP_STREAM_LAYERS:
                collected.append(streams)
        gain = self.norm_weight
        if gain is None:
            raise ValueError(
                "Glm5NextModel.forward has no norm_weight; the final norm's gain "
                "is a mapped checkpoint tensor and nothing was loaded onto it"
            )
        # The collapse: an unweighted mean over the stream axis, then the norm, in
        # that order, which is the order the reference composes them.
        collapsed = streams.mean(dim=1).to(embedded.dtype)
        normed = self._rms_norm(collapsed, gain)
        if collect_layer_streams:
            return normed, (*collected, *taps)
        return normed


# ---------------------------------------------------------------------------
# Weight loading support.
# ---------------------------------------------------------------------------

#: The dtype a quantised weight's checkpoint bytes are stored in. Ocp
#: ``float8_e4m3fn``. Declared here rather than imported, which is the fork's own
#: convention for this constant: three shipped model modules each declare their own,
#: and the sibling loader's copy of it is a private name.
_FP8_DTYPE = torch.float8_e4m3fn

#: The leaf suffix every weight parameter in this tree carries, so a scale grid can
#: be named from its weight. The weight map's own parameter names end in it.
_WEIGHT_LEAF_SUFFIX = "_weight"


def _scale_prep_leaves(module: nn.Module) -> list[str]:
    """The declared weight leaves whose scale grid is actually on the module.

    Presence and not declaration, because presence is what the caller does next:
    for every leaf this returns the caller takes a bare ``getattr`` of the sibling
    grid with no default, so a leaf kept on the strength of a declaration would
    raise ``AttributeError`` the moment that grid had not arrived. The routed expert
    bank is where the two answers part -- it declares ``router_weight``, and no
    ``router_weight_scale_inv`` exists anywhere in this tree.

    A declaration test would be wrong in the other direction too. Scale grids in
    this tree are deliberately plain attributes rather than registered parameters,
    for the reason recorded on
    :meth:`Glm5NextForConditionalGeneration._load_out_of_band_scales`, so no module
    in this file declares one and a membership test against ``declared_param_names``
    would return nothing at all.

    The name helper is reached through the class because that ``@staticmethod`` is
    its single definition; a second copy of the naming rule here is the drift this
    file's one-classifier convention exists to prevent.
    """
    return [
        leaf
        for leaf in getattr(module, "declared_param_names", ())
        if leaf.endswith(_WEIGHT_LEAF_SUFFIX)
        and hasattr(
            module,
            Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf),
        )
    ]


def _publish_compute_frame_operands(
    module: nn.Module,
    error_cls: type[ValueError],
    health_attr: str,
) -> int:
    """Put one module's dense-route weights in the frame the kernel multiplies in.

    One definition for the dense MLP and the shared expert, which load the same
    three projections onto the same kernel. Returns how many projections were
    published at the granularity the dense kernel indexes. The transpose count and
    both frames go into the health record on ``health_attr``.

    Three steps, independent, in this order.

    1. Compensate the checkpoint's scale grid by the compensation factor, once only.
       The loader squeezes the weight bytes into the 240 range and attaches these
       grids raw, so without this step the product reaching the kernel is the squeeze
       factor times the checkpoint's. The factor is read from
       ``compensate_block_scales``, never written here, and the step is never
       conditional on extents -- the compensator's own platform gate decides whether
       the multiply happens, and on a platform that needs no squeeze it is a no-op
       that still reports.
    2. Publish the checkpoint's own ``128``-tile scale grid as the grid the kernel
       indexes, unchanged. This runs for extents that are whole ``128`` blocks; a
       weight whose extents are not is recorded and skipped, because a silent skip
       looks exactly like a working publish. No weight byte is rescaled here.
    3. Transpose the weight and its grid into the compute frame. This is never
       skipped: a skipped transpose leaves the forward refusing at layer 0.

    Step 1 runs before step 2 so that the grid the module carries is the compensated
    one. The floored-block count is recorded per leaf because the ``minval`` floor is
    step 1's and breaks the uniformity of the multiply.

    The transpose is here and not in the forward. The loader delivers the
    checkpoint's own layout -- ``[I, H]`` for gate and up, ``[H, I]`` for down, since
    the shard table shards gate and up on the intermediate width -- while
    ``blockwise_fp8_mm`` reads its weight as ``[K, N]`` with ``K`` the contraction
    extent, so somebody must transpose. This package's rule is that every consumer
    transposes at compute time, and this step deliberately does not follow it:
    :meth:`Glm5NextSharedExperts.prepare_scale_operands` builds the kernel scale
    operand once at load, from the stored weight's own extents, so a forward that
    transposed would multiply a transposed weight against an operand built from the
    other frame -- the shapes agree, the numbers are wrong, and no check in this file
    would see it. Transposing before that prep reads the module keeps one frame
    authority for the whole load path, and costs one copy per projection per load
    rather than one per token on the served path.

    The transpose runs after the publish, and there is no arithmetic between them:
    the publish attaches the grid the loader delivered and the transpose relabels two
    axes. A ``128`` block of the transposed weight is the transpose of the matching
    block of the original, so the scale that block carries is the same number either
    way, which is why the frames agree.

    Args:
        module: the loaded module, with all three weights and their sibling grids
            attached. Its ``declared_param_names`` decides which leaves are visited,
            through :func:`_scale_prep_leaves`.
        error_cls: the caller's own route error, so a refusal names the class the
            reader is looking at rather than a shared one.
        health_attr: where the per-projection record is written.

    Raises:
        error_cls: if a weight or a grid is not 2-D, or if a grid is not at the
            checkpoint's own ``128``-tile granularity. A grid at any other
            granularity is refused rather than reshaped: it was built for a different
            consumer, and no shape check downstream would object.
    """
    # ``SCALE_BLOCK_SIZE`` is the dense kernel's own declaration of the grid it
    # indexes; ``TILE_SIZE`` is the checkpoint's tiling. They are equal today and are
    # still read from their own modules, so the day one moves this step follows the
    # one that moved.
    from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE
    from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

    health: dict[str, dict[str, object]] = {}
    published = 0
    for leaf in _scale_prep_leaves(module):
        grid_name = Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)
        weight = getattr(module, leaf)
        grid = getattr(module, grid_name)
        if weight.dim() != 2:
            raise error_cls(
                f"{leaf} must be 2-D to give the retile its extents, got shape "
                f"{tuple(weight.shape)}"
            )
        if grid.dim() != 2:
            raise error_cls(f"{grid_name} must be 2-D, got shape {tuple(grid.shape)}")
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        record: dict[str, object] = {"loader_frame": (rows, cols)}

        # The scale compensation, exactly once per grid. The weight bytes arrive
        # already squeezed into the 240 range by the loader while this file attaches
        # their scale grids raw. Only half a matched pair ran, so the product the
        # kernel multiplied was the squeeze factor times the checkpoint's -- an exact
        # 50% at the current factor. The loader's own module header states the pair:
        # squeeze the bytes and compensate the per-block scale by the inverse factor.
        #
        # Here, and not beside the publish, for three reasons. Both grid routes have
        # converged by this line, so the unsharded grid is covered as well as the
        # sharded one. It runs once per projection per load rather than once per
        # token. And it sits ahead of the extent branch below: that branch skips the
        # publish but still transposes whatever grid is attached, so a compensation
        # placed in the publish arm would miss exactly the extents no publish covers.
        #
        # The function is reused, never rewritten. It carries the platform gate, the
        # ``minval`` floor and the floored-block census, and the routed bank already
        # calls that same function in its own loader. The bank never reaches this prep
        # -- it retiles inside ``Glm5NextRoutedExperts.prepare_scale_operands`` -- so
        # this call cannot double-compensate it, and no second copy of the arithmetic
        # exists to drift.
        compensation = compensate_block_scales(grid)
        report_floored_blocks(compensation, grid_name)
        grid = compensation.scale_inv
        # Written back before the branch, not after it. The publish arm rebinds this
        # attribute to the public grid further down, but the skip arm never rebinds it
        # and the transpose takes whatever is attached. Without this ``setattr`` a
        # skipped projection would carry the raw grid into the compute frame, at the
        # one granularity the publish does not touch.
        setattr(module, grid_name, grid)
        # A counter, not a literal. ``record`` is fresh per leaf, so this reads 1 for
        # a grid compensated once and would read 2 if a second call were ever added to
        # this loop body. ``scale_compensated`` is the platform answer and is False
        # where the squeeze is a no-op, so a no-op platform cannot be mistaken for a
        # working compensation.
        record["scale_compensations_applied"] = (
            int(record.get("scale_compensations_applied", 0)) + 1
        )
        record["scale_compensated"] = compensation.applied
        record["scale_blocks_floored"] = len(compensation.floored_blocks)

        if rows % SCALE_BLOCK_SIZE or cols % SCALE_BLOCK_SIZE:
            # No grid the kernel can index exists for these extents. Recorded, not
            # silent. The transpose below still runs: the frame is wrong for the
            # kernel whatever the granularity is.
            record.update(
                {
                    "retiled": False,
                    "published": False,
                    "reason": (
                        f"[{rows},{cols}] is not a whole number of "
                        f"{SCALE_BLOCK_SIZE}x{SCALE_BLOCK_SIZE} blocks"
                    ),
                }
            )
        else:
            checkpoint_grid = (rows // TILE_SIZE, cols // TILE_SIZE)
            if tuple(grid.shape) != checkpoint_grid:
                raise error_cls(
                    f"{grid_name} has shape {tuple(grid.shape)}; this step "
                    f"publishes the checkpoint's own {TILE_SIZE}-tile grid "
                    f"{checkpoint_grid} for a [{rows},{cols}] weight, which since "
                    f"the block narrowed is the grid the dense kernel indexes. A "
                    f"grid at any other granularity is refused rather than "
                    f"reshaped: it was built for a different consumer."
                )
            # No retile. This arm used to coarsen the checkpoint's 128 grid onto a
            # 256 public grid and requantise the weight bytes against the retained
            # scale, because the dense kernel indexed its scales in 256-wide blocks.
            # The kernel now indexes the 128-wide blocks the checkpoint stores, so the
            # pair the checkpoint delivered is the pair the kernel consumes and there
            # is nothing to rescale.
            #
            # The MoE bank is not this path and it does not coarsen either: it
            # prepares its own operands in
            # ``Glm5NextRoutedExperts.prepare_scale_operands``, against kernels that
            # index the same 128 tiles.
            record.update(
                {
                    "retiled": False,
                    "published": True,
                    "reason": (
                        f"the dense kernel now indexes the "
                        f"checkpoint's own {TILE_SIZE}-tile grid, so this "
                        f"projection is published as loaded and no retile runs"
                    ),
                    "checkpoint_grid": checkpoint_grid,
                    "public_grid": checkpoint_grid,
                    # Zero by absence, not by measurement. No producer ran on
                    # this path, so nothing was emitted unsupplied, nothing was
                    # dropped and nothing was rescaled inexactly. ``published``
                    # above is what says why they are zero.
                    "emitted_unsupplied": 0,
                    "input_scales_dropped": 0,
                    "inexact_rescales": 0,
                }
            )
            published += 1

        # The transpose, unconditional. A dense buffer and not a bare view: the
        # kernel reads its weight as a dense buffer, and a transposed view's strides
        # are not that buffer. Where the copy is taken is the helper's subject; both
        # operands are on the device by this line, which is what the caller's
        # pre-flight established.
        weight.data = _transposed_on_the_host(weight.data, 0, 1)
        transposed_grid = _transposed_on_the_host(getattr(module, grid_name), 0, 1)
        setattr(module, grid_name, transposed_grid)
        record["transposed"] = True
        record["compute_frame"] = tuple(weight.data.shape)
        record["compute_grid"] = tuple(transposed_grid.shape)
        health[leaf] = record
    setattr(module, health_attr, health)
    # The count is how many projections left this step with a grid the dense kernel
    # can index. Every whole-256 extent is a whole-128 extent, so nothing that counted
    # under the old 256 grid stops counting; what is new is that an extent which is a
    # whole 128-wide block but not a whole 256-wide one -- 384, for one -- now counts
    # instead of being skipped.
    return published


# ---------------------------------------------------------------------------
# The mHC bind: one rule, called by both decoder layer families, run once per
# layer on the load path.
# ---------------------------------------------------------------------------

#: Where a layer keeps its two bound mHC instances, and where it keeps the record of
#: what was bound. Both are plain attributes holding a dict, deliberately outside
#: ``_modules`` and ``_parameters``, on the precedent
#: :func:`_publish_compute_frame_operands` sets for its own health dict.
#:
#: Registering the two instances instead would put their three parameters each into
#: ``named_parameters()``, and callers count that list exactly: it is ``0`` on a
#: declared-but-unloaded tree and the declared count after materialisation.
MHC_SITES_ATTR = "_mhc_sites"
MHC_BIND_HEALTH_ATTR = "_mhc_bind_health"

#: Which parameter of :class:`Glm5NextHyperConnection` each leaf role fills. The leaf
#: spellings are the checkpoint's own (``hc_attn_fn``) and the three parameter names
#: are the reference model's own, so the two differ by a prefix rather than by
#: meaning. Three entries and no default, so a fourth role cannot be bound by
#: accident.
MHC_ROLE_PARAMETERS: dict[str, str] = {
    "fn": "fn",
    "base": "hc_base",
    "scale": "hc_scale",
}

#: The two sites, named by the middle word of their own leaves rather than chosen:
#: ``hc_attn_*`` is the site the attention half runs and ``hc_ffn_*`` the site the
#: feed-forward half runs, which is the order the reference model composes them in.
#: :func:`_mhc_leaves_by_site` derives the same two keys from ``MHC_LEAVES``.
MHC_ATTENTION_SITE = "attn"
MHC_FFN_SITE = "ffn"


def _mhc_leaves_by_site(
    leaves: Sequence[str] = MHC_LEAVES,
) -> dict[str, dict[str, str]]:
    """The map's mHC leaves grouped as ``{site: {role: leaf}}``.

    Derived from the weight map's own tuple, never retyped here. The six names live
    once, in ``weight_loaders_fp8.MHC_LEAVES``, and the grouping is read off the name
    shape ``hc_<site>_<role>``, which is also how the map emits them.

    A leaf that is not spelled that way, or whose role is not one of the three the
    class takes, raises rather than being skipped. A silently dropped leaf would
    leave one site holding the zeros its constructor allocated, the forward would
    compute a plausible number from them, and no check in this file would object.

    Raises:
        Glm5NextHyperConnectionError: if a leaf cannot be resolved to a site and a
            role.
    """
    grouped: dict[str, dict[str, str]] = {}
    for leaf in leaves:
        parts = leaf.split("_")
        if len(parts) != 3 or parts[0] != "hc" or parts[2] not in MHC_ROLE_PARAMETERS:
            raise Glm5NextHyperConnectionError(
                f"the mHC leaf {leaf!r} is not spelled hc_<site>_<role> with a "
                f"role in {sorted(MHC_ROLE_PARAMETERS)}, so this bind cannot say "
                f"which site it belongs to or which parameter it fills. The weight "
                f"map and this file disagree about the family's names"
            )
        grouped.setdefault(parts[1], {})[parts[2]] = leaf
    return grouped


def _mhc_site(
    module: nn.Module, streams: torch.Tensor | None, site: str
) -> Glm5NextHyperConnection | None:
    """Which route one call takes, refusing both ways, for either site.

    Returns the named mHC site of ``module`` when the call carries streams, and
    ``None`` when the plain residual add is the right thing to do.

    One rule, two sites. The attention half asks for ``MHC_ATTENTION_SITE`` from
    inside each layer forward; the feed-forward half asks for ``MHC_FFN_SITE`` from
    the carrier, which is where ``_ffn_half`` is called. Both ask the same question
    -- does this module's weights agree with the route this call takes -- so both get
    the same three answers from this one body rather than from two copies that can
    drift. :func:`_mhc_attention_site` and :func:`_mhc_ffn_site` are the two names,
    and they add nothing but the site.

    The keyword is optional because two callers need it absent -- the draft head
    hands this same sparse-attention class a one-stream ``[T, H]`` and the checkpoint
    gives its layer none of the six mHC leaves -- and optional-with-a-default is
    exactly how a silent wrong default gets in. So the branch refuses both ways and
    one rule decides for both layer families:

    * **no streams on a layer that carries mHC weights** is refused. Serving that
      call would quietly reinstate the one-stream network, and nothing downstream
      could tell.
    * **streams on a layer that has no site to run them** is refused, and the message
      says which of the two reasons it is: the layer carries none of the six leaves
      (the draft head's case), or it carries them and the load-time bind never ran,
      which is a caller that skipped ``_run_load_time_preps``.

    The carry test reads the leaves, not the bind. Reading the bind instead would let
    a loaded-but-unbound layer take the plain add silently, which is the same defect
    wearing a different hat. The read is a short-circuiting ``next`` over six names,
    so a layer that carries them stops at the first.

    Raises:
        Glm5NextHyperConnectionError: either way round, naming the case.
    """
    carried = next((leaf for leaf in MHC_LEAVES if getattr(module, leaf, None) is not None), None)
    if streams is None:
        if carried is not None:
            raise Glm5NextHyperConnectionError(
                f"this layer carries the mHC weight {carried} and was called with "
                f"no streams. A one-stream call here would run the residual add "
                f"this block replaces, so it is refused rather than served: pass "
                f"the four streams, or call a layer that carries no mHC weight"
            )
        return None

    sites = getattr(module, MHC_SITES_ATTR, {})
    if not sites:
        if carried is not None:
            raise Glm5NextHyperConnectionError(
                f"this layer carries the mHC weight {carried} but no site is bound "
                f"to it, so a streams call has nothing to run. The bind happens on "
                f"the load path, in _run_load_time_preps; a caller that built the "
                f"tree by hand has to run it too"
            )
        raise Glm5NextHyperConnectionError(
            "this layer carries none of the six mHC weights and was called WITH "
            "streams. The checkpoint gives the draft head's layer none of them, so "
            "this call belongs on the one-stream path: pass no streams"
        )
    return sites[site]


def _mhc_attention_site(
    module: nn.Module, streams: torch.Tensor | None
) -> Glm5NextHyperConnection | None:
    """The attention half's site, or ``None`` for the plain add. See :func:`_mhc_site`."""
    return _mhc_site(module, streams, MHC_ATTENTION_SITE)


def _mhc_ffn_site(
    module: nn.Module, streams: torch.Tensor | None
) -> Glm5NextHyperConnection | None:
    """The feed-forward half's site, or ``None`` for the plain add.

    Asked by the carrier rather than by a layer, because ``_ffn_half`` is
    ``Glm5NextModel``'s and the site therefore composes where that call is made. See
    :func:`_mhc_site` for the three answers.
    """
    return _mhc_site(module, streams, MHC_FFN_SITE)


def _bind_hyper_connection_sites(
    module: nn.Module,
    text_config: Glm5NextTextConfig,
    device: torch.device,
) -> int:
    """Hand one layer's six loaded mHC tensors to its two mHC instances.

    Returns how many sites were bound: ``2`` on a layer whose six leaves are loaded,
    ``0`` on a layer that carries none of them.

    It runs after the load and not at construction because the six leaves are
    ``register_parameter(name, None)`` declarations until
    ``_materialise_declared_parameters`` registers placeholders for them, so an
    instance built in ``__init__`` and handed ``self.hc_attn_fn`` would be handed
    ``None``. It reaches the tree through the same single production caller as the
    other load-time preps, :meth:`_run_load_time_preps`.

    Three cases, and the middle one is the refusal.

    * All six loaded: both sites are bound.
    * None loaded: the layer is skipped and the skip is recorded. This case is real
      rather than defensive -- the weight map emits the six leaves only for the
      layers in ``layer_types``, so a draft-head block built from the same layer
      class declares all six and is loaded none of them.
    * Some loaded: raised, naming every leaf that is missing. A half-bound site would
      compute from the zeros its constructor allocated.

    Shapes are recorded, not enforced. The package's end-to-end load tests run
    against a miniature checkpoint that writes each plain key at an arbitrary small
    shape on purpose, so a shape check here would refuse every one of those loads for
    a shape the fixture never meant to be right. What each site received goes into the
    record instead, and the shape that matters is checked where it is used, in the
    forward, against the reference pair.

    ``.data`` assignment, not a copy. The site parameter keeps its own
    ``nn.Parameter`` object and takes the loaded tensor's storage, so the bind copies
    no weight bytes.

    One exposure, disclosed: a plain dict is not visited by ``nn.Module._apply``, so
    a ``.to(device)`` issued after this bind would move the layer's own parameter and
    leave the site pointing at the old storage. This is the exposure the other
    load-time prep operands already carry, and the answer here is the same -- the bind
    runs after the weights are on the device, it refuses an operand that is somewhere
    else, and each leaf's ``data_ptr`` goes into the record so the two pointers can be
    compared later without a new instrument.

    The post gate's multiplier is left at the class default of ``2.0``, which is the
    reference model's own factor. Nothing here chooses a number.

    Args:
        module: the layer, with the six leaves declared and -- if the checkpoint
            carried them -- loaded.
        text_config: sizes both instances, and carries the framework overrides
            ``mhc_sinkhorn_iters`` and ``mhc_eps`` on its ``neuron_config``.
        device: where the load put the weights.

    Raises:
        Glm5NextHyperConnectionError: if some but not all of the six leaves are
            loaded, if a loaded leaf is still a shape-free placeholder, or if one is
            not on ``device``.
    """
    sites = _mhc_leaves_by_site()
    loaded: dict[str, torch.Tensor] = {}
    unloaded: list[str] = []
    for roles in sites.values():
        for leaf in roles.values():
            operand = getattr(module, leaf, None)
            if operand is None:
                unloaded.append(leaf)
            else:
                loaded[leaf] = operand

    # A visit is counted whether it binds or skips, so a reader can tell one visit
    # from two. ``_run_load_time_preps`` may be called more than once on the same
    # tree, so a second visit is normal and is recorded rather than refused.
    previous = getattr(module, MHC_BIND_HEALTH_ATTR, None)
    binds = int((previous or {}).get("binds", 0)) + 1

    if not loaded:
        setattr(module, MHC_SITES_ATTR, {})
        setattr(
            module,
            MHC_BIND_HEALTH_ATTR,
            {
                "bound_sites": 0,
                "binds": binds,
                "unloaded_leaves": sorted(unloaded),
                "skipped_because": (
                    "the checkpoint carried none of the six mHC leaves for this "
                    "layer, so there is nothing to bind"
                ),
            },
        )
        return 0

    if unloaded:
        raise Glm5NextHyperConnectionError(
            f"this layer carries {len(loaded)} of the six mHC weights and is "
            f"missing {sorted(unloaded)}, so a bind would leave a site holding "
            f"the zeros its constructor allocated. Loaded: {sorted(loaded)}"
        )

    for leaf, operand in sorted(loaded.items()):
        if torch.nn.parameter.is_lazy(operand):
            raise Glm5NextHyperConnectionError(
                f"{leaf} is still a shape-free placeholder, so the load has not "
                f"filled it and binding it now would hand a site an empty tensor"
            )
        if not _is_on_device(operand.device, device):
            raise Glm5NextHyperConnectionError(
                f"{leaf} is on {operand.device} and this load targets {device}. "
                f"The bound site is held in a plain dict that no later "
                f"``.to(device)`` visits, so a bind from the wrong device would "
                f"strand this weight there permanently"
            )

    bound: dict[str, Glm5NextHyperConnection] = {}
    record: dict[str, dict[str, object]] = {}
    for site in sorted(sites):
        instance = Glm5NextHyperConnection(
            text_config, neuron_config=text_config.neuron_config
        )
        # The instance allocates its three parameters with no device, so it is
        # built where the default device is whatever this load targets. Assigning
        # an operand's data onto a parameter of another type raises, so the
        # instance is moved onto the load's device first. A load on the default
        # device moves nothing and hands over the same storage as before.
        instance.to(device)
        site_record: dict[str, object] = {}
        for role, leaf in sorted(sites[site].items()):
            operand = loaded[leaf]
            parameter = getattr(instance, MHC_ROLE_PARAMETERS[role])
            parameter.data = operand.data
            site_record[leaf] = {
                "parameter": MHC_ROLE_PARAMETERS[role],
                "shape": tuple(operand.shape),
                "dtype": str(operand.dtype),
                "device": str(operand.device),
                # An operand with no storage has no address to record.
                "data_ptr": (
                    0 if operand.device.type == "meta" else int(operand.data_ptr())
                ),
            }
        bound[site] = instance
        record[site] = site_record

    setattr(module, MHC_SITES_ATTR, bound)
    setattr(
        module,
        MHC_BIND_HEALTH_ATTR,
        {
            "bound_sites": len(bound),
            "binds": binds,
            "device": str(device),
            "sinkhorn_iters": int(bound[sorted(bound)[0]].sinkhorn_iters),
            "hc_eps": float(bound[sorted(bound)[0]].hc_eps),
            "post_mult_value": float(bound[sorted(bound)[0]].post_mult_value),
            "sites": record,
        },
    )
    return len(bound)


class Glm5NextWeightLoadError(ValueError):
    """A checkpoint could not be read onto this model tree.

    A ``ValueError`` subclass to match the three named errors this file already
    raises (``Glm5NextHyperConnectionError``, ``Glm5NextBlockQuantRouteError``,
    ``Glm5NextSharedExpertRouteError``) rather than to claim the failure is a
    file-system one -- the checkpoint argument may name a local directory or a remote
    repository, and :meth:`Glm5NextForConditionalGeneration.load_weights` must refuse
    both the same way.

    Every message this class carries names the checkpoint path, because the
    argument is passed straight through from the runner and a refusal that did
    not name it would send a reader to the wrong place.
    """


class _MetaSlice:
    """One checkpoint tensor's header, answering data reads with meta tensors.

    The loaders read a slice by indexing it, and every index they use is a plain
    torch index. So the shape arithmetic of a shard or a fusion is done by
    indexing a meta tensor of the stored shape, which is torch's own answer to
    the question rather than a second implementation of it here.
    """

    def __init__(self, source: object) -> None:
        self._source = source
        self._empty: torch.Tensor | None = None

    def get_shape(self) -> list[int]:
        """The stored shape, as the header gives it."""
        return self._source.get_shape()

    def get_dtype(self) -> str:
        """The stored dtype, as the header gives it."""
        return self._source.get_dtype()

    def _meta(self) -> torch.Tensor:
        if self._empty is None:
            shape = tuple(int(extent) for extent in self._source.get_shape())
            # The dtype comes off one row, not off a name table. Safetensors names
            # its dtypes in its own spelling and torch has no reader for that
            # spelling, so a table here would be a second place every dtype in the
            # checkpoint is written down, the fp8 ones this model depends on
            # included. One row costs the trailing dimensions times the item size --
            # About 4 KB for a [1536, 4096] fp8 weight -- not a few bytes. A 0-D
            # tensor has no row, so it is taken whole.
            probe = self._source[0:1] if shape and shape[0] else self._source[:]
            self._empty = torch.empty(shape, dtype=probe.dtype, device="meta")
        return self._empty

    def __getitem__(self, index: object) -> torch.Tensor:
        return self._meta()[index]


class _MetaShapeCheckpoint(SafetensorsCheckpoint):
    """The checkpoint reader that answers with shapes and reads no weight data.

    Only the two places that touch bytes are replaced. File discovery, the key
    index and the load pipeline stay the shared reader's own, so a fusion or a
    shard that the real load performs is performed here too.
    """

    def _get_slice(self, name: str) -> _MetaSlice:
        """One tensor's slice, wrapped so its data reads as meta."""
        return _MetaSlice(super()._get_slice(name))

    def _load_to_page_cache(
        self,
        rank: int,
        world_size: int,
        cached_files_store: "torch.distributed.Store",
        shutdown_event: "threading.Event",
    ) -> None:
        """Announce this rank's files without reading their bytes.

        The announcement is not optional. The pipelined load's main loop waits for
        exactly the keys this method writes: it turns on a file only when
        ``cached_files_store.check([file_name])`` answers, and nothing else ever calls
        ``add``. A method that returned without adding them would leave that loop
        spinning on its 1 ms sleep until the process was killed, so the shape-only
        pass keeps the round-robin and the store write and drops only the read-through
        that pulls the whole checkpoint into ram.
        """
        for index, file_name in enumerate(self._source.get_file_names()):
            if shutdown_event.is_set():
                return
            if index % world_size != rank:
                continue
            self._source.download_file(file_name)
            cached_files_store.add(file_name, 1)


class Glm5NextForConditionalGeneration(nn.Module):
    """The blockwise-FP8 glm-5.3-Flash implementation.

    The module path, this class name and the ``from_configs`` signature are what
    ``factory.py`` imports and calls. The name is duplicated with the factory's
    selector on purpose -- that is the plugin's selector-to-implementation
    convention, the same pair ``llama3/factory.py`` uses -- so neither side is
    renamed here.
    """

    # KDA banks are indexed by live request slots, independently of token pages.
    request_indexed_kda_state = True

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        if config.text_config is None:
            raise ValueError(
                "Glm5NextConfig.text_config is required to build the decoder "
                "stack; got None"
            )
        self.config = config
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        self.world_size = _resolve_world_size()

        self.model = Glm5NextModel(config, self.world_size)
        # Mirrors the weight map's own condition: a tied head has no separate
        # checkpoint key and therefore no separate parameter.
        if not self.text_config.tie_word_embeddings:
            _declare_parameters(self, "lm_head_weight")

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> Glm5NextForConditionalGeneration:
        """Build from an HF config, the signature the factory calls."""
        config = Glm5NextConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    # ── declared parameter names ─────────────────────────────────────────

    def declared_parameter_names(self) -> tuple[str, ...]:
        """Every parameter attribute path this tree declares, in tree order.

        ``named_parameters()`` cannot be used, because declared-but-unmaterialised
        parameters are ``None`` and torch skips ``None`` entries in both
        ``named_parameters()`` and ``state_dict()``.
        """
        names: list[str] = []
        for module_path, module in self.named_modules():
            for leaf in getattr(module, "declared_param_names", ()):
                names.append(f"{module_path}.{leaf}" if module_path else leaf)
        return tuple(names)

    def released_parameters(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Every checkpoint parameter released after its load-time prep, by tree path."""
        released: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for module_path, module in self.named_modules():
            for leaf, record in getattr(module, RELEASED_PARAMETERS_ATTR, {}).items():
                released[f"{module_path}.{leaf}" if module_path else leaf] = record
        return released

    # ── weight loading ──────────────────────────────

    def _checkpoint_block_size(self) -> tuple[int, int]:
        """The quantisation block shape this checkpoint declares, as two ints.

        Read off ``quantization_config.weight_block_size``, which the config adapter
        lifts verbatim, so the rule that converts a weight shard into grid rows is
        told the checkpoint's own tile instead of taking the parser's fallback. The
        fallback is right for this checkpoint and would be silently wrong for the next
        one, which is why this is threaded rather than defaulted.

        Refused rather than defaulted when the field is absent or malformed. The only
        caller reaches this after finding a scale grid in the checkpoint, so a
        checkpoint that ships scale grids and declares no block shape is a
        contradiction, and guessing 128 there would scale real weight blocks by the
        wrong grid row.
        """
        declared = self.config.weight_block_size
        if not declared or len(declared) != 2:
            raise Glm5NextWeightLoadError(
                f"this checkpoint ships block-FP8 scale grids but declares "
                f"quantization_config.weight_block_size={declared!r}; a sharded "
                f"grid cannot be converted from weight rows to grid rows without "
                f"the block shape, and defaulting it would scale weight blocks by "
                f"the wrong grid row"
            )
        return (int(declared[0]), int(declared[1]))

    def _placeholder_dtype(
        self,
        checkpoint_keys: str | list[str],
        *,
        param_name: str,
        mappings: dict,
    ) -> torch.dtype:
        """The dtype the loader will hand back for one mapped parameter.

        Load-bearing rather than cosmetic. The checkpoint reader takes its target
        dtype off the placeholder and casts the tensor it built to that dtype after a
        warning, so a placeholder typed one step too wide does not merely log -- it
        changes the values that reach the model.

        The four cases come from
        :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.classify_mapped_keys`,
        which is the one classifier this method and the loader choice share, so the
        dtype and the loader cannot disagree about what a key is. This is the second
        consumer of that classifier, and it moves whenever the classifier gains a
        kind.

        A stacked expert bank is fp8 for the same reason a lone quantised weight is,
        and it is named in the arm below rather than left to fall through. A kind this
        method did not name would fall past the ``quantised_weight`` arm, miss the
        sibling clause -- a bank has no sibling scale entry, because its scales are
        interleaved inside its own -- and reach ``text_config.torch_dtype``. Every
        bank placeholder on the real configuration would then be bf16 while its loader
        delivers fp8, which is exactly the mismatch the reader warns about.

        The sibling clause, and the defect it repairs. A quantised weight whose scale
        travels in the same map entry classifies ``quantised_weight`` and takes fp8
        from the case above. But each of the four scaled MLA projections has its own
        scale-grid entry, which leaves each of those weights alone in its entry -- and
        a lone weight key classifies ``plain``. Without the clause below they took the
        config dtype, the reader narrowed the checkpoint's fp8 bytes to bf16, and
        :meth:`Glm5NextMLAAttention._dequantised_projection_weight` then saw a real
        dtype and returned the weight unchanged. The dequant did nothing, silently, on
        every scaled projection of every sparse-attention layer, and the transpose
        that follows it operated on narrowed bytes -- exactly the defect the dequant
        exists to repair, arriving through the placeholder dtype instead.

        The clause asks the map whether a sibling scale grid exists for this weight,
        which is the same question ``_dequantised_projection_weight`` asks of the
        module -- one question, asked of the two places that have to agree. It adds no
        second classifier of the three cases: ``classify_mapped_keys`` still decides,
        and this only distinguishes the two kinds of ``plain``.

        Two groups are typed by name rather than by kind. The linear-attention decay
        and gate bias arrive as plain keys, so the kinds above would hand them the
        config dtype -- and the checkpoint holds them in float32. The reference keeps
        that width all the way to the gate; a narrowing placeholder spends it before
        the gate's exponential and sigmoid ever read them. The second group is
        ``FLOAT32_PLAIN_LEAVES``: the four mHC mix leaves and the router correction
        bias, which the checkpoint also holds in float32.
        """
        kind = classify_mapped_keys(checkpoint_keys)
        if kind == MAPPED_KEY_SCALE_GRID:
            return torch.float32
        if kind in (MAPPED_KEY_QUANTISED_WEIGHT, MAPPED_KEY_STACKED_BANK):
            return _FP8_DTYPE
        if param_name.rsplit(".", 1)[-1] in KDA_BARE_LEAVES:
            # The GPU reference declares both in float32 and its kernels then read
            # them as float32 as well.
            return torch.float32
        if param_name.rsplit(".", 1)[-1] in FLOAT32_PLAIN_LEAVES:
            # The mHC mix state and the router correction bias. Both halves of
            # ``Glm5NextHyperConnection`` are fp32 in and fp32 out, and the router
            # widens the bias to float32 before it corrects the scores, so a
            # narrowing placeholder spends the width before either one reads it.
            return torch.float32
        if self._sibling_scale_grid_name(param_name) in mappings:
            return _FP8_DTYPE
        return self.text_config.torch_dtype

    @staticmethod
    def _sibling_scale_grid_name(param_name: str) -> str:
        """The scale-grid parameter that would accompany this weight, by name.

        Returns a name that is deliberately unmatchable for a parameter that is
        not a weight, rather than ``None``, so every caller is a membership test
        and none has to branch on the shape of the answer.
        """
        if not param_name.endswith(_WEIGHT_LEAF_SUFFIX):
            return ""
        base = param_name[: -len(_WEIGHT_LEAF_SUFFIX)]
        return f"{base}_{FP8_SCALE_SUFFIX}"

    def _materialise_declared_parameters(
        self, mappings: dict, device: torch.device
    ) -> int:
        """Give every declared parameter a shape-free placeholder and its loader.

        Returns how many were materialised.

        Why this step exists at all. The checkpoint reader decides what to load by
        iterating ``list(model.named_parameters())``, and torch omits a
        ``register_parameter(name, None)`` declaration from that list. This tree is
        every parameter declared and none materialised, so before this step the list
        is empty and a load that skipped it would read no weight at all and report
        success.

        Why the placeholder claims no shape. A shaped placeholder would have to carry
        the per-rank sharded shape, which the modules in this file already know and
        which would become a second place to get it wrong. It does not have to:
        ``UninitializedParameter`` is torch's own parameter-without-a-shape-yet, and
        torch's shape check when the tensors arrive is guarded by exactly that case --
        ``if not is_param_lazy and input_param.shape != param.shape``, with no
        ``assign`` term in it. An ordinary zero-size placeholder is refused there; the
        lazy one is accepted and filled from the checkpoint. ``assign=False`` raises
        on a lazy parameter, which is why the load below passes ``assign=True``.

        The placeholder still carries the only two things the reader takes off it: the
        dtype, and the weight loader attached to it.

        The walk is the same one :meth:`declared_parameter_names` does, and is
        repeated here only because registering a parameter needs the module and the
        leaf name rather than the dotted path.

        Why it is two passes and not one. Choosing a loader can refuse -- a malformed
        expert bank entry, for one -- and a single pass that registered as it went
        would leave every parameter before the refusing one registered and every one
        after it declared, which is a tree no later load can tell apart from a fresh
        one. So nothing is registered until every placeholder and every loader has
        been built: a refusal leaves the tree exactly as it arrived, and
        ``named_parameters()`` still reads zero.
        """
        planned: list[tuple[nn.Module, str, nn.Parameter]] = []
        for module_path, module in self.named_modules():
            for leaf in getattr(module, "declared_param_names", ()):
                name = f"{module_path}.{leaf}" if module_path else leaf
                checkpoint_keys = mappings.get(name, name)
                placeholder = nn.parameter.UninitializedParameter(
                    dtype=self._placeholder_dtype(
                        checkpoint_keys, param_name=name, mappings=mappings
                    ),
                    device=device,
                    requires_grad=False,
                )
                # May raise. Deliberately before the first registration below.
                #
                # ``owner`` is the module that declared this parameter, and only the
                # expert-bank loader reads it: the expert geometry lives on
                # ``Glm5NextRoutedExperts`` and nowhere else, so the loader is handed
                # the declaring module rather than deriving a second partition from
                # the key count.
                #
                # ``geometry`` is this site's to resolve, because this is the only
                # place that holds both halves: the declaring module, which knows its
                # own widths, and ``self.world_size``, which only two declaring
                # classes are told. It is ``None`` for every replicated family and at
                # world size 1.
                loader = loader_for_mapped_keys(
                    checkpoint_keys,
                    param_name=name,
                    owner=module,
                    geometry=_shard_geometry_for(module, leaf, self.world_size),
                )
                if loader is not None:
                    set_weight_loader(placeholder, loader)
                planned.append((module, leaf, placeholder))

        for module, leaf, placeholder in planned:
            module.register_parameter(leaf, placeholder)
        return len(planned)

    def load_weights_lite(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Shape the tree from the checkpoint's headers, reading no weight data.

        The runner calls this on a CPU-compile start, where no weights are wanted and
        the whole module is moved to meta afterwards. Before this method that arm
        loaded nothing at all: every parameter stayed a
        ``register_parameter(name, None)`` declaration, so the root forward refused at
        its embedding table and the mHC sites were never bound.

        It runs the real load, and that is the whole design. The shapes a parameter
        ends up with are decided by the loaders -- which checkpoint keys fuse into one
        tensor, how the shard geometry narrows it -- and those rules live in one
        place. So this method changes where the numbers come from and nothing else:
        :class:`_MetaShapeCheckpoint` answers with meta tensors of the checkpoint's
        own shapes, and every loader, prep and bind then executes the code the real
        load executes.

        ``device`` is ignored and meta is used instead. The runner passes the CPU
        because that is where it wants compile-time constants read from, but a CPU
        tensor of these shapes would allocate the whole model; meta allocates nothing
        and is where the runner moves the module two lines later anyway.
        """
        meta = torch.device("meta")
        self.to(meta)
        self.load_weights(
            checkpoint_path,
            meta,
            cache_dir,
            reader=_MetaShapeCheckpoint(checkpoint_path, cache_dir),
        )

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
        *,
        reader: object | None = None,
    ) -> None:
        """Read the checkpoint's weights onto ``device``.

        ``reader`` is the hook :meth:`load_weights_lite` replaces, and it is
        keyword-only with a default so the three-argument call the runner makes is the
        call it always was. Passed ``None``, this method opens the real checkpoint on
        the line it always opened it. The signature is the shipped one every other
        model in this fork uses, so no runner file changes.

        Four steps, in an order two of them force:

        1. Open the checkpoint. This is first so that a checkpoint that cannot be
           opened leaves the tree exactly as it was -- every parameter still declared
           and none materialised.
        2. Build the key map at the checkpoint's own quantisation setting with the
           checkpoint's own skip list. The map is what the reader is told to look for,
           so a family missing here is a family never read.
        3. Materialise a placeholder per declared parameter and attach its loader.
           Forced to come before step 4, for the reason
           :meth:`_materialise_declared_parameters` records.
        4. Read the tensors and assign them. ``assign=True`` is required, not
           stylistic: it is what replaces each placeholder with the tensor the reader
           built, and ``assign=False`` raises on a lazy placeholder.

        The rank resolved here reaches no loader that shards on it except through the
        shard table: ``blockwise_scale_loader``'s own docstring records that a sharded
        scale grid follows the weight's own shard geometry and lands with the module
        that declares that geometry. The rank is resolved and passed because the
        reader's contract takes one and hands it to every loader transform.

        What still refuses is named.
        :class:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.Glm5NextExpertBankNotLoadableError`
        comes out of step 3 for what this package cannot serve honestly: a malformed
        entry (odd key count, or a scale key out of alternation), a multi-weight entry
        carrying no scale key, and a bank whose owning module declares no expert
        geometry. It is deliberately not caught and re-wrapped here -- its message
        already names the parameter and the defect, and wrapping it in this method's
        own error would bury both behind a sentence about checkpoints. It is a
        ``ValueError`` either way, like ``Glm5NextWeightLoadError``.
        """
        try:
            checkpoint = reader or SafetensorsCheckpoint(checkpoint_path, cache_dir)
            num_files = checkpoint.get_num_files()
        except Exception as exc:
            raise Glm5NextWeightLoadError(
                f"no safetensors checkpoint could be opened at "
                f"{checkpoint_path!r} (cache_dir={cache_dir!r}); no parameter "
                f"was materialised, so the model tree is unchanged"
            ) from exc
        if num_files < 1:
            raise Glm5NextWeightLoadError(
                f"the checkpoint at {checkpoint_path!r} holds no .safetensors "
                f"file; no parameter was materialised, so the model tree is "
                f"unchanged"
            )

        # The two quantisation settings sit on the top-level config, not on the text
        # config: the config adapter lifted them from the checkpoint's top-level
        # ``quantization_config``, while ``torch_dtype`` is the text config's own.
        # Reading either off the wrong object raises ``AttributeError`` rather than
        # defaulting.
        mappings = build_weight_mappings(
            self.text_config,
            quantised=self.config.is_block_quantized,
            modules_to_not_convert=tuple(self.config.modules_to_not_convert or ()),
        )

        rank = _resolve_rank()
        if rank >= self.world_size:
            raise Glm5NextWeightLoadError(
                f"rank {rank} is not inside the world size {self.world_size} "
                f"this model was built for; the world size is read once at "
                f"construction and the rank is read here, so a process group "
                f"that appeared between the two would shard against a rank the "
                f"model does not know about"
            )

        materialised = self._materialise_declared_parameters(mappings, device)
        if materialised == 0:
            raise Glm5NextWeightLoadError(
                "no declared parameter was materialised, so the checkpoint "
                "reader would iterate an empty parameter list and load nothing "
                f"while reporting success; declared_parameter_names() reports "
                f"{len(self.declared_parameter_names())} names"
            )

        rank_sharded = checkpoint.load_sharded_pipelined(
            rank, self.world_size, self, mappings, device
        ).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Everything below runs after the line above, and that is the whole ordering
        # contract of this method: the line above is what turns a shape-free
        # placeholder into a real on-device tensor, so a scale read or a prep placed
        # before it would work on placeholders.
        self._load_out_of_band_scales(checkpoint, mappings, device)
        self._run_load_time_preps(device)

    def _load_out_of_band_scales(
        self,
        checkpoint: SafetensorsCheckpoint,
        mappings: dict,
        device: torch.device,
    ) -> int:
        """Read every scale grid the loaders drop, straight from the checkpoint.

        Returns how many were read.

        Why any scale is read out of band. A map entry holding a weight and its scale
        is served by
        :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.wrap_with_blockwise_fp8_downscale`,
        whose base transform keeps the weight slice and drops the companion, so the
        scale reaches nothing through the reader. Widening the map so each of those
        scales became its own parameter would move the difference set other tests
        assert, so the scales are read here instead, on the fork's own precedent for
        exactly this problem in ``qwen3/model.py``.

        The indexing is explicit, for the reason the precedent states in its own
        comment: ``load_sharded_pipelined`` indexes as a side effect, and a lookup on
        an unindexed checkpoint silently misses. This method is called after that
        reader has run, so the index is already built -- and it calls
        :meth:`~vllm_neuron.utils.checkpoints.SafetensorsCheckpoint._ensure_indexed`
        anyway, because a later caller that reads scales without the pipelined load
        would otherwise get zeros with no complaint.

        A miss is a refusal, not a default. The precedent falls back to ``torch.ones``
        when a key is absent, which is right for a KV scale that may legitimately not
        be published. It is wrong here: a missing block scale means the fp8 bytes of
        that projection cannot be dequantised, and a silent 1.0 would make the bytes
        look like numbers.

        Where the scale is put: as a plain attribute on the owning module, named the
        way the consumer already looks it up, ``f"{leaf}_{FP8_SCALE_SUFFIX}"``. Plain,
        not a registered parameter: registering it would add a name to
        ``named_parameters()`` that the map does not carry, which is the map widening
        this method exists to avoid.

        A routed expert bank is read here too. A bank's E scale grids are interleaved
        with its E weight keys inside one map entry, so the lone-grid path below
        cannot see them: it wants a single companion scale and finds E. The bank
        branch asks the classifier whether an entry is a bank, hands the whole key
        list to the stacking loader, and stores the result under the same
        plain-attribute name -- so the bank gains three attributes per layer while
        ``named_parameters()`` and ``build_weight_mappings`` stay where they are. No
        module in this file declares a ``prepare_scale_operands`` for a bank yet, so
        the prep loop does not visit one.

        A grid the map declares as its own parameter is not read here. The scaled
        attention projections pair the weight with its scale key so the loader chooser
        downscales the bytes, and also map the scale key to its own parameter; that
        parameter path loads and compensates the grid, and a plain attribute of the
        same name set here would shadow it.
        """
        checkpoint._ensure_indexed()
        declared_grids: set[str] = set()
        for keys in mappings.values():
            key_list = [keys] if isinstance(keys, str) else list(keys)
            if classify_mapped_keys(key_list) == MAPPED_KEY_SCALE_GRID:
                declared_grids.add(key_list[0])
        read = 0
        for param_name, keys in mappings.items():
            key_list = [keys] if isinstance(keys, str) else list(keys)
            scales = scale_keys(key_list)
            if classify_mapped_keys(key_list) == MAPPED_KEY_STACKED_BANK:
                # A bank does reach here, and it is the one entry shape the
                # lone-grid path below cannot serve: its E scale grids travel
                # interleaved with its E weight keys inside this one entry, so there
                # is no companion entry to read and no single ``scales[0]`` to read it
                # from. The classifier is asked rather than re-derived, so "what is a
                # bank" has one answer in this file and not two.
                #
                # The rank is resolved here, not taken as an argument. The loader's
                # transform needs one, and both call sites pass this method three
                # positionals; widening its signature to thread a rank through would
                # move working code for a value ``_resolve_rank`` already owns.
                module_path, _, leaf = param_name.rpartition(".")
                attribute = self._sibling_scale_grid_name(leaf)
                if not attribute:
                    raise Glm5NextWeightLoadError(
                        f"{param_name} is a bank of "
                        f"{len(scales)} expert scale grids but its leaf name "
                        f"does not end in {_WEIGHT_LEAF_SUFFIX!r}, so the "
                        f"stacked grid has no name to be stored under; the map "
                        f"and this reader disagree about what a weight "
                        f"parameter is called"
                    )
                absent = [
                    key
                    for key in key_list
                    if key not in checkpoint._tensor_name_to_file
                ]
                if absent:
                    raise Glm5NextWeightLoadError(
                        f"{param_name} is a bank of {len(scales)} experts and "
                        f"{len(absent)} of its {len(key_list)} checkpoint keys "
                        f"are absent, first {absent[0]!r}, so its fp8 bytes "
                        f"cannot be dequantised; reading a default of 1.0 here "
                        f"would use the bytes as if they were numbers"
                    )
                module = self.get_submodule(module_path) if module_path else self
                # The bank's own loader is what de-interleaves and stacks these grids
                # -- this rank's experts through ``local_expert_indices``, each grid
                # compensated by ``compensate_block_scales``, stacked on the leading
                # axis in expert order, so row ``[e]`` matches the weight bank's row
                # ``[e]``. It was written to be called rather than attached, and this
                # is that caller.
                #
                # The bank's geometry reaches this call, so a column-sharded bank gets
                # column-sharded grids. ``None`` at expert-parallel degree 1 and at
                # world size 1.
                bank_geometry = _shard_geometry_for(module, leaf, self.world_size)
                grid = stacked_expert_scale_loader(
                    key_list,
                    param_name=param_name,
                    owner=module,
                    geometry=bank_geometry,
                    block_size=self._checkpoint_block_size(),
                ).load(
                    [checkpoint._get_slice(key) for key in key_list],
                    _resolve_rank(),
                )
                setattr(module, attribute, grid.to(dtype=torch.float32, device=device))
                read += 1
                continue
            if len(key_list) < 2 or len(scales) != 1:
                # A lone scale grid already has its own parameter, and a bank
                # takes the branch above.
                continue
            if scales[0] in declared_grids:
                # The companion is a declared parameter: the parameter path
                # loads and compensates it.
                continue
            module_path, _, leaf = param_name.rpartition(".")
            attribute = self._sibling_scale_grid_name(leaf)
            if not attribute:
                raise Glm5NextWeightLoadError(
                    f"{param_name} carries the scale key {scales[0]!r} but its "
                    f"leaf name does not end in {_WEIGHT_LEAF_SUFFIX!r}, so the "
                    f"scale has no name to be stored under; the map and this "
                    f"reader disagree about what a weight parameter is called"
                )
            if scales[0] not in checkpoint._tensor_name_to_file:
                raise Glm5NextWeightLoadError(
                    f"the scale grid {scales[0]!r} of {param_name} is not in "
                    f"the checkpoint, so its fp8 bytes cannot be dequantised; "
                    f"reading a default of 1.0 here would use the bytes as if "
                    f"they were numbers"
                )
            module = self.get_submodule(module_path) if module_path else self
            # A sharded weight's grid is sharded with it, and this is the only route
            # such a grid travels. The grid of a dense-MLP projection is not a declared
            # parameter -- it arrives in the same map entry as its weight -- so it
            # reaches this reader and never ``loader_for_mapped_keys``. Left whole it
            # would describe the full weight while the weight itself is this rank's
            # half, and the dequant would scale the wrong blocks.
            #
            # The geometry is the weight's, converted to grid rows by
            # ``shard_geometry_for_grid``, so the block boundary is checked once in one
            # place and a misaligned shard refuses by name. ``None`` at world size 1
            # and for every replicated family.
            geometry = _shard_geometry_for(module, leaf, self.world_size)
            if geometry is None:
                grid = checkpoint._get_slice(scales[0])[:]
            else:
                # The block shape is this checkpoint's, threaded rather than
                # defaulted. Read here, inside the branch that already found a scale
                # grid, so a checkpoint with no grids never has to declare one.
                grid = sharded_scale_grid_loader(
                    geometry, param_name, self._checkpoint_block_size()
                ).load([checkpoint._get_slice(scales[0])], _resolve_rank())
            setattr(module, attribute, grid.to(dtype=torch.float32, device=device))
            read += 1
        return read

    def _run_load_time_preps(self, device: torch.device) -> tuple[int, int]:
        """Call both load-time preps, each behind its own device pre-flight.

        Returns ``(projection prep calls, scale prep calls)``.

        Once every prep on a module has run, the checkpoint tensors its class declares
        under ``RELEASED_AFTER_PREP`` are released: that class's forward reads the
        prepared operands in their place, so the originals would otherwise sit on the
        device beside them for the life of the process. A class whose forward still
        reads what it loaded declares nothing. Each module records what it released
        under :data:`RELEASED_PARAMETERS_ATTR`, and :meth:`released_parameters` reads
        those records back by tree path.

        This is the single production caller of ``prepare_projection_weights`` and
        ``prepare_scale_operands``.

        Why the pre-flight exists, and why it is authored here rather than in the
        preps. Both preps build their operands on the device of the tensors they are
        given and store them by plain ``setattr`` into a dict attribute that
        ``nn.Module._apply`` never visits. So a prep run before the weights reached the
        device would leave the operands on the CPU permanently, while every later
        ``.to(device)`` reported success. Neither prep checks a device, so the refusal
        has to live on the caller's side.
        """
        projection_calls = 0
        scale_calls = 0
        for path, module in self.named_modules():
            if hasattr(type(module), "prepare_projection_weights"):
                names = [n for n, _, _ in module.projection_widths()]
                self._require_prep_operands_on_device(path, module, names, device)
                module.prepare_projection_weights()
                projection_calls += 1
                # The absorb split reads the prepared ``kv_b_proj``, so it has to run
                # after the prep that builds it -- here, immediately after, behind the
                # same device pre-flight and inside the same single call site. A module
                # that declares no absorb split is skipped by the same ``hasattr``
                # test this loop already uses on the two preps.
                #
                # It adds no return value on purpose: this method returns
                # ``(projection calls, scale calls)`` and its callers read that pair,
                # so a third element would change a working contract for a count that
                # can be read straight off the module.
                if hasattr(type(module), "prepare_absorb_weights"):
                    module.prepare_absorb_weights()
            # The checkpoint's grids are at 128-tile granularity and arrive in the
            # loader's frame, so the frame bridge runs here -- on the load path, after
            # the shards are attached and before the prep reads the grid two branches
            # below. It used to bridge the granularity too, onto a 256 public grid;
            # since the block narrowed, the prep consumes the checkpoint's own 128 grid
            # and only the frame is bridged. Gated by the same ``hasattr`` test, so a
            # module that declares no retile is skipped and the routed bank -- which
            # retiles inside its own prep -- is untouched.
            #
            # It adds no return value either, for the reason recorded just above. What
            # the retile did is on the module, in its own health record.
            if hasattr(type(module), "retile_checkpoint_scale_grids"):
                names = [
                    leaf[: -len(_WEIGHT_LEAF_SUFFIX)]
                    for leaf in _scale_prep_leaves(module)
                ]
                self._require_prep_operands_on_device(path, module, names, device)
                module.retile_checkpoint_scale_grids()
            if hasattr(type(module), "prepare_scale_operands"):
                # The projection names come off the module's own declaration tuple, so
                # this call cannot ask for a projection the shared expert does not
                # have -- and, through ``_scale_prep_leaves``, only for one whose scale
                # grid is present to be read two lines below. Passed by keyword: the
                # prep takes six positional operands, and a silent reordering of three
                # weights against three scales would compute a wrong answer at exactly
                # the right shapes.
                names = [
                    leaf[: -len(_WEIGHT_LEAF_SUFFIX)]
                    for leaf in _scale_prep_leaves(module)
                ]
                self._require_prep_operands_on_device(path, module, names, device)
                operands = {}
                for name in names:
                    leaf = f"{name}{_WEIGHT_LEAF_SUFFIX}"
                    operands[leaf] = getattr(module, leaf)
                    operands[f"{name}_scale"] = getattr(
                        module, self._sibling_scale_grid_name(leaf)
                    )
                module.prepare_scale_operands(**operands)
                scale_calls += 1
            # The release reads the class's own declaration and nothing else: a
            # class whose forward still reads a loaded tensor declares nothing
            # and keeps everything, whichever preps it defines.
            released = getattr(type(module), "RELEASED_AFTER_PREP", ())
            if released:
                _release_replaced_parameters(module, *released)
            # The mHC bind, on the same ``hasattr`` gate the three steps above use. It
            # fires on the two decoder layer classes, and those two declare none of the
            # three preps, so this branch adds a step to the walk without reordering
            # one. It adds no return value, for the same reason the two steps above do
            # not: what the bind did is on the layer, in its own record.
            #
            # ``self.text_config`` is passed because both site instances are sized from
            # the config's own dials and a layer keeps no config of its own, and
            # ``device`` is passed because the bind refuses an operand that is not where
            # this load put it.
            if hasattr(type(module), "bind_hyper_connection_sites"):
                module.bind_hyper_connection_sites(self.text_config, device)
        return projection_calls, scale_calls

    def _require_prep_operands_on_device(
        self,
        module_path: str,
        module: nn.Module,
        names: list[str],
        device: torch.device,
    ) -> int:
        """Every operand a prep is about to read is already on ``device``.

        Returns how many operands were checked, so a caller can tell a passing check
        from a check that had nothing to look at.

        A weight or scale that is still a shape-free placeholder, or sitting on another
        device, is named here and the prep is not called -- because after the prep
        there is nothing left to detect: the operands are built and stored, on whatever
        device they were built on, and no later move touches them. An absent operand is
        skipped here and named by the prep itself.
        """
        checked = 0
        for name in names:
            for attribute in (
                f"{name}{_WEIGHT_LEAF_SUFFIX}",
                self._sibling_scale_grid_name(f"{name}{_WEIGHT_LEAF_SUFFIX}"),
            ):
                operand = getattr(module, attribute, None)
                if operand is None:
                    continue
                if torch.nn.parameter.is_lazy(operand):
                    raise Glm5NextWeightLoadError(
                        f"{module_path}.{attribute} is still a shape-free "
                        f"placeholder, so the prep would build its operands "
                        f"from a parameter no checkpoint has filled; load the "
                        f"checkpoint before preparing the operands"
                    )
                if not _is_on_device(operand.device, device):
                    raise Glm5NextWeightLoadError(
                        f"{module_path}.{attribute} is on {operand.device} but "
                        f"this load targets {device}; both preps build their "
                        f"operands on the device of the tensors they are given "
                        f"and store them where nn.Module._apply never looks, so "
                        f"preparing now would strand them on {operand.device} "
                        f"for the life of the model"
                    )
                checked += 1
        return checked

    # ── KV cache management ──────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """One ``LayerSpec`` per layer of the hybrid stack, in layer order.

        Field mapping and naming follow the other model families; the construction
        path does not, because that precedent reads instantiated submodules and this
        stack is never instantiated with weights. Geometry is read off the declared
        layer objects instead.

        The two families report different geometry from one uniform read: each layer
        exposes its attention module and that module declares
        ``num_kv_heads_per_rank``, ``head_size``, ``cache_dtype`` and
        ``cache_chunk_size``. ``sliding_window_size`` is ``None`` on every entry --
        this architecture declares no sliding window on either half.

        The four recurrent-state fields are filled by the same uniform read. A
        linear-attention layer holds a short-convolution state and a recurrent state
        instead of a key/value history, and reports the two on ``LayerSpec``'s four
        ``kda_*`` fields. ``Glm5NextKDAAttention`` derives all four from vLLM's own
        state calculators at its own rank geometry; ``Glm5NextMLAAttention`` declares
        none of them. So the read is a defaulting one -- a family that carries no
        recurrent state reports ``None`` on all four by carrying no attribute, rather
        than by this method testing which family it is looking at. That keeps the one
        loop family-blind, and it is why the runner recognises the two halves by the
        fields they carry rather than by a layer name.

        All four move together or not at all. The runner refuses a layer that declares
        part of the geometry, because the conv and recurrent carriers are paired
        positionally, so a partial set would shorten the reported page. One ``getattr``
        per field against one attribute-carrying class satisfies that by construction.
        """
        layers: list[LayerSpec] = []
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.attention
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.{attention.CACHE_NAME_SUFFIX}",
                    num_kv_heads=attention.num_kv_heads_per_rank,
                    head_size=attention.head_size,
                    dtype=attention.cache_dtype,
                    sliding_window_size=None,
                    chunk_size=attention.cache_chunk_size,
                    kda_conv_state_shape=getattr(
                        attention, "kda_conv_state_shape", None
                    ),
                    kda_recurrent_state_shape=getattr(
                        attention, "kda_recurrent_state_shape", None
                    ),
                    kda_conv_state_dtype=getattr(
                        attention, "kda_conv_state_dtype", None
                    ),
                    kda_recurrent_state_dtype=getattr(
                        attention, "kda_recurrent_state_dtype", None
                    ),
                    latent_kv=getattr(attention, "LATENT_KV_CACHE", False),
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Keep the runner's per-layer cache tensors so the runner can name them again.

        The runner calls this on every start-up and nothing guards the call, so a
        family that does not define it raises ``AttributeError`` there before any
        forward runs.

        It is a mapper and nothing else: it allocates no tensor, copies no tensor, runs
        no math and reads no weight. Every entry it keeps is a view of the runner's own
        allocation, so a layer that writes its cache writes the runner's paged buffer --
        which is what a paged cache requires, and the reason the latent view below is
        taken with ``.view`` and not ``.reshape``: a layout that cannot be viewed
        raises here instead of silently handing the layers a private copy whose writes
        are dropped.

        The key is ``get_kv_spec``'s own name, asked for rather than rebuilt, so the
        two cannot drift apart. A layer whose name is absent refuses by name and prints
        what the dict does hold.

        The two families are recognised by the fields the spec carries, never by a
        layer name -- the same test the runner makes:

        * a linear-attention (KDA) layer reports the ``kda_*`` geometry, and the runner
          allocated one ``[state_slots, *shape]`` bank per state, position 0 the short
          convolution and position 1 the recurrent state;
        * a sparse-attention (DSA) layer reports none of it, and the runner allocated
          one ``[blocks, num_kv_heads, block_size, head_size]`` bank for it. MLA keeps
          one latent vector per slot and has no value half, so the layer declares a
          latent cache and the runner sizes a page for one buffer rather than for a
          key/value pair -- which is also why ``num_kv_heads`` is 1. The bank is read
          at position 0 and there is no second position.

        The latent bank is also kept as its sequence view, because that is the shape
        ``Glm5NextDSALayer.forward`` declares: ``[slots, 1, head_size]``, one slot per
        token in position order. At one kv head the paged buffer already has that order
        -- it is block-major and each block holds ``block_size`` consecutive slots, so
        flattening gives slot ``block * block_size + offset``, exactly the runner's own
        slot number. At more than one head the flattening would interleave heads, so
        that case refuses rather than returning a view that looks right and is not.

        The layer modules are not read, only counted. Every geometry this method needs
        is already on the spec the model itself produced, so the loop walks the spec and
        the stack length is checked against it.

        The records land on ``glm5next_layer_banks``, in stack order, one mapping per
        layer. The runner reads that attribute and builds each layer's carrier from it;
        nothing else in this tree reads it. A plain tuple of plain dicts is deliberate:
        ``nn.Module.__setattr__`` leaves it alone, so ``_apply`` never walks these
        tensors and the runner stays their only owner.

        Args:
            kv_caches: the runner's ``layer name -> list of tensors`` mapping, exactly
                what ``initialize_kv_cache`` returns.

        Raises:
            ValueError: the spec and the stack disagree on how many layers there are, a
                layer's spec name is absent from ``kv_caches``, a bank's shape
                disagrees with the spec that asked for it, a layer reports part of its
                recurrent geometry, or a latent bank declares more than one KV head.
        """
        spec_layers = self.get_kv_spec().layers
        stack = len(self.model.layers)
        if len(spec_layers) != stack:
            raise ValueError(
                f"get_kv_spec reports {len(spec_layers)} layer(s) and the stack "
                f"holds {stack}; the carriers are paired positionally, so a "
                f"disagreement here would hand a layer another layer's cache"
            )
        banks: list[dict[str, object]] = []
        for layer_idx, layer_spec in enumerate(spec_layers):
            name = layer_spec.name
            if name not in kv_caches:
                raise ValueError(
                    f"kv_caches has no entry for KV layer '{name}', which "
                    f"get_kv_spec reports at stack position {layer_idx}; the dict "
                    f"holds {sorted(kv_caches)}"
                )
            tensors = list(kv_caches[name])
            recurrent = (
                layer_spec.kda_conv_state_shape,
                layer_spec.kda_recurrent_state_shape,
            )
            if all(value is not None for value in recurrent):
                if len(tensors) != 2:
                    raise ValueError(
                        f"KV layer '{name}' reports recurrent geometry, so the "
                        f"runner allocates exactly two state banks, position 0 the "
                        f"short convolution and position 1 the recurrent state; "
                        f"kv_caches holds {len(tensors)} tensor(s)"
                    )
                record: dict[str, object] = {
                    "name": name,
                    "layer_index": layer_idx,
                    "family": "linear_attn",
                }
                for key, tensor, want in (
                    ("conv_state", tensors[0], recurrent[0]),
                    ("recurrent_state", tensors[1], recurrent[1]),
                ):
                    if tuple(tensor.shape[1:]) != tuple(want):
                        raise ValueError(
                            f"KV layer '{name}' has a {key} bank of "
                            f"{tuple(tensor.shape)}; the spec asked for one "
                            f"{tuple(want)} state per slot, so the bank must be "
                            f"[slots, {', '.join(str(v) for v in want)}]"
                        )
                    record[key] = tensor
                if int(tensors[0].shape[0]) != int(tensors[1].shape[0]):
                    raise ValueError(
                        f"KV layer '{name}' has {int(tensors[0].shape[0])} "
                        f"convolution slot(s) against "
                        f"{int(tensors[1].shape[0])} recurrent slot(s); the two "
                        f"states of one request live at one slot number"
                    )
                record["state_slots"] = int(tensors[0].shape[0])
                banks.append(record)
                continue
            if any(value is not None for value in recurrent):
                raise ValueError(
                    f"KV layer '{name}' reports part of its recurrent geometry "
                    f"({recurrent}); the two states are paired positionally, so a "
                    f"missing member would shorten the page the runner allocated"
                )
            if not tensors:
                raise ValueError(
                    f"KV layer '{name}' has no cache tensor at all; the runner "
                    f"allocates one latent bank for a sparse-attention layer"
                )
            bank = tensors[0]
            if bank.dim() != 4:
                raise ValueError(
                    f"KV layer '{name}' has a latent bank of {tuple(bank.shape)}; "
                    f"the runner allocates [blocks, num_kv_heads, block_size, "
                    f"head_size] for a latent cache"
                )
            blocks, heads, block_size, width = (int(value) for value in bank.shape)
            if heads != int(layer_spec.num_kv_heads) or width != int(
                layer_spec.head_size
            ):
                raise ValueError(
                    f"KV layer '{name}' has a latent bank of {tuple(bank.shape)}, "
                    f"whose head count and width are ({heads}, {width}); the spec "
                    f"this model produced asked for "
                    f"({int(layer_spec.num_kv_heads)}, "
                    f"{int(layer_spec.head_size)})"
                )
            if heads != 1:
                raise ValueError(
                    f"KV layer '{name}' declares {heads} KV heads; the sequence "
                    f"view this method keeps is the paged bank flattened over "
                    f"blocks and slots, which is that sequence's slot order only "
                    f"at ONE head -- at more the flattening interleaves heads"
                )
            banks.append(
                {
                    "name": name,
                    "layer_index": layer_idx,
                    "family": "self_attn",
                    "latent_bank": bank,
                    "latent_cache": bank.view(blocks * block_size, heads, width),
                    "blocks": blocks,
                    "block_size": block_size,
                    "slots": blocks * block_size,
                    "head_size": width,
                }
            )
        self.glm5next_layer_banks = tuple(banks)

    # ── forward ──────────────────────────────────────────────────────────

    def _head_weight(self) -> torch.Tensor:
        """The tensor the vocabulary projection multiplies by, tied or untied.

        Both arms return a ``[vocab, hidden]`` tensor, which is the orientation
        :func:`torch.nn.functional.linear` wants and the orientation both
        checkpoint keys already have, so the caller needs no transpose and no
        branch of its own.

        Which arm is production: the real checkpoint declares ``tie_word_embeddings``
        false, so the untied arm is the one that runs and the tied arm exists because
        the config admits it. ``__init__`` above declares ``lm_head_weight`` only when
        the flag is false, mirroring the weight map's own condition.

        A tied head reads the embedding table itself, not a copy of it. The map adds no
        ``lm_head.weight`` entry in that case, so there is no second tensor to read and
        nothing to keep in step; the table lives on ``self.model`` because that is
        where the embedding lookup is.
        """
        if self.text_config.tie_word_embeddings:
            table = self.model.embed_tokens_weight
            if table is None:
                raise ValueError(
                    "Glm5NextForConditionalGeneration ties its head to the "
                    "embedding table, and model.embed_tokens_weight is None; "
                    "the table is a mapped checkpoint tensor and nothing was "
                    "loaded onto it"
                )
            return table
        weight = self.lm_head_weight
        if weight is None:
            raise ValueError(
                "Glm5NextForConditionalGeneration has no lm_head_weight; the "
                "head is a mapped checkpoint tensor and nothing was loaded onto it"
            )
        return weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        layer_carriers: Sequence[dict],
        sampling_positions: torch.Tensor,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        collect_layer_streams: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Logits for the rows the caller wants sampled: stack, select, project.

        The inter-layer carrier is the decoder stack's business, not this root's: the
        streams begin and end inside :meth:`Glm5NextModel.forward`, which expands the
        embedding across the stream axis, mixes each sublayer output back through its
        mHC site, and collapses the streams with an unweighted mean before the final
        norm. So the ``[T, H]`` this root receives is the post-collapse hidden state
        the reference projects, and the head sees the same shape it always saw.

        The head is a plain ``torch`` projection, and that is the checkpoint's own
        declaration rather than a fallback: ``lm_head`` is one of the bare entries in
        this checkpoint's ``modules_to_not_convert`` list, so its weight ships bf16 and
        no block-FP8 kernel applies to it. The nearest form in this package is
        :meth:`Glm5NextMTPModel.compute_draft_logits`, which projects the shared head
        weight the same way for the same reason.

        The selection is ``torch.index_select`` on dim 0, the family's own form, and it
        happens before the projection rather than after. On the real geometry the
        vocabulary is 154,880 wide and hidden is 4,096, so projecting every row of a
        prefill would build ``tokens x 154,880`` values to keep a handful of them;
        computing ``logits_indices`` is exactly what the runner does to avoid that.

        ``sampling_positions`` is required, with no default, because every dict that
        reaches this method is built by one of the runner's three builders and all
        three set the key unconditionally. A ``None`` default would therefore never be
        taken by the runner, and the only behaviour it could add is the whole-prefill
        projection the selection exists to prevent.

        There is no ``**kwargs`` sink, deliberately, and this is where the family
        precedent is not followed. The runner passes eight keys today plus up to four
        conditional ones, and three of them -- ``sampling_params``, ``logit_mask`` and
        ``spec_decode_metadata`` -- carry on-device sampling, which this tree
        implements nowhere: there is no sampler on this class and no
        ``on_device_sampling_config``. A sink would accept those three silently and
        return unsampled logits while reporting success. Naming the parameters instead
        makes an unconsumed key a ``TypeError`` at the call.

        The quantisation policy is resolved here, once per call, and threaded down as
        an argument -- the convention every compute method in this file follows. It is
        not cached on the instance: the resolution reads four attributes off the config
        and builds one spec, the skip list is carried by reference and only matched
        later inside ``get_scheme``, and a field would become a second authority for a
        policy the config already holds.

        ``self.model(...)`` is called, not ``self.model.forward(...)``, so torch's
        module hooks fire. Reading the stack's per-layer boundaries through forward
        hooks is how this file is instrumented, and calling the bound method directly
        would make those hooks silently not fire.

        The head is resolved first, before the stack runs. It is not needed until the
        last line, and reading it there would spend a whole 45-layer forward before
        discovering that the tensor it feeds was never loaded. Resolving it first makes
        that a named refusal with nothing dispatched, which is the same shape
        :meth:`Glm5NextModel.forward` gives its own two mapped tensors.

        Args:
            input_ids: ``[T]`` integer token ids.
            layer_carriers: one mapping per layer, in stack order, forwarded unread to
                :meth:`Glm5NextModel.forward`, which refuses a count that disagrees
                with the stack.
            sampling_positions: row indices into the stack output to project, the
                runner's ``logits_indices``.
            block_size: tokens per block, forwarded to the expert bank unread.
            moe_group: the MoE ``GroupCoordinator``, forwarded unread.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select, as an int64
                device tensor (one graph for every rank) or a python int.
            collect_layer_streams: return the stack's per-layer carriers after the
                logits, as additional outputs of this graph. Default False, and the
                runner adds the keyword only where it has a directory to write them to,
                so an ordinary serve traces the signature it always traced. The tuple
                is flat because these are graph outputs: a nested tuple is one output
                to a caller and a structure the capture has to flatten.

        Returns:
            ``[len(sampling_positions), vocab_size]`` logits, in the dtype the head
            weight and the stack output share -- or, under ``collect_layer_streams``,
            that tensor first and then one ``[T, hc_mult, H]`` carrier per layer of the
            stack.

        Raises:
            ValueError: when the head tensor this call needs was never loaded, or when
                the stack refuses its own inputs.
        """
        head = self._head_weight()
        quant_config = Glm5NextQuantConfig.from_model_config(self.config)
        # The keyword is added, not passed as False, so an ordinary forward hands its
        # stack the six keywords it always handed over and no seventh.
        collecting = {"collect_layer_streams": True} if collect_layer_streams else {}
        stack_output = self.model(
            input_ids,
            layer_carriers=layer_carriers,
            quant_config=quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
            **collecting,
        )
        if collect_layer_streams:
            hidden_states, layer_streams = stack_output
        else:
            hidden_states, layer_streams = stack_output, ()
        rows = torch.index_select(hidden_states, dim=0, index=sampling_positions)
        logits = torch.nn.functional.linear(rows, head)
        if collect_layer_streams:
            return (logits, *layer_streams)
        return logits
