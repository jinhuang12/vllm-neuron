# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``glm5_next``) blockwise-FP8 modeling skeleton.

``inc-glm53f-013`` -- WP1: model skeleton and ``KVSpec``. This module is the
file D14 declares a **coordinated merge point, rank-1-equivalent for the
duration of the campaign**: this increment creates the module header, the
imports, the ``Glm5NextForConditionalGeneration`` tree, ``get_kv_spec``, and
**forward stubs raising** :class:`NotImplementedError`, so that every later
increment's declared scope is a name that already exists. Each section below
names its D14 owner increment. **Creating a name is not implementing it.**

WHAT THIS INCREMENT CLAIMS, AND NOTHING MORE
--------------------------------------------
The KV geometry: ``get_kv_spec()`` returns one :class:`LayerSpec` per layer of
the hybrid 45-layer stack, 11 carrying the MLA latent geometry and 34 carrying
the KDA geometry. Every compute site here is a stub. That is deliberate and is
why this increment is **NON-KERNEL-CLASS** (P13): a stub computes nothing, so
there is no kernel-class functionality to place, and each compute site carries
its own substrate declaration where it lands.

NO PARAMETER IS ALLOCATED, AND THAT IS LOAD-BEARING
---------------------------------------------------
The stack this skeleton describes is 45 layers of 4096 hidden with 288 routed
experts. Allocating it is not a CPU-mode unit test. So every parameter is
**declared** rather than materialised: :func:`_declare_parameters` calls
``register_parameter(name, None)``, which is torch's own way to reserve a
parameter attribute path that a later increment fills in. The attribute path
genuinely exists on the module (``module.o_proj_weight`` returns ``None``),
the name is enumerable through :meth:`declared_parameter_names`, and the tree
allocates **zero** ``torch.nn.Parameter`` objects. ``get_kv_spec`` therefore
reads geometry off layer objects -- the same construction shape as
``llama3/model.py:1781`` and ``synthetic/synthetic.py:98`` -- without the
allocation that shape would otherwise imply.

WHERE THE PARAMETER NAMES COME FROM -- THEY ARE NOT CHOSEN HERE
---------------------------------------------------------------
Per the lead ruling *"-013's skeleton parameter names: the LANDED weight map's
param-name side is the authority"* -- recorded at
``artifacts/campaigns/glm-5.3-flash-port/increments/evidence-013.md`` L212,
since the original ``approvals/lead-ruling-013-param-name-authority.md`` was
deleted in the 2026-08-31 residue purge -- every parameter
attribute path below is **derived from**
:func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.build_weight_mappings`
as landed by ``inc-glm53f-011`` / ``inc-glm53f-012``, measured off the bytes at
HEAD -- never invented and never "improved". The derivation is asserted
mechanically in ``test/vllm_neuron/model/glm5_next/test_kv_spec.py``, so the
map-versus-skeleton equality is checkable by reading code rather than by
loading a real checkpoint on hardware. The flat ``<leaf>_weight`` shape (a
parameter on the parent module, not a ``.weight`` on a child ``Linear``) is the
map's own convention, which follows the fork's MoE precedent
``gpt_oss/model_mxfp4.py:2336-2358``.

WHAT IS DELIBERATELY ABSENT
---------------------------
The quant-method dispatcher is ``inc-glm53f-023``'s (D14, M2) and is **not**
here: ``quantization.py:33-39`` states that sequencing fact, not an assignment
to this increment. Sharded scale loading is later work
(``weight_loaders_fp8.py:1107-1114`` states the same kind of fact).
``vllm_neuron/model/kv_cache.py`` is **untouched**: widening ``LayerSpec`` with
KDA recurrent-state fields is ``inc-glm53f-015``'s declared surface at M1, and
its acceptance asserts that the pin's 6-field construction still works with
zero signature breaks. This skeleton builds pin-shaped ``LayerSpec`` values on
the six fields that exist.
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
    FP8_SCALE_SUFFIX,
    MAPPED_KEY_QUANTISED_WEIGHT,
    MAPPED_KEY_SCALE_GRID,
    MAPPED_KEY_STACKED_BANK,
    build_weight_mappings,
    classify_mapped_keys,
    consumer_block_quant_size,
    DeferredShardGeometry,
    dequantise_blockwise,
    loader_for_mapped_keys,
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
    # -- inc-glm53f-100 -- ANNOTATION ONLY, and that is the whole point. This
    #    module imports no vllm symbol at runtime and this block keeps it that
    #    way: the guard is false at import time, so nothing here can make the
    #    module fail to import on a lane without vllm, while the return type of
    #    ``_resolve_tp_group`` below still names the real class rather than
    #    ``object``. The type is the one this fork's own parallel layer names
    #    for it (``parallel/neuron_parallel_state.py:38-42``).
    from vllm.distributed.parallel_state import GroupCoordinator

# ---------------------------------------------------------------------------
# Parameter declaration -- the mechanism that makes a name exist without
# allocating the tensor behind it.
# ---------------------------------------------------------------------------


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    """True for any fp8 dtype the installed ``torch`` has.

    CLASSIFIED, NEVER NAME-MATCHED, and the rule is the fork's own rather than
    this increment's: a one-byte floating dtype whose ``str()`` names ``float8``
    (``vllm_neuron/accuracy/testing.py:86``, and its test-side twin
    ``test/vllm_neuron/accuracy/test_glm5next_tolerance_registry.py:142``).
    Classified rather than hard-coded because which fp8 dtypes exist is a
    property of the installed ``torch``, not of this file.
    """
    return dtype.itemsize == 1 and "float8" in str(dtype)


def _is_on_device(where: torch.device, target: torch.device) -> bool:
    """Is a tensor sitting on the device this load targets?

    NOT plain ``==``. ``torch.device("cpu")`` and ``torch.device("cpu", 0)``
    compare unequal while naming the same place, and a target written without an
    index means "this kind of device" rather than "index None". So the kind must
    match, and the index only has to match when the target names one.
    """
    if where.type != target.type:
        return False
    if target.index is None:
        return True
    return where.index == target.index


def _declare_parameters(module: nn.Module, *names: str) -> None:
    """Reserve parameter attribute paths on ``module`` without allocating.

    ``register_parameter(name, None)`` is torch's own declaration form: the
    name enters the module's parameter registry, ``getattr`` resolves it to
    ``None``, and no storage is created. ``named_parameters()`` and
    ``state_dict()`` both skip ``None`` entries by design, so the declared set
    is also recorded on ``declared_param_names`` -- that tuple, not
    ``named_parameters()``, is what :meth:`declared_parameter_names` walks.
    """
    for name in names:
        module.register_parameter(name, None)
    module.declared_param_names = (
        *getattr(module, "declared_param_names", ()),
        *names,
    )


# ---------------------------------------------------------------------------
# Geometry resolution for ``get_kv_spec``
# ---------------------------------------------------------------------------


def _resolve_world_size() -> int:
    """The rank count to divide head counts by, or 1 when not distributed.

    Same shape as ``synthetic/synthetic.py:99-103``, which is this fork's only
    config-derived (rather than module-derived) KV-spec precedent.
    ``NeuronConfig`` carries no tensor-parallel degree field, so the process
    group is the only available source and its absence is not an error.
    """
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def _resolve_rank() -> int:
    """This process's rank in the tensor-parallel group, or 0 when not distributed.

    ``inc-glm53f-091``. The twin of :func:`_resolve_world_size` above, and
    deliberately the same shape -- same source, same two exception types, same
    "absence is not an error" reading -- because the two values are read together
    and a pair that disagreed about what "not distributed" means would shard
    against a rank the world size does not contain.

    ``load_sharded_pipelined`` requires a rank to shard by
    (``utils/checkpoints.py:273-282``). The shipped ``qwen3`` path takes it from
    ``self.tp_group.rank_in_group`` (``qwen3/model.py:974``), which is not
    available here: ``Glm5NextForConditionalGeneration`` declares no ``tp_group``
    member, only ``world_size`` (``:3031``). Reading the process group directly is
    what ``-013`` already chose for the world size, so this keeps one convention
    in the file rather than importing a second one.

    D14 BOUNDARY, recorded because this is a new module-level helper: it sits in
    the module-level helper block that ``inc-glm53f-013`` owns, immediately after
    its twin and before ``_per_rank``. It adds no import -- ``torch`` is already
    the module's own -- and no ``-013`` helper is edited.
    """
    try:
        return torch.distributed.get_rank()
    except (RuntimeError, ValueError):
        return 0


def _per_rank(count: int, world_size: int) -> int:
    """Per-rank head count, floored at 1 (``synthetic.py:105-107``)."""
    return max(1, count // max(world_size, 1))


def _resolve_tp_group() -> GroupCoordinator | None:
    """The tensor-parallel group to reduce a row-parallel partial across.

    ``None`` means there is nothing to reduce -- one rank, so the "partial" sum
    is already the whole sum. Returning ``None`` rather than a one-rank group is
    the same reading :func:`_shard_geometry_for` makes one screen below: at world
    size 1 nothing about this increment may change the path a landed single-rank
    test takes.

    ``inc-glm53f-100``. The third of the module's ``_resolve_*`` twins and
    deliberately their shape: read the process state, treat its absence as "not
    distributed" rather than as an error, and be the ONE place in the file that
    knows how. Tests already inject world size by monkeypatching
    :func:`_resolve_world_size` (``test/vllm_neuron/model/glm5_next/
    test_load_weights.py:3180``), so a reduction group that arrives the same way
    keeps one injection convention in the file instead of two.

    WHY THE GROUP AND NOT A SECOND CONVENTION. ``get_tp_group`` is what all eight
    shipped model files use for exactly this -- ``llama3``, ``gpt_oss``,
    ``qwen3``, ``qwen3_vl`` each bind ``self.tp_group = get_tp_group()`` in a
    constructor and call ``all_reduce`` on it after a row-parallel matmul, e.g.
    ``qwen3/model.py:170`` and ``:535-537``, the convention written down at
    ``parallel/DESIGN.md:149-151``. This package's classes are built before any
    world size is known (``_declare_parameters`` / ``_materialise_declared_
    parameters``, the reason ``-094``'s geometry is a table rather than a line in
    each ``__init__``), so the group is resolved at the call site instead of
    bound in a constructor. Same group, same call, later lookup.

    THE IMPORT IS LOCAL because this module imports no vllm symbol at runtime,
    and the guard comes FIRST so a single-rank run never performs the import at
    all. Both readings have a first-hand precedent in this fork's own parallel
    layer: ``tp_barrier`` imports ``get_tp_group`` inside the function body and
    then tests ``group.world_size > 1`` before using it
    (``parallel/neuron_parallel_state.py:1584-1597``). The group's own
    ``world_size`` is re-checked here for the same reason it is checked there --
    it, not the global degree, is the authority on how many ranks the collective
    would span.

    D14 BOUNDARY, recorded because this is a new module-level helper: it sits in
    the module-level helper block ``inc-glm53f-013`` owns, after its twins and
    after ``_per_rank`` -- AFTER on purpose, so ``_per_rank`` keeps the lines
    ``weight_loaders_fp8.py:1499`` cites it by. It adds no runtime import; the
    annotation's ``GroupCoordinator`` arrives through the ``TYPE_CHECKING`` block
    in the header, which is false at import time.
    """
    if _resolve_world_size() <= 1:
        return None
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    return group if group.world_size > 1 else None


# --------------------------------------------------------------------------- #
# inc-glm53f-094 -- WHICH PARAMETER FAMILIES ARE SHARDED, AND ON WHICH DIM.
#
# WHY THIS IS A TABLE AND NOT A LINE IN EACH __init__. The shipped precedent
# (qwen3 ``model.py:224``) attaches a shard loader to a parameter object inside
# the constructor using ``self.world_size``. Neither half of that exists here:
# this package declares parameters as ``register_parameter(name, None)``
# (``_declare_parameters`` above) and builds the objects only in
# ``_materialise_declared_parameters``, and of the ten declaring classes only
# ``Glm5NextKDAAttention`` and ``Glm5NextModel`` are told a world size. So the
# geometry is declared per CLASS here and consumed at the one attachment site
# that already exists, which is where the root's world size lives. Ruled at
# design entry ``design-20260905-ah``.
#
# WHAT EACH ENTRY MEANS. ``shard_dim`` is the dimension in the FINAL PARAMETER
# shape (``utils/weight_loader.py:150``), which in this package is the checkpoint
# shape ``[out_features, in_features]`` -- the projection weights are transposed
# once at load time by ``prepare_projection_weights`` (``:2806``), never at load,
# so what a loader slices is the untransposed tensor. ``width`` returns THIS
# RANK's extent along that dim. Both readings are recorded with their cites in
# ``increments/shard-table-094.md`` Part 1b, and the table is the frozen artifact
# this code is checked against.
#
# EVERY FAMILY NOT NAMED HERE IS REPLICATED, and that is a declaration rather
# than a default: the table's Parts 1 and 4 record the reason per family, and the
# majority of the surface is in them -- the whole indexer, every norm, the router,
# the embeddings and the head. A replicated family reaches
# ``loader_for_mapped_keys`` with ``geometry=None`` and therefore takes exactly
# the path it took before this increment.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _DeclaredShard:
    """One family's declared shard: the dim, and this rank's extent along it.

    ``width`` IS ``None`` FOR A FAMILY WHOSE WIDTH ARRIVES WITH THE TENSOR
    (``inc-glm53f-101``). Two cases need it and both are readings rather than
    conveniences: ``Glm5NextSharedExperts`` carries no width at all
    (``:1520-1535`` -- ``num_shared_experts`` and ``swiglu_limit``), and the dense
    MLP's three are moved onto the same route so ONE rule pads both, because
    ``intermediate_size`` is a config product and the checkpoint tensor is the
    width source that cannot disagree with the weights.

    ``pad_to_consumer_block`` rounds the full width UP to the smallest multiple of
    ``world_size * BLOCK_QUANT_SIZE`` before dividing, so every rank's shard is a
    whole consumer block. It is a FLAG and not the number, because the number
    lives in the consumer and is imported at attachment time rather than at
    import time -- see
    :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.consumer_block_quant_size`.
    Ruled at design entry ``design-20260905-ap``, remedy part 2.
    """

    shard_dim: int
    width: Callable[[nn.Module, int], int] | None
    #: Why this family is sharded on this dim, in one clause, for the reader who
    #: finds this table before finding the shard table.
    because: str
    pad_to_consumer_block: bool = False
    #: True for the routed expert bank alone. Its intermediate width is divided by
    #: ``tp_per_ep = world_size // ep_degree`` -- the ranks INSIDE one
    #: expert-parallel group -- and not by the whole world, because the experts
    #: themselves are already divided across the groups. The package's own
    #: definition: ``parallel/neuron_parallel_state.py:218-232`` lays the groups out
    #: that way and ``functional/moe/moe_blockwise.py:55-56`` divides the same two
    #: numbers. Ruled at design entry ``design-20260905-ap``, remedy part 3(a).
    shards_within_expert_parallel_group: bool = False
    #: This family's per-rank shard must ALREADY be a whole consumer block, and the
    #: load REFUSES BY NAME when it is not. ``inc-glm53f-106``, and it is the other
    #: half of dropping ``pad_to_consumer_block`` from the routed bank: the two flags
    #: are the two answers to one question -- an inadmissible width is either PADDED
    #: up to admissibility or REFUSED -- and the bank's answer is refuse, because its
    #: sharded axis carries per-expert structure that a pad silently inflates (ruled
    #: at DECISIONS 162: "a pad that is right on an unstructured axis is wrong on an
    #: axis that carries structure"). Declaring the two separately, rather than
    #: inferring refuse-when-not-padded, is DECISIONS 296-298: the inference is a
    #: proxy for "is consumed by the block-FP8 kernel" that holds only while the bank
    #: is the sole family of its kind, and it would wrongly refuse a future
    #: unquantised deferred family. Mutually exclusive with the pad, enforced in
    #: ``DeferredShardGeometry.__post_init__`` rather than trusted here.
    require_consumer_block: bool = False


def _kda_head_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``heads * head_dim``, per-rank ALREADY.

    ``Glm5NextKDAAttention`` divides in its own constructor --
    ``num_kv_heads_per_rank = _per_rank(num_heads, world_size)`` -- so the world
    size is NOT applied again here. Dividing twice is the defect this function
    exists to make impossible to write by accident: the argument is accepted and
    deliberately unused, so the signature stays uniform across the table.
    """
    del world_size
    return int(module.num_kv_heads_per_rank) * int(module.head_dim)


def _kda_head_count(module: nn.Module, world_size: int) -> int:
    """This rank's head count, for the three families whose extent is H, not H*D."""
    del world_size
    return int(module.num_kv_heads_per_rank)


# THE DENSE MLP'S WIDTH FUNCTION IS GONE (``inc-glm53f-101``), and its removal is
# the fix rather than a tidy-up. It read ``_per_rank(module.intermediate_size,
# world_size)``, which FLOORS: at the registered TP=64 the real 12288 gave 192 rows
# per rank, and 192 is neither a whole 128-row checkpoint tile nor a whole 256-row
# consumer block, so every dense projection on every dense layer refused by name
# and the model could not load at all. Measured in
# ``increments/probe-101-grid-host-r6.out``. The three families now declare
# ``width=None`` with ``pad_to_consumer_block=True``: the width is read off the
# checkpoint tensor and rounded UP to a multiple of ``world_size * 256``, which is
# 16384 at 64 and 32 ranks and 12288 unchanged at 16 or fewer. Ruled at design
# entry ``design-20260905-ap``, remedy part 2.


# -- inc-glm53f-100 -- the MLA families' three widths. ---------------------- #
#
# WHICH PATTERN THESE FOLLOW, and why it is NOT the KDA's.
# ``Glm5NextMLAAttention`` keeps ``num_attention_heads`` as the MODEL's head
# count at every world size -- the attribute is never overwritten with a per-rank
# value -- so the division happens HERE, and NOT the way ``_kda_head_count``
# deliberately refuses to divide a second time by reading an attribute that is
# already per-rank.
#
# THE DENSE MLP WAS THIS COMMENT'S EXAMPLE AND IS NO LONGER ONE. Its width
# function was deleted by ``inc-glm53f-101`` -- the reason is recorded a few
# lines above this table -- and its three leaves now declare ``width=None`` and
# let the loader read the extent off the checkpoint tensor. So of the five
# declaring families, three defer and only KDA and MLA resolve a width in this
# file. Pointing at a deleted function as the pattern to follow would be a
# dangling reference; naming the current split is the fact that survives the
# next table change.
# The class's own per-rank reader is :meth:`Glm5NextMLAAttention._heads_per_rank`,
# which floors through the same ``_per_rank``, so the width a loader slices to and
# the width ``projection_widths`` expects are the same expression on the same two
# inputs. They also read the same world size in production: the root binds
# ``self.world_size = _resolve_world_size()`` (``:4945``) and passes exactly that
# to ``_shard_geometry_for`` (``:5154``, ``:5445``), which is what the class's
# reader resolves too. If a caller ever made the two disagree the load would stop
# at ``prepare_projection_weights``'s width check with the site named, rather
# than compute the wrong function at plausible shapes -- and conjunct (1) of this
# increment's acceptance asserts the agreement rather than assuming it.
#
# WHAT IS NOT HERE IS A DECLARATION. ``q_a_proj``, ``kv_a_proj_with_mqa`` and both
# latent layernorms take no entry in the table below because they are REPLICATED:
# they read or write the compressed latent, of which MLA keeps one per token
# rather than one per head (``NUM_LATENT_KV_HEADS = 1``), so there is no head axis
# to split. ``increments/shard-table-094.md`` Part 5 records that per family.


def _mla_head_count(module: nn.Module, world_size: int) -> int:
    """This rank's MLA head count. The one divisor the three widths below share."""
    return _per_rank(int(module.num_attention_heads), world_size)


def _mla_q_b_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``q_b_proj`` output width: its heads at the full query width.

    The query head width is ``qk_nope_head_dim + qk_rope_head_dim``, summed rather
    than taken from the nope width alone for the reason ``projection_widths``
    states: the rotary slice is 0 on this checkpoint and that 0 is a value, so a
    config that had one would be short by exactly that slice.
    """
    return _mla_head_count(module, world_size) * (
        int(module.qk_nope_head_dim) + int(module.qk_rope_head_dim)
    )


def _mla_kv_b_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``kv_b_proj`` output width: its heads' key AND value halves.

    Both halves are one weight and are sharded as one, because the absorb split
    cuts this same weight per head (``prepare_absorb_weights``) -- a rank that
    held the keys of its heads and the values of another's would mix heads at
    exactly the right total width.
    """
    return _mla_head_count(module, world_size) * (
        int(module.qk_nope_head_dim) + int(module.v_head_dim)
    )


def _mla_o_proj_width(module: nn.Module, world_size: int) -> int:
    """This rank's ``o_proj`` INPUT width: its heads' value width.

    Row-parallel, so what each rank computes is a partial sum over its own heads
    and the cross-rank sum is the consumer's business -- here that consumer is
    :meth:`Glm5NextMLAAttention.project_output`, which performs it, unlike the KDA
    sibling's ``o_proj`` (a gap this increment does not close; it is recorded as
    ``inc-glm53f-094`` debt against ``inc-glm53f-054``'s design read).
    """
    return _mla_head_count(module, world_size) * int(module.v_head_dim)


#: Declaring class name -> declared leaf -> its shard. Keyed by NAME rather than by
#: the class object so this table can sit above every class it names.
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
        # Row-parallel: the head width is o_proj's INPUT, so the partial sum is
        # per-rank and the reduction is the consumer's business, not the loader's.
        "o_proj_weight": _DeclaredShard(
            1, _kda_head_width, "row-parallel -- the head width is its input"
        ),
        # One value per head, not per channel.
        "b_proj_weight": _DeclaredShard(0, _kda_head_count, "one row per head"),
        "A_log": _DeclaredShard(0, _kda_head_count, "one decay per head"),
        "dt_bias": _DeclaredShard(0, _kda_head_count, "one bias per head"),
    },
    # -- inc-glm53f-100 -- the MLA families the ratified table defers to it
    #    (``increments/shard-table-094.md`` Part 5, Group B). THREE leaves, one per
    #    projection whose declared width carries the head count; the two latent
    #    projections are absent because they are replicated, which Part 5 records
    #    as a declaration rather than a default.
    "Glm5NextMLAAttention": {
        # Column-parallel: each rank owns whole heads of the projection's output.
        # ``q_b_proj`` expands the query latent into per-head queries, so its
        # OUTPUT carries the head axis and dim 0 is that axis in the checkpoint
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
        # Row-parallel: the head width is o_proj's INPUT, so each rank computes a
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
    # inc-glm53f-101. The shared expert's three, on the same route as the dense
    # three above and for the stronger version of the same reason: this class holds
    # NO width to divide, so there is nothing at the attachment site to resolve.
    # It is NOT an expert-parallel entity -- it runs on every token, so it shards
    # its intermediate width across the FULL world like the dense MLP and its
    # consumer is the dense blockwise route. Ruled at design entry
    # ``design-20260905-ap``, remedy part 3(d); the alternative (an EP-TP-group
    # shard replicated across groups) is recorded there for a stage-7 ruling and is
    # not authored here.
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
    # inc-glm53f-101, remedy part 3. The routed bank's three, and the ONE family
    # whose divisor is not the world size. Its experts are already divided across
    # the expert-parallel groups, so what remains to divide inside a group is the
    # intermediate width, by tp_per_ep.
    #
    # ``shard_dim`` HERE IS THE PER-EXPERT CHECKPOINT DIM, not the stacked
    # parameter's, and the difference is stated because both are defensible. The
    # bank's parameter carries a LEADING expert axis, so its dims are
    # [E, out, in] and the intermediate sits one place to the right of the number
    # below. The number is the checkpoint tensor's because that is what the bank
    # loader actually slices -- it selects each expert's columns BEFORE stacking
    # (``weight_loaders_fp8.py``'s ``_stack_local_expert_weights``), so a dim
    # counted in the stacked shape would slice the wrong axis of every expert.
    # THE THREE ROWS BELOW SPELL THEIR FLAGS AS KEYWORDS, and that is deliberate
    # rather than a style preference (``inc-glm53f-106``). They used to pass
    # ``True, True`` positionally for the pad and the EP divisor; dropping the pad
    # would have shifted the SECOND ``True`` into the pad's place and defaulted
    # ``shards_within_expert_parallel_group`` to False, which divides the bank by the
    # world instead of by its group and would have been a silent behaviour change in a
    # repair whose whole subject is a silent behaviour. Naming them also makes them
    # findable: a keyword grep over this table could not see the old positional value,
    # which is how the defect review B90-101 found survived thirteen rounds.
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

    ``inc-glm53f-094``. The single reader of :data:`_SHARD_GEOMETRY`, called from
    ``_materialise_declared_parameters`` for every declared parameter.

    AT WORLD SIZE 1 IT RETURNS ``None`` FOR EVERYTHING, deliberately. A one-rank
    shard is the whole tensor, so attaching a loader to say so would put a second
    code path under every landed single-rank test -- and those tests are the
    control that the sharded path is what changed. This is also what makes
    conjunct (1)'s ``world_size=1`` reading a real control rather than a
    restatement.

    TWO KINDS COME BACK SINCE ``inc-glm53f-101``. A family that declares a width
    function gets its extent resolved here, exactly as before. A family declaring
    ``width=None`` gets a :class:`DeferredShardGeometry` instead, which carries the
    dim, the rank count and the pad multiple and leaves the extent to the loader
    that will hold the tensor. The consumer's block size is imported HERE rather
    than written into the table, so the number has one home.

    THE ROUTED BANK'S RANK COUNT IS NOT THE WORLD SIZE, and it is the only family
    of which that is true. Its experts are divided across the expert-parallel
    groups already, so what one group divides is the intermediate width, by
    ``tp_per_ep = world_size // ep_degree``. At ``tp_per_ep == 1`` the bank is
    whole inside its group and ``None`` comes back -- and that happens when the world
    size EQUALS the degree, one rank per group, NOT at degree 1. Corrected by
    ``inc-glm53f-106`` (review B90-101, finding B1): at degree 1 with a world above 1,
    ``tp_per_ep`` is the whole world, so a deferred geometry comes back and the bank's
    intermediate width is sharded across every rank. The old sentence named degree 1
    as the case that returns ``None``, which is the opposite of what the branch below
    computes.

    A SCALE GRID ANSWERS WITH ITS WEIGHT'S GEOMETRY, and that is
    ``inc-glm53f-105``'s addition. A blockwise grid holds one value per weight
    tile, so it has no shard of its own to declare -- it is sharded if and only if
    its weight is, on the same dimension. The table therefore keeps one row per
    WEIGHT and a grid leaf is resolved back to it, rather than the table carrying
    near-duplicate rows that could disagree with the weights they describe.
    ``weight_loaders_fp8.py`` converts weight rows to grid rows on arrival
    (``shard_geometry_for_grid``), which is why what crosses this boundary is the
    weight's geometry and not the grid's.
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
    # Declared before the branch because the geometry CARRIES it: the EP-TP group
    # the column reader consults only exists above 1, so a bank geometry that did
    # not say which degree it was built at made the reader treat degree 1 as a
    # missing group. Every non-bank family is built at 1 and means it.
    ep_degree = 1
    if declared.shards_within_expert_parallel_group:
        # The divisor is the ranks inside ONE expert-parallel group. Read off the
        # module that already resolved both degrees at construction
        # (``Glm5NextRoutedExperts.__init__``: ``tp_degree`` is the world size and
        # ``ep_degree`` comes from ``_resolve_ep_degree``), so the two halves cannot
        # disagree with the partition the same object built.
        ep_degree = max(1, int(getattr(module, "ep_degree", 1)))
        num_shards = max(1, world_size // ep_degree)
        if num_shards <= 1:
            # One rank per expert-parallel group: every group member holds its
            # experts whole, so there is no intermediate shard and the replicated
            # path is the right one.
            return None
    if declared.width is None:
        # Both flags read the SAME number from the consumer and mean opposite things
        # about an inadmissible width: pad it up, or refuse it. ``inc-glm53f-106`` adds
        # the second. The number is imported either way, never typed here, for the
        # reason ``consumer_block_quant_size``'s own docstring gives.
        return DeferredShardGeometry(
            shard_dim=declared.shard_dim,
            num_shards=num_shards,
            pad_to_multiple_of=(
                consumer_block_quant_size() if declared.pad_to_consumer_block else None
            ),
            expert_parallel_degree=ep_degree,
            require_multiple_of=(
                consumer_block_quant_size() if declared.require_consumer_block else None
            ),
        )
    if declared.shards_within_expert_parallel_group:
        # No family declares both today, and this refusal is what keeps it that
        # way: a resolved width is divided by the WORLD size below, so a family
        # that also asked for the expert-parallel divisor would get one divisor in
        # its extent and another in its rank count and shard itself wrong.
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

    A missing key is raised rather than defaulted: silently substituting a
    head count would make the KDA half of the KV spec wrong in a way no shape
    assertion downstream could attribute back to here.
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

    The MLA latent is **not** a quantised cache: ``quantization.py:69-74``
    records that this checkpoint's ``quantization_config`` declares no
    ``kv_cache_quant_algo`` at all and that the blockwise weight scheme is
    rejected for KV caches. So the cache dtype is the model's own dtype.
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

    ``NeuronConfig.kda_state_dtype`` (``neuron_config.py:181-184``) is declared
    as *"a torch dtype NAME (e.g. "bfloat16") so it survives a JSON
    additional_config round-trip. None = follow the model's own dtype."* This
    resolver implements that declared rule, which is why it has a real
    ``None`` branch: an unresolved override falls back to the model dtype and
    the result is **never** ``None``. An override that does not name a torch
    dtype is raised rather than passed through as a string, because a ``str``
    reaching ``LayerSpec.dtype`` would be a silently wrong cache allocation.
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

    On this checkpoint ``qk_rope_head_dim == 0`` because ``mla_use_nope`` is
    set (``config.py:123-127``), so the sum **is** ``kv_lora_rank == 512``.
    The sum is used rather than the bare rank because the cached latent is the
    compressed KV vector concatenated with whatever rotary slice exists, and
    on a config that had one the bare rank would be short by that slice.
    """
    return int(text_config.kv_lora_rank) + int(text_config.qk_rope_head_dim)


# ---------------------------------------------------------------------------
# ``Glm5NextQuantConfig`` / quant-method selection -- D14 owner:
# ``inc-glm53f-023`` (M2, Lane B 1st). LANDED HERE.
#
# THE DISPATCHER GAP THIS CLOSES. ``quantization.py`` already parsed the
# checkpoint's ``quantization_config`` into a ``QuantizationSpec`` carrying
# ``weight_block_size (128, 128)``. What did not exist was anything that turned
# that spec into a METHOD -- so a spec declaring blockwise FP8 resolved to
# nothing, and a call site had no way to ask "what do I run for this module?".
# Measured at the unmodified parent ``6affd98``: the spec reported
# ``(128, 128)`` while ``Glm5NextQuantConfig()`` raised, and a well-formed but
# unsupported shape such as ``[64, 64]`` was ACCEPTED silently.
#
# WHAT THIS SECTION DOES NOT DO, deliberately:
#   * it does NOT attach itself to ``Glm5NextForConditionalGeneration`` -- that
#     tree is ``-013``'s / ``-054``'s D14 section;
#   * it does NOT reach a kernel; the MoE block-quant call site is ``-027``'s;
#   * it does NOT name or import the vendor quantisation enum. There is no
#     blockwise member in it at this pin and adding one is forbidden to this
#     campaign (plan section 11, constraint B.6), so the route is D5(b)'s direct
#     inner-kernel call and this class carries a plugin-side method object.
#
# Imports are FUNCTION-LOCAL rather than added to the module import block, for
# two reasons: the import block is ``-013``'s D14 section and this increment
# does not widen its surface into it; and it is the file family's own idiom --
# ``llama3/quantization.py`` imports its modeling module inside
# ``resolve_attention_mlp_classes`` for the same reason.
# ---------------------------------------------------------------------------


class Glm5NextQuantConfig:
    """Per-module blockwise-FP8 quantisation policy for this arch.

    Holds the parsed :class:`~vllm_neuron.model.glm5_next.quantization.QuantizationSpec`
    and the method resolved from it, and answers the one question a modeling
    call site has: *what do I run for this module?*

    Attributes:
        spec: The parsed spec, or ``None`` for an unquantized checkpoint.
        method: The method resolved for the model-wide scheme, or ``None`` when
            nothing is quantized. Per-module resolution goes through
            :meth:`get_quant_method`, which is the form that survives mixed
            precision; this attribute is the model-wide answer today because
            :meth:`QuantizationSpec.get_scheme` returns one scheme for every
            module.

    Construction raises rather than degrading when the checkpoint declares a
    block shape this build has no authored path for -- see
    :data:`~vllm_neuron.model.glm5_next.quantization.SUPPORTED_WEIGHT_BLOCK_SIZES`.
    """

    def __init__(self, spec: object | None) -> None:
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        self.spec = spec
        self.method = resolve_quant_method(spec)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_model_config(cls, config: Glm5NextConfig) -> Glm5NextQuantConfig:
        """Build from the config the model itself is built from.

        This is the recognition path end to end: the checkpoint's
        ``quantization_config`` was lifted onto :class:`Glm5NextConfig` by
        ``config.py``, is parsed into a spec by
        :meth:`QuantizationSpec.from_model_config`, and is resolved to a method
        here. Nothing in the chain is hand-fed.
        """
        from vllm_neuron.model.glm5_next.quantization import QuantizationSpec

        return cls(QuantizationSpec.from_model_config(config))

    # ------------------------------------------------------------------
    # Per-module query
    # ------------------------------------------------------------------
    def get_quant_method(
        self,
        layer_index: int | None = None,
        prefix: str = "",
    ) -> object | None:
        """Return the method for the module at ``(layer_index, prefix)``.

        Returns ``None`` when that module is not quantized, which means "run the
        unquantized path". The arguments are forwarded to
        :meth:`QuantizationSpec.get_scheme`, so when per-layer dispatch becomes
        real it lands there and every call site here is already passing what it
        needs.
        """
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        return resolve_quant_method(self.spec, layer_index, prefix)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def is_block_quantized(self) -> bool:
        """True when a blockwise-FP8 method was resolved."""
        return self.method is not None

    @property
    def block_shape(self) -> tuple[int, int] | None:
        """``(block_h, block_w)`` of the resolved method, or ``None``.

        Delegates rather than storing a copy: the method is the authority for
        the shape, and a second copy is a second thing that can go stale.
        """
        return None if self.method is None else self.method.block_shape


# ---------------------------------------------------------------------------
# ``Glm5NextHyperConnection`` -- mHC wiring. D14 owner: ``inc-glm53f-030``
# (M2, Lane B 6th, and Lane B's last).
#
# WHAT LANDS HERE, AND WHAT DELIBERATELY DOES NOT
# -----------------------------------------------
# The whole mHC layer: the pre block, ONE entry into ``inc-glm53f-028``'s
# Sinkhorn seam, ONE entry into ``inc-glm53f-029``'s combine seam, and the
# residual plumbing between them. **Calling this layer from the decoder is NOT
# here.** ``Glm5NextKDALayer`` and ``Glm5NextDSALayer`` are other increments'
# D14 sections (``-038``, ``-051``), and D14's rule is to raise rather than
# widen, so this increment stops at its own section boundary and the decoder
# call site is left to those owners. :meth:`Glm5NextHyperConnection.forward` is
# shaped as the seam they will call.
#
# NO WEIGHT-MAP FAMILY IS ADDED, and that is a lead ruling rather than an
# omission. ``weight_loaders_fp8.py:88-89`` declared ``multi_hyper_connections``
# absent here -- no settled leaf names -- until ``inc-glm53f-078`` read the real
# shard index and grounded it. This increment's acceptance is a synthetic
# tiny case that builds these parameters in-test, so the map is not on its
# route, and the checkpoint-leaf question stays a lead-owned open question for a
# later design revision. ``weight_loaders_fp8.py`` is untouched.
#
# NO TOKEN TILING IS AUTHORED HERE, also a lead ruling. ``-028``'s kernel
# refuses its ``M`` above ``PARTITION_MAX`` by raising ``SinkhornError`` and has
# no torch path (``sinkhorn.py:321-330``); the refusal is allowed to propagate
# unchanged. A host-side tiling loop and a large-``M`` torch fallback are both
# out of scope, the second by P13 outright. The measured consequence -- see the
# token-ceiling note on :meth:`mhc_pre` -- is recorded for the design revision
# that settles the policy.
# ---------------------------------------------------------------------------


class Glm5NextHyperConnectionError(ValueError):
    """A rank, extent or configuration this layer refuses, named not coerced.

    Only the cross-argument agreements the two seams cannot see are checked
    here. Each seam already refuses its own extents by name
    (``sinkhorn.py:312-342``, ``hyper_connection.py:238-308``), and restating
    those bounds would create a second authority that can drift from the first
    -- the same reason ``-033``'s route error checks only its own call site.
    """


class Glm5NextHyperConnection(nn.Module):
    """Multi-hyper-connection (mHC) residual mixing, one layer call.

    The operation is the pinned base's own pair of mHC blocks, ``MHCPreOp`` and
    ``MHCPostOp`` (``vllm/model_executor/layers/mhc.py`` at ``vllm==0.24.0``),
    with the two device-side pieces routed to this fork's NKI kernels:

    1. **pre** -- project the ``hc_mult`` residual streams through ``fn``, RMS
       scale, and split the result into three heads: ``pre_mix`` (which folds
       the streams into the sub-block's single input), ``post_mix``, and the
       ``[S, S]`` per-token stream-mixing matrix. The mixing matrix is then
       Sinkhorn-normalised by **``inc-glm53f-028``'s kernel**.
    2. the caller's sub-block runs on the folded input.
    3. **post** -- mix the streams back out by **``inc-glm53f-029``'s kernel**.

    WHERE THE ARITHMETIC COMES FROM -- IT IS NOT CHOSEN HERE
    -------------------------------------------------------
    Every line below is the base's, read at tag ``v0.24.0`` because the
    campaign's target base is the 0.24 line, and read in **two independent
    spellings that agree**: ``mhc_pre_torch`` / ``mhc_post_torch`` in
    ``vllm/model_executor/kernels/mhc/torch.py`` (the plain-torch backend), and
    ``mhc_pre_ref`` / ``mhc_post_ref`` in ``tests/kernels/test_mhc_kernels.py``
    (the TileLang-repo reference). The acceptance carries both and asserts they
    agree, so the transcription rests on two statements rather than on one
    reading of one file.

    THE ONE COMPOSITION QUESTION THIS SECTION HAD TO ANSWER
    ------------------------------------------------------
    ``-028``'s seam normalises **one** ``[M, N]`` matrix; the target needs
    **``T`` independent** ``[S, S]`` ones, and this increment's route predicate
    declares the Sinkhorn seam is entered **exactly once per layer call**. The
    two are reconciled by a **block-diagonal embedding**: the ``T`` little
    matrices are scattered onto the diagonal of one ``[T*S, T*S]`` matrix, so
    ``-028``'s column target ``M / N`` is exactly ``1`` -- the target's own
    column target -- and the off-diagonal zeros stay zero under multiplicative
    rescaling, which makes every row sum and every column sum range over
    exactly one token's block. **The alternative was measured and rejected:** a
    flat ``[T*S, S]`` reshape lets ``-028``'s column pass sum ACROSS tokens, and
    ``probe-030-composition-algebra.out`` reads ``max_abs`` up to ``4.68e-01``
    against the target for it while the block-diagonal embedding reads
    ``8.99e-07``, with the off-block maximum exactly ``0.0``. This is layout,
    not authored numerics: all of the Sinkhorn arithmetic stays inside ``-028``'s
    kernel.

    :func:`torch.block_diag` is torch's own member, reused rather than written.

    TWO DIVERGENCES FROM THE BASE THAT THIS LAYER CANNOT REMOVE
    ----------------------------------------------------------
    Both live inside ``-028``'s landed kernel, so both are recorded rather than
    repaired: the base adds ``hc_sinkhorn_eps`` to every Sinkhorn denominator
    while ``-028`` adds an inert ``1e-30`` (``sinkhorn.py:148``), and the base's
    schedule is ``softmax`` then one column pass then ``(R-1)`` row/column pairs
    while ``-028``'s is ``R`` pairs starting with a row pass. Sinkhorn-Knopp has
    one fixed point, so at the target's ``20`` iterations the gap is small --
    ``probe-030-layer-delta.out`` measures it using at most **3.2 %** of the
    declared tolerance budget, in exact double precision so the reading is the
    schedule-and-eps gap alone. The acceptance measures it again through the
    real kernels.

    PRECISION: fp32 IN AND OUT, following ``-028`` and ``-029``
    ---------------------------------------------------------
    The base's mHC takes bf16 and its own kernel test therefore compares at
    ``atol=5e-2``; this increment is declared at ``atol=1e-5``, three orders
    tighter, and bf16's ~3 decimal digits cannot express that difference. Both
    seams are fp32 in and fp32 out for exactly this reason, so this layer keeps
    fp32 across them and returns the seam's own dtype. Casting the result to the
    decoder's residual dtype is the decoder's business, as ``-027`` and ``-033``
    both left it.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        neuron_config: NeuronConfig | None = None,
        post_mult_value: float = 1.0,
    ) -> None:
        """Size the layer from the checkpoint's own dials.

        Args:
            text_config: carries ``hc_mult``, ``hc_sinkhorn_iters`` and
                ``hc_eps`` (``config.py:225-227``).
            neuron_config: framework overrides. ``mhc_sinkhorn_iters`` and
                ``mhc_eps`` (``neuron_config.py:194,197``) win when not
                ``None``, which is the override contract ``-013``'s section note
                for this class already stated.
            post_mult_value: the base's ``hc_post_mult_value``. **Its default is
                the base's own test value**, ``hc_post_alpha = 1.0``
                (``tests/kernels/test_mhc_kernels.py:126``); no fork config
                field carries it, so it is a constructor argument rather than an
                invented config default.

        TWO EPSILONS FOR THE BASE'S THREE, GROUNDED ON THE CHECKPOINT. The
        base's signature takes three (``rms_eps``, ``hc_pre_eps``,
        ``hc_sinkhorn_eps``) and the fork's config carries two fields. The
        split follows what the checkpoint sets, not what the base's test sets:

        * ``hc_pre_eps`` and ``hc_sinkhorn_eps`` are mHC-native, and the
          checkpoint's own ``text_config.hc_eps`` is ``1e-06``, so both keep
          ``hc_eps`` and the base's collapse onto one value is faithful for
          them. That is the value ``inc-glm53f-030`` measured its tiny case on,
          and nothing it recorded moves.
        * ``rms_eps`` is an RMSNorm epsilon, and the checkpoint's RMSNorm
          epsilon is ``1e-05`` -- a different number. It lives on
          ``Glm5NextTextConfig.rms_norm_eps`` (``inc-glm53f-080``) and reaches
          the router seam through :meth:`Glm5NextRoutedExperts.route_tokens`.
          It reaches no mHC line: ``self.hc_eps`` and the three sites that
          consume it below are unchanged.

        WHAT THIS CORRECTS. The earlier wording argued the single field was
        faithful for all three uses, and grounded that on the base's own kernel
        test setting ``hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6``
        (``tests/kernels/test_mhc_kernels.py:121``). That is the base's number,
        not the target's. Two thirds of the claim stand on the checkpoint's own
        ``hc_eps``; the RMSNorm third is settled against the checkpoint
        instead, which is where it always belonged.

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
        self.post_mult_value = float(post_mult_value)

        # ``hc_mult3`` is the base's own name for the projection's output width:
        # ``hc_mult`` pre weights + ``hc_mult`` post weights + ``hc_mult ** 2``
        # mixing weights, in that order, which is the order the three heads are
        # sliced out of ``mixes`` below.
        self.hc_mult3 = 2 * hc_mult + hc_mult * hc_mult

        # ORDINARY PARAMETERS, not ``_declare_parameters``' reservations, and
        # that is the lead's ruling for this section: the declared acceptance is
        # a synthetic case whose test SETS these tensors. The three names are
        # the base's own ``mhc_pre`` argument names, taken rather than chosen,
        # exactly as ``-029`` took its seam's signature -- they are THIS CLASS'S
        # OWN SIGNATURE AND NOT MAP NAMES.
        #
        # RE-GROUNDED BY ``inc-glm53f-082``. This note used to add "the weight
        # map declares this family absent, so there is no map name to reserve",
        # which ``inc-glm53f-078`` falsified: the map now emits six mHC names
        # per layer (``MHC_LEAVES``). Those six are reserved FLAT ON THE LAYER
        # by ``Glm5NextKDALayer`` and ``Glm5NextDSALayer``, not here, because
        # this class is not bound into a layer anywhere in this tree. The three
        # ``nn.Parameter`` names below are unchanged.
        self.fn = nn.Parameter(
            torch.zeros(self.hc_mult3, hc_mult * hidden, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(self.hc_mult3, dtype=torch.float32))

    # ── mHC pre -- the folded input, and ONE Sinkhorn dispatch ────────────
    def mhc_pre(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The base's ``mhc_pre``, with its Sinkhorn on ``-028``'s kernel.

        Args:
            residual: ``[T, S, H]`` -- the ``S = hc_mult`` residual streams.

        Returns:
            ``(post_mix, comb_mix, layer_input)`` -- ``[T, S, 1]``,
            ``[T, S, S]`` and ``[T, H]``, all fp32. ``comb_mix[t, i, j]``
            weights input stream ``i`` into output stream ``j``, the base's
            convention and the one ``-029``'s kernel reads.

        THE TOKEN CEILING, MEASURED AND DELIBERATELY NOT WORKED AROUND. The
        block-diagonal embedding puts ``T * S`` on the Sinkhorn's ``M``, and
        ``-028`` refuses ``M > PARTITION_MAX``. So this layer serves
        ``T <= PARTITION_MAX // S`` -- **32** at the checkpoint's ``hc_mult 4``
        -- where ``-029``'s combine kernel on its own would serve ``T <= 128``.
        The Sinkhorn therefore binds first. Nothing here pads, tiles or falls
        back: the seam's ``SinkhornError`` propagates to the caller unchanged,
        which is what the lead ruled and what P13 requires. The reading is in
        ``probe-030-m129.out``.

        Raises:
            Glm5NextHyperConnectionError: on a non-3-D ``residual`` or a stream
                or hidden extent that contradicts this layer's configuration.
            SinkhornError: from the seam, on a token count this layer's
                embedding puts above ``-028``'s ``M`` bound. Propagated, never
                caught.
        """
        from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise

        tokens, streams, hidden = self._require_streams(residual)

        flat = residual.reshape(tokens, streams * hidden).to(torch.float32)
        mixes = flat @ self.fn.to(torch.float32).t()
        # The RMS scale. Both upstream spellings divide the squared sum by the
        # projection's own input width -- ``hc_mult * hidden_size`` in
        # ``mhc_pre_torch``, ``fn.shape[-1]`` in ``mhc_pre_ref`` -- and those are
        # the same number.
        sqrsum = flat.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / float(streams * hidden) + self.hc_eps)

        scale = self.hc_scale.to(torch.float32)
        base = self.hc_base.to(torch.float32)
        pre_mix = (
            torch.sigmoid(mixes[:, :streams] * scale[0] + base[:streams])
            + self.hc_eps
        )
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
        # ``softmax`` and the ``+ eps`` are the base's, and they sit OUTSIDE the
        # seam because ``-028``'s kernel starts from an affinity matrix. This is
        # elementwise glue, which P13 leaves to torch.
        comb_start = torch.softmax(comb_logits, dim=-1) + self.hc_eps

        # ---- ENTRY 1 of 1 into ``-028``'s Sinkhorn seam. ----------------- #
        # The counted dispatch. One call for all ``T`` tokens, which is what the
        # block-diagonal embedding buys and what the route predicate declares.
        normalised = sinkhorn_normalise(
            torch.block_diag(*comb_start.unbind(0)), iters=self.sinkhorn_iters
        )
        comb_mix = self._diagonal_blocks(normalised, tokens, streams)

        layer_input = (pre_mix.unsqueeze(-1) * residual.to(torch.float32)).sum(dim=1)
        return post_mix.reshape(tokens, streams, 1), comb_mix, layer_input

    # ── mHC post -- ONE combine dispatch ──────────────────────────────────
    def mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """The base's ``mhc_post``, on ``-029``'s kernel.

        Args:
            x: ``[T, H]`` -- the sub-block's single-stream output.
            residual: ``[T, S, H]`` -- the streams, unchanged from
                :meth:`mhc_pre`'s input.
            post_layer_mix: ``[T, S, 1]`` from :meth:`mhc_pre`.
            comb_res_mix: ``[T, S, S]`` from :meth:`mhc_pre`.

        Returns:
            ``[T, S, H]`` fp32 -- the seam's own return dtype, not re-cast.

        Raises:
            HyperConnectionError: from the seam, on any inadmissible rank or
                extent. Propagated, never caught: a geometry the kernel cannot
                serve must not quietly reach a torch path (P13, D6).
        """
        from vllm_neuron.functional.mhc.hyper_connection import (
            hyper_connection_combine,
        )

        # ---- ENTRY 1 of 1 into ``-029``'s combine seam. ------------------ #
        # Argument names and order are the seam's, which are the base's, so this
        # is a call rather than a translation -- ``hyper_connection.py:375-376``
        # asks for exactly that.
        return hyper_connection_combine(
            x=x.to(torch.float32),
            residual=residual.to(torch.float32),
            post_layer_mix=post_layer_mix.to(torch.float32),
            comb_res_mix=comb_res_mix.to(torch.float32),
        )

    # ── one layer call ────────────────────────────────────────────────────
    def forward(self, residual: torch.Tensor, sublayer: object) -> torch.Tensor:
        """One mHC layer call: pre, then the sub-block, then post.

        This is the shape the decoder sections call, and it is why "per layer
        call" is a well-defined read window for the two counters: one entry here
        is exactly one Sinkhorn dispatch and one combine dispatch.

        Args:
            residual: ``[T, S, H]`` residual streams.
            sublayer: the wrapped sub-block, a callable ``[T, H] -> [T, H]``.
                Annotated ``object`` rather than ``Callable`` on purpose -- the
                module-level import block is ``-013``'s D14 section, and ``-023``
                set the precedent that a later section adds nothing to it, so
                every import in this section is function-local.

        Returns:
            ``[T, S, H]`` fp32 -- the re-mixed streams.

        Raises:
            Glm5NextHyperConnectionError: if ``sublayer`` is not callable, or if
                its output does not have the ``[T, H]`` shape the combine needs.
        """
        if not callable(sublayer):
            raise Glm5NextHyperConnectionError(
                f"sublayer must be a callable [T, H] -> [T, H], got "
                f"{type(sublayer).__name__}"
            )
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

    # ── layout helpers -- no arithmetic, and that is the point ────────────
    def _require_streams(self, residual: torch.Tensor) -> tuple[int, int, int]:
        """``(T, S, H)``, once the cross-argument agreements hold.

        Only what the seams cannot see: they read their extents off the tensors
        they are handed, so neither can tell that a stream or hidden extent
        disagrees with the CONFIG this layer was built from -- and a wrong
        stream count would silently mis-slice ``mixes`` instead of failing.
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

    @staticmethod
    def _diagonal_blocks(
        normalised: torch.Tensor, tokens: int, streams: int
    ) -> torch.Tensor:
        """Read the ``T`` per-token blocks back off a ``[T*S, T*S]`` diagonal.

        The inverse of :func:`torch.block_diag` for equal-sized blocks. Pure
        indexing: it selects, and computes nothing.
        """
        index = torch.arange(tokens, device=normalised.device)
        return normalised.reshape(tokens, streams, tokens, streams)[
            index, :, index, :
        ]


# ---------------------------------------------------------------------------
# ``Glm5NextMoEBlock`` and its two expert containers. D14 owners:
# ``inc-glm53f-031`` (expert partitioning), ``-032`` (router call site),
# ``-027`` (block-quant kernel call site), ``-033`` (shared-expert path) --
# all M2 Lane B, serialized because they share one class.
# ---------------------------------------------------------------------------


class Glm5NextRoutedExperts(nn.Module):
    """The routed-expert bank at ``mlp.experts``.

    One parameter per projection covers **all** ``n_routed_experts`` experts:
    the landed map sends ``mlp.experts.<leaf>_weight`` to a *list* of
    ``n_routed_experts`` checkpoint keys
    (``weight_loaders_fp8.py:670-679``), because this checkpoint stores one
    tensor per expert while the fork's parameter side is per-projection. The
    router lives here too, and carries a bias because
    ``topk_method == "noaux_tc"`` (``weight_loaders_fp8.py:654-655``).
    """

    # ── expert partitioning -- D14 owner: ``inc-glm53f-031`` ─────────────
    #
    # WHAT THIS SECTION CLOSES. At the unmodified parent ``031535b`` this bank
    # reported ``num_routed_experts == 288`` and nothing else: it carried no
    # per-rank expert count, no local expert index set, and no shard plan
    # (measured -- ``num_local_experts``, ``local_expert_indices`` and
    # ``expert_partition`` were all absent). So an assertion written against
    # ``num_routed_experts`` would have passed at the parent and certified
    # nothing; the partition below is what this increment actually authors.
    #
    # WHY THE ARITHMETIC IS IMPORTED RATHER THAN WRITTEN HERE. It lives in
    # ``factory.py``, which nothing on the arch-lookup path pulls the modeling
    # module into. The import is FUNCTION-LOCAL, following the landed
    # ``inc-glm53f-023`` precedent in this file and for the same two reasons:
    # this file's module import block is ``-013``'s D14 section, and the
    # file family's own idiom is a local import at the consuming member.
    #
    # THE 288/64 REFUSAL IS GONE, AND IT WAS NEVER THE FORK'S RULE --
    # ``inc-glm53f-087``. This paragraph used to say the refusal at the registered
    # tensor-parallel degree freeze of 64 was deliberate and visible, and that it
    # was campaign gap G4 surfaced where the model is built. That was wrong on the
    # only point that mattered: an expert bank divides by the EXPERT-PARALLEL
    # degree, never by the tensor-parallel one. The fork's one landed EP-aware
    # bank divides ``num_local_experts // self.ep_degree``
    # (``gpt_oss/model_bf16.py:1072``) and shards the INTERMEDIATE dimension by TP
    # instead (``:986-988``). So with expert parallelism off -- this campaign's
    # route -- the degree is 1, all 288 experts are local on every rank and
    # NOTHING RAISES at TP = 64. The raggedness gate is kept, and its subject is
    # now the expert-parallel degree, which is the one thing the old code got
    # right: a named error rather than ``gpt_oss``'s silent floor division.

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
        # ``_resolve_world_size()`` is ``-013``'s helper, called rather than
        # edited: an explicit ``world_size`` is what a caller with a degree
        # supplies, and ``None`` means "read the process group", which is 1 when
        # this stack is built undistributed.
        #
        # ``tp_degree`` KEEPS ITS NAME AND ITS MEANING and simply stopped being
        # the divisor (``inc-glm53f-087``): it is the tensor-parallel world size,
        # which is what shards the intermediate dimension.
        self.tp_degree = (
            _resolve_world_size() if world_size is None else int(world_size)
        )
        # THE DIVISOR. ``ep_degree`` is a trailing optional addition in the same
        # shape ``world_size`` already used, so no landed call site moves.
        self.ep_degree = _resolve_ep_degree(ep_degree)
        self.expert_partition = require_uniform_expert_partition(
            self.num_routed_experts, self.ep_degree
        )
        # Uniform by the gate above, so rank 0's count is every rank's count.
        self.num_local_experts = self.expert_partition.counts[0]
        # THE CHECKPOINT'S SwiGLU BOUND -- hand-off item (v), ruled in scope for
        # ``inc-glm53f-054a`` at DECISIONS 409.
        #
        # The reference clamps the ROUTED bank, not only the MLP: its
        # ``Glm5NextTextExperts`` stores this bound at
        # ``modeling_glm5_next.py:118`` and clamps ``gate`` from above and ``up``
        # on both sides at ``:139-140``, BEFORE ``F.silu(gate) * up`` at ``:142``.
        # The bank's own compute path had no route to the value at all -- not one
        # line of this class mentioned it -- so the kernel ran an UNCLAMPED SwiGLU
        # and computed a different function from the reference's on every token
        # whose projection left the box. On the published checkpoint that is a
        # bound of ``10.0`` and 42 MoE layers, with no shape moving and nothing
        # raising: the routed half of
        # ``B22-M1-shared-expert-swiglu-clamp-omitted``, whose SHARED half
        # ``inc-glm53f-033`` repaired. Measured in
        # ``../../../artifacts/campaigns/glm-5.3-flash-port/increments/probe-054a-swiglu-clamp-r2.out``.
        #
        # IT REFUSES RATHER THAN DEFAULTING. A literal here would be a bound this
        # code invented, and a ``None`` reaching the kernel is exactly the silent
        # omission being closed -- all four of its limit parameters default to
        # ``None``. The read is at construction, like the shared expert's
        # (``Glm5NextSharedExperts.__init__``, named rather than numbered because
        # this file cites in-file lines by number and every insertion shifts
        # every number below it), so no call site can hand this path a bound the
        # checkpoint never declared.
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

        Delegates to the plan rather than recomputing the arithmetic, so there
        is one partition and not two that can disagree.
        """
        return self.expert_partition.local_expert_indices(rank)

    # ── router call site -- D14 owner: ``inc-glm53f-032`` ────────────────
    #
    # SCOPE. This section adds ONE method and edits no line above it.
    # ``__init__`` and ``local_expert_indices`` are ``inc-glm53f-031``'s landed
    # code and ``forward`` is ``inc-glm53f-013``'s stub; all three stay
    # byte-identical, so nothing ``-031``'s recorded acceptance asserts can move.
    #
    # WHY THE CONFIG IS AN ARGUMENT RATHER THAN STATE. The routing
    # hyperparameters live on ``Glm5NextTextConfig``, and ``-031``'s ``__init__``
    # does not retain it. Adding a field would edit that landed ``__init__``, so
    # the config is threaded in at the call instead -- the smaller change, and
    # the one D14's section-ownership rule permits.
    #
    # WHAT THIS SITE DOES NOT DO. It returns GLOBAL expert indices over all
    # ``n_routed_experts``, exactly as the checkpoint's router does. Mapping
    # those onto this rank's ``expert_partition`` slice is the DISPATCH step,
    # and it belongs to the block-quant call site (``-027``) and the
    # shared-expert path (``-033``). Doing it here would put two owners on one
    # behaviour.
    #
    # ``rms_norm_eps`` IS THE CONFIG'S, MEASURED NOT ASSUMED. The earlier note
    # here read the key as absent. Both halves of that reading were true when
    # taken -- neither ``Glm5NextTextConfig`` nor the pinned
    # ``fixtures/config.json`` carried an ``rms_norm_eps`` -- but the conclusion
    # drawn from them, that the substrate's own ``eps=1e-6`` was the only value
    # available, is refuted by the checkpoint: its ``text_config`` carries
    # ``rms_norm_eps = 1e-05``. ``inc-glm53f-080`` adds the field and
    # re-transcribes the fixture, so this method resolves the epsilon from the
    # config it already receives and hands THAT number to the seam. ``eps``
    # stays a parameter so a caller can still override it; a caller that passes
    # nothing now gets the checkpoint's value rather than the kernel's.

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
                ``text_config.rms_norm_eps`` -- the checkpoint's ``1e-05``.
                Pass a float to override it. See the section note above.

        Returns:
            ``(router_logits [T, E], expert_index [T, k] int32,
            expert_affinities [T, E] float32)``. ``expert_affinities`` is the
            scattered form the downstream MoE consumes: the gate weight at each
            selected expert's column, zero elsewhere.

        The seam is entered with ``correction_bias=self.router_bias``, and that
        parameter is ``mlp.gate.e_score_correction_bias``
        (``weight_loaders_fp8.py:662-665``) -- the ``noaux_tc`` correction bias,
        NOT a router projection bias. The seam's own signature keeps the two
        apart by name, because adding this tensor to the logits instead of to
        the sigmoid scores would compute a different router that no shape check
        could catch.
        """
        # The config's epsilon is the operative one unless a caller overrides
        # it. Resolved here rather than as a signature default, because a
        # signature default hard-wires one number into the code and this one
        # belongs to the checkpoint.
        if eps is None:
            eps = float(text_config.rms_norm_eps)

        # Function-local import, following the landed ``inc-glm53f-023`` and
        # ``-031`` precedent in this file: this file's module import block is
        # ``-013``'s D14 section.
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

    # ── block-quant kernel call site -- D14 owner: ``inc-glm53f-027`` ─────
    #
    # SCOPE. This section adds ONE method and edits no line above or below it.
    # ``__init__`` and ``local_expert_indices`` are ``inc-glm53f-031``'s landed
    # code, ``route_tokens`` is ``inc-glm53f-032``'s, and ``forward`` is
    # ``inc-glm53f-013``'s stub; all four stay byte-identical, so nothing any
    # of those three recorded acceptances asserts can move. The method sits on
    # ``Glm5NextRoutedExperts`` rather than on ``Glm5NextMoEBlock`` under D14's
    # sub-class rule -- the same reading ``-032`` landed for the sibling
    # router-call-site row, and for the same reason: the routed bank is where
    # the expert weights and the partition live.
    #
    # D5(b): THE INNER KERNEL IS CALLED DIRECTLY. The public ``moe_cte``
    # dispatcher will not forward block scales, so this site enters
    # ``inc-glm53f-025``'s ``blockwise_fp8_moe`` seam instead of the dispatcher.
    #
    # NO QUANTISATION ENUM MEMBER IS NAMED OR ADDED (plan section 11 constraint
    # B.6, and the ``-023`` section above already declares the same negative).
    # The route is selected by WHICH FUNCTION IS CALLED, and the not-block-quant
    # branch RAISES BY NAME rather than falling through to the substrate's
    # ``QuantizationType.NONE`` default (``functional/mlp.py:81``, ``:249`` --
    # a default reached by OMISSION, so a call site that forgot to route gets it
    # with no error at all). The acceptance's dispatch-counter clause exists to
    # exclude exactly that silent fallback; the raise makes it IMPOSSIBLE rather
    # than merely detectable.
    #
    # WHY THE WEIGHTS, SCALES AND CONFIG ARE ARGUMENTS. ``-031``'s ``__init__``
    # declares ``gate_proj_weight`` / ``up_proj_weight`` / ``down_proj_weight``
    # by ``register_parameter(name, None)`` and NO scale parameter, and it
    # retains no config. So there is no landed layout to read: the fused
    # ``[E, H, 2, I_TP]`` gate/up tensor the kernel needs is not the
    # checkpoint's per-projection storage, and turning one into the other is the
    # weight loader's step, not this section's. Adding fields to that landed
    # ``__init__`` would edit ``-031``'s code, which D14's section-ownership
    # rule does not permit here, so everything this site consumes is threaded in
    # at the call -- the smaller change, and ``-032``'s own precedent for the
    # config argument. What IS read off ``self`` is the partition
    # (``num_local_experts``, ``num_experts_per_tok``), which is what ``-031``
    # landed this bank to carry.
    #
    # WHAT THIS SITE DOES NOT DO, deliberately: it does not touch the shared
    # expert (``-033``), it authors no numerics of its own (``-025``'s kernel and
    # ``-024``'s retile producer own those), it does not fuse or transpose
    # checkpoint tensors, and it does not assemble the layer forward
    # (``-013`` / ``-054``). It reuses ``build_blockwise_mapping`` (F9) for the
    # token-block mapping rather than re-deriving one.
    #
    # Imports are FUNCTION-LOCAL, following the landed ``-023``, ``-031`` and
    # ``-032`` precedent in this file: the module import block is ``-013``'s D14
    # section. The two ``to_kernel_scale_layout`` helpers in this campaign have
    # THE SAME NAME AND DIFFERENT SIGNATURES
    # (``functional/blockwise_fp8_mm.py:309`` takes ``(weight_scale, rows,
    # cols)``; ``functional/moe/moe_blockwise_fp8.py:170`` takes
    # ``(consumer_scales, num_experts, rows, cols, projection)``), so the one
    # this site needs is imported from its own module UNDER AN ALIAS that names
    # the module -- an unqualified import of both would be an arity failure that
    # reads as a shape bug.

    def block_quant_expert_mm(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        gate_up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_up_consumer_scales: torch.Tensor,
        down_consumer_scales: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
        *,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int = 0,
    ) -> torch.Tensor:
        """Run this rank's routed experts through the block-quant NKI kernel.

        Args:
            hidden_states: ``[T, H]`` real tokens only -- the kernel's
                padding-token slot is appended here, not by the caller
                (``bwmm_shard_on_I.py:157``).
            expert_affinities: ``[T, E]`` scattered router scores over ALL
                ``n_routed_experts``, which is the form :meth:`route_tokens`
                returns: the gate weight at each selected expert's column and
                zero elsewhere. THIS SITE DOES THE GLOBAL-TO-LOCAL MAPPING, so
                the caller hands the router's output straight through and slices
                nothing. ``inc-glm53f-032``'s landed note above assigns the step
                here in those words. Until repair batch R5 this parameter was
                documented as ``[T, E_local]`` and the extent check enforced
                that, which no caller could satisfy from ``route_tokens`` at any
                expert-parallel degree above 1 -- finding ``B21-027``.
            expert_parallel_rank: which rank's expert slice to select, read
                through :meth:`local_expert_indices`. It is an argument rather
                than an attribute because ``__init__`` is ``inc-glm53f-031``'s
                landed code and this section edits no line above itself. The
                default 0 is the only rank that exists at degree 1.
            gate_up_proj_weight: ``[E_local, H, 2, I_TP]`` fp8-e4m3, retiled
                onto ``BLOCK_QUANT_SIZE`` granularity.
            down_proj_weight: ``[E_local, I_TP, H]`` fp8-e4m3, retiled.
            gate_up_consumer_scales: the retile producer's FLAT emission for the
                fused gate/up bank, shape
                :func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.consumer_scale_shape`
                at ``projection=GATE_UP``. Both fusion halves must be present:
                the producer writes one half per call and leaves the other
                ``NaN``, and merging them is the loader's step.
            down_consumer_scales: the same, at ``projection=DOWN``.
            quant_config: the resolved per-model quantisation policy. This is
                the route selector; see the section note above.
            block_size: tokens per block, a multiple of ``BLOCK_QUANT_SIZE``.
                Defaults to ``BLOCK_QUANT_SIZE`` itself.
            moe_group: the MoE ``GroupCoordinator``, forwarded verbatim to
                ``build_blockwise_mapping``. It is UNREAD on both of that
                function's flows when ``tp_degree == 1``
                (``moe_blockwise.py:354`` and ``:398`` gate every use behind
                ``tp_degree > 1``), which is why an undistributed call site may
                leave it ``None``; a sharded one supplies the real group rather
                than having this site invent one.
            tp_degree: ranks sharding each expert's intermediate dimension.

        Returns:
            ``[T, H]`` -- the padding-token row is sliced off. The dtype is the
            seam's own (``bfloat16`` on the NKI route) and is not re-cast here:
            the layer forward decides the residual dtype and that is
            ``inc-glm53f-054``'s section.

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
            DOWN,
            GATE_UP,
            TILE_SIZE,
        )
        from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
            ExpertAffinityScaleMode,
            blockwise_fp8_moe,
        )
        from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
            to_kernel_scale_layout as moe_to_kernel_scale_layout,
        )

        # ---- ROUTE SELECTION. The certifying component (D1.4). ----------- #
        if not quant_config.is_block_quantized:
            raise Glm5NextBlockQuantRouteError(
                "block_quant_expert_mm is the D5(b) block-quant route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised expert path at this "
                "site, and silently continuing is what would reach the "
                "substrate's QuantizationType.NONE default by omission."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextBlockQuantRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes scales retiled from "
                f"({TILE_SIZE}, {TILE_SIZE}) checkpoint blocks onto "
                f"{BLOCK_QUANT_SIZE} granularity and has no path for any other "
                f"checkpoint block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        if hidden_states.dim() != 2:
            raise Glm5NextBlockQuantRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        tokens, hidden = (int(extent) for extent in hidden_states.shape)
        if gate_up_proj_weight.dim() != 4 or gate_up_proj_weight.shape[2] != 2:
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight must be [E, H, 2, I_TP], got shape "
                f"{tuple(gate_up_proj_weight.shape)}"
            )
        num_experts = int(gate_up_proj_weight.shape[0])
        intermediate = int(gate_up_proj_weight.shape[-1])
        if num_experts != int(self.num_local_experts):
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight carries {num_experts} experts but this "
                f"rank owns {self.num_local_experts}; the bank and the "
                f"partition must agree"
            )
        if int(gate_up_proj_weight.shape[1]) != hidden:
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight has H={int(gate_up_proj_weight.shape[1])} "
                f"but hidden_states has H={hidden}"
            )
        if tuple(down_proj_weight.shape) != (num_experts, intermediate, hidden):
            raise Glm5NextBlockQuantRouteError(
                f"down_proj_weight must be [E={num_experts}, "
                f"I_TP={intermediate}, H={hidden}], got shape "
                f"{tuple(down_proj_weight.shape)}"
            )
        # ---- THE DISPATCH STEP. Global router columns -> this rank's slice.  #
        # ``inc-glm53f-032``'s landed note above says in its own words that
        # mapping global expert indices onto this rank's ``expert_partition``
        # slice "belongs to the block-quant call site (``-027``)". Until repair
        # batch R5 this site did the opposite: it REFUSED the global form, so
        # nothing could feed it from ``route_tokens`` above degree 1 and the MoE
        # path could not run on more than one rank. Finding ``B21-027``.
        #
        # The fork's own MoE call site is the precedent, including the guard:
        # ``gpt_oss/model_mxfp4.py:1506-1516`` maps only when the degree is
        # above 1 and passes the affinities through untouched at degree 1.
        routed = int(self.num_routed_experts)
        if tuple(expert_affinities.shape) != (tokens, routed):
            raise Glm5NextBlockQuantRouteError(
                f"expert_affinities must be [T={tokens}, E={routed}] -- the "
                f"GLOBAL router width, which is what route_tokens returns and "
                f"what this site maps onto this rank's {num_experts} experts -- "
                f"got shape {tuple(expert_affinities.shape)}"
            )
        if routed != num_experts:
            local_indices = torch.tensor(
                self.local_expert_indices(int(expert_parallel_rank)),
                dtype=torch.int64,
                device=expert_affinities.device,
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
                f"(bwmm_shard_on_I.py:667)"
            )

        # ---- The token-block mapping, over the REAL token count. --------- #
        # REUSED (F9), not authored.
        #
        # THE ORDER HERE IS LOAD-BEARING, and until repair batch R5 round 2 it
        # was the wrong way round. The kernel needs a padding-token slot, and
        # this site used to append it to the affinities BEFORE building the
        # mapping. That made the mapping see ``T + 1`` tokens, which is odd for
        # every even real token count, and the mapping's two NKI subkernels
        # need ``chunk_size % 128 == 0`` and ``total_tokens % f_len == 0``
        # (``moe_blockwise.py:520, 536``). Both fail on an odd count, so the
        # mapping fell to ``_build_blockwise_mapping_torch`` on EVERY call at
        # EVERY shape the plan declares -- a silent torch fallback for per-token
        # device work the fork already ships NKI subkernels for
        # (``moe_blockwise.py:10-11``). Finding ``B21-027``, P13.
        #
        # The mapping is now built over the real ``T`` and the padding slot is
        # appended afterwards. The fork's own MoE call site is the precedent:
        # ``gpt_oss/model_mxfp4.py:1518-1536`` never appends a row before the
        # mapping.
        #
        # ``conditions`` is deliberately unconsumed: it feeds the
        # ``*_hybrid`` dynamic-while variant of the vendor kernel, and D5(b)
        # calls the non-hybrid inner member, whose signature has no such
        # parameter. Named with a leading underscore rather than dropped so the
        # unused fourth return is visible instead of implied.
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

        # ---- The padding-token slot, appended AFTER the mapping. --------- #
        # The kernel reads a ``-1`` token position as the LAST row of
        # ``hidden_states`` (``bwmm_shard_on_I.py:157``), so the hidden tensor
        # grows one zero row and the result is sliced back at the end.
        #
        # The affinities grow the same slot in their FLAT form, because that is
        # what the mapping returns: ``expert_affinities_masked`` is
        # ``[T * E_local, 1]`` and the layout is token-major
        # (``moe_blockwise.py:103-105``, a ``view(-1, 1)`` of ``[T, E]``).
        # Appending ``E_local`` zero entries to the flat tensor is therefore the
        # same tensor as appending one zero ROW before the view -- measured
        # byte-identical, not assumed (``probe-R5r2-pad-order-equivalence.py``,
        # reading ``Q2``). Only this order keeps ``total_tokens`` even.
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

        # ---- The scale bridge. ``-025``'s helper, at its own arity. ------ #
        # The mapping is built from the fp32 affinities so the nonzero mask is
        # exact, and the masked tensor is cast to the activation dtype only
        # afterwards -- the kernel multiplies it into the hidden states.
        gate_up_proj_scale = moe_to_kernel_scale_layout(
            gate_up_consumer_scales,
            num_experts,
            hidden,
            intermediate,
            projection=GATE_UP,
        )
        down_proj_scale = moe_to_kernel_scale_layout(
            down_consumer_scales,
            num_experts,
            hidden,
            intermediate,
            projection=DOWN,
        )

        output = blockwise_fp8_moe(
            hidden_states=padded_hidden,
            expert_affinities_masked=expert_affinities_masked.to(
                hidden_states.dtype
            ),
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block,
            token_position_to_id=token_position_to_id,
            # ``[N]`` from the mapping, ``[N, 1]`` at the seam
            # (``moe_blockwise_fp8.py:323``).
            block_to_expert=block_to_expert.reshape(-1, 1),
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            # WHERE THE ROUTER WEIGHT MULTIPLIES -- ``inc-glm53f-054a``, review
            # finding recorded at DECISIONS 449.
            #
            # THIS LINE CHANGES THE MEANING OF LANDED ``-027``/R5 CODE. The call
            # below used to pass no scaling mode at all, so it inherited the
            # seam shim's default of ``PRE_SCALE``
            # (``moe_blockwise_fp8.py:337-339``), which multiplies the router
            # weight into the HIDDEN STATES before the gate and up projections.
            # The checkpoint's reference multiplies it after the DOWN projection
            # instead: ``modeling_glm5_next.py:132-133`` projects
            # ``hidden_states[token_idx]`` unscaled and then writes
            # ``F.linear(current, down_proj[e]) * top_k_weights[...]``. That is
            # ``POST_SCALE``, and it is a different function, not a rearranged
            # one -- this repository's own torch implementation says so in its
            # own words at ``vllm_neuron/functional/moe/moe_cte.py:490-492``
            # (the DIRECTORY matters: ``nkilib`` ships a ``moe_cte.py`` of its
            # own, whose lines there are different): "this is NOT mathematically
            # equivalent to POST_SCALE because the nonlinear activation breaks
            # the linearity: act(a * x) != a * act(x)".
            #
            # AND IT WAS NOT A SMALL DIFFERENCE. The pinned config normalises the
            # top-8 gate weights and scales them by 2.5, so the eight affinities
            # sum to 2.5 and average 0.3125. That is a statement about the MEAN
            # and about nothing else: the router divides by its own sum and then
            # multiplies by the scaling factor
            # (``modeling_glm5_next.py:179-182``), so a single weight lies
            # anywhere in ``(0, 2.5)`` and DOES exceed 1 when one expert
            # dominates a token. Two consequences, measured in
            # ``increments/probe-054a-affinity-scaling-mode-r1.out`` under the
            # campaign artifacts root:
            # the per-token contribution was wrong by up to 76.9% against a 1%
            # acceptance tolerance, and the SwiGLU bound three paragraphs below
            # bound ``affinity * gate`` rather than ``gate``, so it bit above
            # 32.0 instead of the checkpoint's 10.0. That second one is the
            # routed half of ``B22-M1-shared-expert-swiglu-clamp-omitted``
            # reopened one layer underneath the repair: on every token whose
            # projection landed between 10 and 32 the kernel ran the SwiGLU
            # UNCLAMPED where the reference clamps it. Naming the mode here is
            # what closes that half.
            #
            # EXPLICIT RATHER THAN INHERITED, WHICH IS THE FORK'S OWN
            # CONVENTION. Every one of the five landed MoE call sites in this
            # repository names this parameter and chooses ``POST_SCALE`` -- the
            # SAME value this call chooses, and the opposite of the shim default
            # they were all declining to inherit. All five sit on the gpt_oss
            # expert bank: ``GptOssExperts._run_moe_block_tkg``
            # (``:1339`` quantised, ``:1257`` bf16),
            # ``GptOssExperts._run_moe_tkg`` (``:1390``) and
            # ``GptOssExperts.forward_prefill`` (``:1567`` quantised, ``:1409``
            # bf16). A default carried silently is what let a whole-function
            # divergence land with no shape moving and nothing raising.
            #
            # THE ENUM COMES FROM THE SEAM MODULE, NOT FROM ``nkilib``
            # DIRECTLY, and that is deliberate. ``nkilib`` exports this name
            # from two paths -- ``core.moe.moe_cte.moe_cte``, which
            # ``moe_blockwise_fp8.py:83-87`` imports, and
            # ``core.utils.common_types``, which the gpt_oss quantised model's
            # module-level import block imports at its ``:60-66``
            # (``ExpertAffinityScaleMode`` on ``:62``) -- and the consuming code
            # compares members with ``==``
            # (``vllm_neuron/functional/moe/moe_cte.py:537``, ``:570``,
            # ``:595``, and the kernel this call actually reaches, at
            # ``bwmm_shard_on_I.py:1157`` and ``:1176``). If those two paths ever
            # resolve to distinct enum classes, an equality against the wrong one
            # is False on every branch and the router weight is dropped
            # ENTIRELY rather than misplaced. Importing from the seam this call
            # enters means the member compared is the member that module holds,
            # so the question cannot arise.
            #
            # THE TWO PATHS ARE IN FACT ONE OBJECT, read live on the installed
            # ``nkilib`` under lease grant 037
            # (``increments/record-054a-nkilib-probe-037.md`` §5 under the campaign
            # artifacts root):
            # ``ENUM_SAME_OBJECT=True``, with both paths reporting
            # ``nkilib.core.utils.common_types`` as the defining module and the
            # class carrying four members, not two. The import above is kept as
            # it stands: it was arranged not to depend on the answer, and an
            # arrangement that survives either answer is still the right one.
            #
            # CONFIGURATION, NOT KERNEL-CLASS WORK. The mode is a parameter the
            # kernel already declares and both of the seam's routes already
            # forward -- ``blockwise_fp8_moe`` passes ``**kernel_kwargs``
            # verbatim to the NKI launch and to the torch oracle alike
            # (``moe_blockwise_fp8.py:414``, ``:429``, ``:455``, ``:473``). No
            # torch fallback is introduced and no new kernel is written, so P13
            # is not engaged.
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            # THE CHECKPOINT'S SwiGLU BOUND, hand-off item (v), DECISIONS 409.
            #
            # THE ASYMMETRY IS THE REFERENCE'S. ``gate`` is bounded from ABOVE
            # only (``modeling_glm5_next.py:139`` passes ``min=None``) and ``up``
            # on BOTH sides (``:140``). Making both two-sided would be a second
            # wrong function rather than a tidier one, which is the reasoning
            # ``Glm5NextSharedExperts.shared_expert_mm``'s own transliteration
            # note records for the shared half of the same defect.
            #
            # THE CLAMP IS THE KERNEL'S ALREADY, so this is configuration and not
            # new kernel-class work (P13 is not engaged and no torch fallback is
            # introduced). The four parameters are declared at
            # ``moe_blockwise_fp8.py:340-343``, on the traced shim
            # ``_torch_compatible_blockwise_mm_baseline_shard_intermediate``
            # (``:314``), which forwards them to the nkilib kernel at ``:372``.
            # They travel there as ``blockwise_fp8_moe``'s ``**kernel_kwargs``,
            # which that seam forwards VERBATIM on BOTH of its routes: the NKI
            # launch at ``:462`` and the torch oracle at ``:445``, whose own
            # forward into the vendor's torch reference is at ``:519``.
            #
            # THE CPU LANE EXERCISES THE KERNEL'S OWN CLAMP, not a torch copy of
            # it. Under ``VLLM_NEURON_CPU_MODE=1`` with ``NKI_SIMULATOR=1``, this
            # campaign's CPU-lane environment, ``can_run_kernel`` returns True
            # (``utils/neuron_utils.py:20-21``), so the seam takes the NKI route
            # and these four keywords land on the shim that declares them by
            # name. The torch-oracle route is reached only when there is no
            # simulator or NKI is disabled.
            #
            # AND THIS IS THE REPOSITORY'S FIRST CALLER to send a clamp limit
            # through this seam. No other call site and no test passes one, so
            # nothing landed covers the keywords' onward journey. That is what
            # makes the routed item's FAIL/PASS pair an instrument rather than
            # ceremony: with these four keywords removed the item must go RED,
            # and a keyword some route accepted and then ignored would leave it
            # green.
            #
            # WHY EXPLICIT ``None``. The lower bound on ``gate`` is a positive
            # declaration that the reference has none there, in a call whose four
            # limits all default to ``None`` -- the default that made the omission
            # silent for every increment before this one.
            gate_clamp_upper_limit=self.swiglu_limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.swiglu_limit,
            up_clamp_lower_limit=-self.swiglu_limit,
            # ---- EVERY REMAINING KERNEL PARAMETER, STATED -- DECISIONS 459/464 ----
            #
            # WHY EACH ONE IS WRITTEN even where its value equals the kernel's own
            # default. A default is only safe to inherit if there is one default.
            # Grant 037 read the installed kernel and found that is not the case
            # here: the wrapper this call enters defaults
            # ``expert_affinities_scaling_mode`` to ``PRE_SCALE``
            # (``moe_blockwise_fp8.py:337-339``), which is the WRONG scaling point
            # for this checkpoint, and the vendor's own two torch layers disagree
            # with each other on the same parameter -- ``PRE_SCALE`` at
            # ``bwmm_shard_on_I_torch.py:55`` against ``POST_SCALE`` at
            # ``moe_cte_torch.py:49``. This seam's oracle already records that
            # divergence in its own docstring (``moe_blockwise_fp8.py:500-503``).
            # A value two vendor layers read differently is a value the caller
            # must state, so every parameter this bank's arithmetic depends on is
            # named here with the reason the MODEL requires it.
            #
            # THE REFERENCE COMPUTES BOTH PROJECTIONS AND MULTIPLIES THEM:
            # ``modeling_glm5_next.py:142`` is ``return F.silu(gate) * up``, over
            # the two halves ``:138`` chunks out of the fused bank. Skipping the
            # gate projection would drop the ``silu`` factor altogether.
            skip_gate_proj=False,
            # THE AFFINITY MULTIPLIES THE OUTPUT, NOT THE INTERMEDIATE. This is
            # the same line that fixes ``POST_SCALE`` above -- at
            # ``modeling_glm5_next.py:133`` the router weight multiplies the
            # ``[tokens, H]`` result of the down projection, so it is not applied
            # on the ``I`` dimension. One reading of one line settles both
            # parameters, which is why they cite the same place.
            expert_affinity_multiply_on_I=False,
            # NO ACTIVATION SCALE IS HANDED OVER -- DECISIONS 464 (1).
            #
            # This checkpoint quantises WEIGHTS per block and declares
            # ``"activation_scheme": "dynamic"``
            # (``test/vllm_neuron/model/glm5_next/fixtures/hf-config.json``;
            # modelled at ``config.py:439`` and admitted only as ``"dynamic"`` at
            # ``quantization.py:116``), so there is no static per-token activation
            # scale in the checkpoint to hand over. The kernel consumes these two
            # only under ``ActivationQuantMode.PER_TOKEN`` or ``PER_TENSOR``
            # (``bwmm_shard_on_I.py:1033``, ``:1055``, ``:1084``, ``:1086``) --
            # a mode that is NOT a caller parameter, being absent from the
            # kernel's signature -- and no site in this repository has ever passed
            # a non-``None`` one (``moe_blockwise_fp8.py:326-327`` declares them,
            # ``:384-385`` forwards them, and those are the only sites).
            #
            # THIS IS AN OPEN READING, NOT A SETTLED NUMERICS CLAIM. Under a
            # dynamic scheme something has to compute the per-token scale, and
            # whether the kernel does it is unread -- the branch at
            # ``bwmm_shard_on_I.py:318`` needs a host grant to see. Carried as
            # debt ``D-054a-ACTQ``.
            gate_up_hidden_scale=None,
            down_hidden_scale=None,
            # EXECUTION STRATEGY, NOT SEMANTICS. These three are written at the
            # kernel's own defaults, read off the installed kernel under grant 037
            # (``increments/record-054a-nkilib-probe-037.md`` §7 rows ``:201``,
            # ``:202``, ``:207``): ``accumulation_dtype`` at
            # ``bwmm_shard_on_I.py:129``, ``checkpoint_activation`` at ``:127``,
            # ``is_tensor_update_accumulating`` at ``:121``. Stating them means a
            # vendor bump that moves a default cannot move this model's numbers
            # without moving this line too. ``accumulation_dtype=None`` is a
            # declaration that the kernel resolves the accumulation dtype itself,
            # not an omission.
            accumulation_dtype=None,
            checkpoint_activation=False,
            is_tensor_update_accumulating=True,
            # ``compute_dtype`` IS DELIBERATELY NOT A KEYWORD HERE -- DECISIONS
            # 464 (2), and it is cited rather than passed.
            #
            # The value the kernel needs is the checkpoint's own dtype,
            # ``bfloat16`` (``fixtures/hf-config.json``, ``text_config.dtype``),
            # and the kernel already receives exactly that: the wrapper defaults
            # ``compute_dtype`` to ``nl.bfloat16`` (``moe_blockwise_fp8.py:335``).
            # It is not written as a keyword because ONE ``kernel_kwargs`` mapping
            # feeds BOTH of this seam's routes verbatim -- ``:455`` to the torch
            # oracle, ``:473`` to the NKI launch -- so a single keyword cannot
            # carry the two different objects the two routes need:
            # ``nl.bfloat16`` would hand the oracle an NKI object,
            # ``torch.bfloat16`` would reach the kernel, and ``nl`` is not on this
            # module's imports at all. The oracle therefore keeps the vendor
            # default ``None`` (``bwmm_shard_on_I_torch.py:53``,
            # ``moe_cte_torch.py:47``): a KNOWN seam divergence, recorded rather
            # than discovered later. The per-route translation belongs to the
            # seam and is TO BE FILED at this increment's fold -- no such increment
            # exists on disk yet, and this comment does not claim one does. This
            # call's surface is
            # ``model_fp8.py`` alone. The oracle route is reached only when
            # ``can_run_blockwise_fp8_moe`` is False
            # (``moe_blockwise_fp8.py:439-445``), which the CPU lane does not
            # take, so no acceptance run depends on the divergence.
        )
        return output[:tokens]

    # ── load-time operand prep -- hand-off item (i) of ``inc-glm53f-054a`` ──
    #
    # WHAT THIS SECTION IS FOR. ``block_quant_expert_mm`` above takes the fused
    # ``[E_local, H, 2, I_TP]`` gate/up bank, the retiled ``[E_local, I_TP, H]``
    # down bank, and the two FLAT consumer scale emissions. The checkpoint
    # stores none of those: ``__init__`` declares three per-projection weights
    # and no scale parameter at all. This section builds all four, ONCE, at load
    # time, and holds them on the module.
    #
    # IT ENROLLS IN A LANDED LOOP AND EDITS NO LINE OF IT.
    # ``Glm5NextForConditionalGeneration._run_load_time_preps`` is the single
    # production caller of either prep and enrols a module by
    # ``hasattr(type(module), "prepare_scale_operands")``. It derives the operand
    # names from this module's OWN declaration tuple through
    # ``_scale_prep_leaves``, whose docstring names this bank as the reason it
    # exists: the bank declares ``router_weight`` and no
    # ``router_weight_scale_inv`` exists anywhere in this tree, so a
    # declaration-only derivation yielded four leaves and eight operands while
    # the presence-reading one yields the three leaves that have grids. The loop
    # therefore hands exactly the six operands below, by keyword.
    #
    # WHY ONE METHOD BUILDS THE WEIGHTS AS WELL, though it is named for scales.
    # The ``-024`` producer emits ``consumer_scales`` and ``retiled_weights``
    # from ONE pass over a bank. Splitting them across the loop's two hooks
    # would retile every bank twice and leave two answers to one question, and
    # the other hook needs ``projection_widths()``, which is the attention
    # section's 2-D contract rather than an expert bank's. The hook contract is
    # unchanged: same name, six keyword operands, an int return.

    #: Where :meth:`prepare_scale_operands` leaves its four operands. A class
    #: attribute for the reason the shared expert's own is one: the name is part
    #: of the contract between the builder and the reader, and neither should
    #: spell it twice.
    PREPARED_KERNEL_OPERANDS_ATTR = "_prepared_kernel_operands"

    #: Where the producer's three health counts are left, for the acceptance to
    #: read. Recorded rather than refused on -- see the method's Raises note.
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
        """Build this rank's four kernel operands ONCE. Returns how many.

        Hand-off item (i) of ``inc-glm53f-054a``. The argument list mirrors the
        shared expert's own -- same names, same order -- because both are called
        by the same loop from the same derivation, and a divergent order at one
        of them is a mis-wiring no shape check would see.

        Args:
            gate_proj_weight: ``[E_local, I_TP, H]`` fp8-e4m3, this rank's slice.
            up_proj_weight: ``[E_local, I_TP, H]`` fp8-e4m3.
            down_proj_weight: ``[E_local, H, I_TP]`` fp8-e4m3.
            gate_proj_scale: ``[E_local, I_TP//128, H//128]`` fp32, the
                checkpoint's own block grid.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[E_local, H//128, I_TP//128]`` fp32.

        Returns:
            How many operands were built -- ``4`` on every successful call:
            the fused gate/up bank, its merged consumer scales, the retiled down
            bank and its consumer scales.

        Raises:
            Glm5NextBlockQuantRouteError: on a missing operand, a weight that is
                not 3-D, an expert count that disagrees with this rank's
                partition, or a fusion merge that left a slot unwritten. Those
                four are structural. The producer's own health counts --
                ``emitted_unsupplied``, ``input_scales_dropped`` and
                ``inexact_rescales`` -- are RECORDED on this module instead of
                refused on, and the acceptance reads them: they are numeric
                properties of a particular checkpoint's scales, so a refusal
                here would turn a reportable measurement into a load failure on
                a case no test has run.

        THOSE ARE THE LOADER'S ORIENTATIONS AND NOT THE KERNEL'S, which is why
        the body below transposes two of the three banks on the way in and one on
        the way out. The registered parameter layout IS the checkpoint layout:
        ``weight_loaders_fp8.py`` passes ``is_storage_transposed=False`` and says
        in its own words that every consumer transposes at compute time instead,
        and each bank is that checkpoint's per-expert slices stacked on a new
        LEADING axis with no transpose (``_stack_local_expert_weights``,
        ``_stack_local_expert_scales``). The checkpoint stores one ``nn.Linear``
        weight per expert, ``[out, in]``, which the reference confirms at
        ``../../../artifacts/campaigns/glm-5.3-flash-port/design/reference/modeling_glm5_next.py:116-117``:
        gate and up are ``[I, H]``, down is ``[H, I]``.

        AN EARLIER REVISION OF THIS BLOCK NAMED THE KERNEL'S ORIENTATIONS HERE,
        and that was worse than merely wrong. It would have led a reader to
        transpose DOWN, whose consumer scale shape is the PRODUCT of the two block
        counts and so identical either way -- the wrong repair passes every shape
        check anything could write and mis-assigns every scale. The gate for this
        increment therefore measures which axis arrives as the producer's ``rows``,
        and plants that exact pair to show the check rejects it.

        THE MERGE IS CHECKED RATHER THAN ASSUMED. ``block_quant_expert_mm``
        requires both fusion halves present and says the producer writes one per
        call, leaving the other ``NaN``. The producer fills its emission with
        ``NaN`` and writes only the slots of its own half, whose flat index is
        ``(h_block * 2 + gate_or_up) * i_256 + i_block``, so the halves are
        disjoint by construction. This takes the gate emission, fills its
        ``NaN`` slots from the up emission, and refuses if one survives -- which
        is what makes that requirement something a load can fail on instead of a
        sentence in a docstring.
        """
        from vllm_neuron.functional.moe.blockwise_fp8_retile import (
            DOWN,
            GATE_UP,
            retile_block_scales,
        )

        #: The producer's fusion selector is an ``int``: its own flat index is
        #: ``(h_block * 2 + gate_or_up) * i_256 + i_block``, so 0 is the gate
        #: half and 1 the up half.
        gate_half, up_half = 0, 1

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
                f"are absent; load the checkpoint before preparing the kernel "
                f"operands"
            )
        for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            weight = supplied[name]
            if weight.dim() != 3:
                raise Glm5NextBlockQuantRouteError(
                    f"{name} must be [E_local, ., .] to give the retile its "
                    f"bank extents, got shape {tuple(weight.shape)}"
                )
            experts = int(weight.shape[0])
            if experts != int(self.num_local_experts):
                raise Glm5NextBlockQuantRouteError(
                    f"{name} carries {experts} experts but this rank owns "
                    f"{self.num_local_experts}; the bank and the partition must "
                    f"agree. A global bank reaching this rank is a load-time "
                    f"error, not something to slice here, because slicing it "
                    f"quietly would put a different rank's experts behind this "
                    f"rank's router columns"
                )

        # ---- THE PRODUCER'S VIEW. ``retile_block_scales`` reads its weight as
        # ``(E, rows, cols)`` where ROWS IS THE H AXIS and COLS THE I AXIS, for
        # BOTH projections: its own refusal text is ``weights must be (E, H, I)``
        # and ``test_moe_path.py``'s landed call site states the same convention in
        # words before doing it. So gate and up, registered ``[E, I_TP, H]``, are
        # handed over in the ``(E, H, I_TP)`` view; down, registered
        # ``[E, H, I_TP]``, is already in that view and is passed unchanged.
        #
        # EACH GRID MOVES WITH ITS WEIGHT, never alone. The producer derives the
        # grid it expects from the weight it received --
        # ``want_scales = (experts, rows // TILE_SIZE, cols // TILE_SIZE)`` -- so a
        # weight transposed by itself is refused where it happens. The dangerous
        # pair is the opposite one: a weight and its grid transposed TOGETHER where
        # neither should be, which for DOWN changes no shape anywhere and every
        # value everywhere.
        #
        # ``.contiguous()`` IS THE FORM THE PRODUCER IS KNOWN TO WORK ON. Every
        # landed call of it hands a freshly built contiguous tensor, and this file
        # already uses ``.t().contiguous()`` for the same job in both
        # ``prepare_projection_weights`` methods. The copy is bounded by what the
        # producer does next anyway: it upcasts its weight to fp32 internally,
        # four times the size of the fp8 copy made here.
        gate = retile_block_scales(
            gate_proj_weight.transpose(1, 2).contiguous(),
            gate_proj_scale.transpose(1, 2).contiguous(),
            GATE_UP,
            gate_half,
        )
        up = retile_block_scales(
            up_proj_weight.transpose(1, 2).contiguous(),
            up_proj_scale.transpose(1, 2).contiguous(),
            GATE_UP,
            up_half,
        )
        down = retile_block_scales(down_proj_weight, down_proj_scale, DOWN)

        # ---- THE FUSION MERGE, and its completeness check.
        gate_up_scales = torch.where(
            torch.isnan(gate.consumer_scales), up.consumer_scales, gate.consumer_scales
        )
        unwritten = int(torch.isnan(gate_up_scales).sum())
        if unwritten:
            raise Glm5NextBlockQuantRouteError(
                f"the fused gate/up consumer scales have {unwritten} slots that "
                f"neither half wrote. The producer writes one half per call and "
                f"leaves the other NaN, so every slot must come from exactly one "
                f"of the two calls above; a survivor means the two emissions do "
                f"not tile the same space"
            )

        # ---- THE FUSED WEIGHT. Each half returns in the ``(E, H, I_TP)`` view
        # it was handed above, so stacking on a new axis 2 gives the
        # ``[E, H, 2, I_TP]`` the kernel's extent check demands. Without those
        # transposes this line produced ``[E, I_TP, 2, H]`` and the consumer's
        # ``gate_up_proj_weight.shape[1] != hidden`` refused it.
        gate_up_weight = torch.stack(
            (gate.retiled_weights, up.retiled_weights), dim=2
        )

        prepared = {
            "gate_up_proj_weight": gate_up_weight,
            "gate_up_consumer_scales": gate_up_scales,
            # BACK TO THE KERNEL'S ORIENTATION. The producer returns its retiled
            # weight in the view it was handed, so down comes back
            # ``(E, H, I_TP)`` and ``block_quant_expert_mm`` requires exactly
            # ``(E, I_TP, H)``. The landed ``test_moe_path.py`` call site performs
            # this same transpose for this same reason. Only the WEIGHT moves: the
            # consumer scales are already in the producer's flat kernel layout and
            # are not a picture of the weight's axes.
            "down_proj_weight": down.retiled_weights.transpose(1, 2).contiguous(),
            "down_consumer_scales": down.consumer_scales,
        }
        setattr(self, self.PREPARED_KERNEL_OPERANDS_ATTR, prepared)
        setattr(
            self,
            self.RETILE_HEALTH_ATTR,
            {
                "gate": (
                    gate.emitted_unsupplied,
                    gate.input_scales_dropped,
                    gate.inexact_rescales,
                ),
                "up": (
                    up.emitted_unsupplied,
                    up.input_scales_dropped,
                    up.inexact_rescales,
                ),
                "down": (
                    down.emitted_unsupplied,
                    down.input_scales_dropped,
                    down.inexact_rescales,
                ),
            },
        )
        return len(prepared)

    def _prepared_kernel_operand(self, name: str) -> torch.Tensor:
        """One prebuilt kernel operand, or a refusal naming what was not done.

        The shared expert's form, for its reason: refusing is what makes "built
        once at load time, never per forward step" checkable. Building on demand
        instead would put a whole-bank retile inside the per-token path and
        nothing would report it.
        """
        prepared = getattr(self, self.PREPARED_KERNEL_OPERANDS_ATTR, None)
        if not prepared:
            raise Glm5NextBlockQuantRouteError(
                "prepare_scale_operands() has not run; this bank's kernel "
                "operands are retiled once at load time, never per forward step"
            )
        return prepared[name]

    # ── the bank's forward -- ``inc-glm53f-054a`` item 2 of 7 ────────────
    #
    # WHAT THIS METHOD IS: the composition, and nothing else. Every piece it
    # needs is already landed. ``block_quant_expert_mm`` above does the route
    # dispatch, the global-to-local expert mapping, the padding slot and the
    # kernel call; ``prepare_scale_operands`` built the four operands that method
    # takes, once, at load time. So this method looks those four up by the names
    # that method declares and calls it. It authors no numerics, no layout and no
    # refusal of its own -- a refusal here would be a second authority on an
    # extent the callee already checks, and two refusals on one extent is how
    # they come to disagree.
    #
    # WHY THE AFFINITIES ARE AN ARGUMENT RATHER THAN ROUTED HERE. ``route_tokens``
    # above returns the GLOBAL router columns and needs the router norm's gamma
    # and the text config, neither of which this bank retains -- ``-032``'s own
    # section note records that the config is threaded in at the call for exactly
    # that reason. Routing inside this method would make it need both and would
    # put the router's call site in two places. The MoE block's forward is where
    # the router and this bank meet, and that is a later item.
    #
    # WHY THERE IS NO CAST. ``block_quant_expert_mm``'s docstring says the return
    # dtype is the seam's own and that the layer forward decides the residual
    # dtype. Casting here would take that decision away from the method the
    # design gives it to.

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
        *,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int = 0,
    ) -> torch.Tensor:
        """Run this rank's routed experts over ``[T, H]`` tokens.

        Args:
            hidden_states: ``[T, H]`` real tokens only.
            expert_affinities: ``[T, E]`` scattered router scores over ALL
                ``n_routed_experts`` -- what :meth:`route_tokens` returns, handed
                straight through.
            quant_config: the resolved quantisation policy, the route selector.
            block_size: tokens per block; defaults to ``BLOCK_QUANT_SIZE``.
            moe_group: the MoE ``GroupCoordinator``, unread at ``tp_degree`` 1.
            tp_degree: ranks sharding each expert's intermediate dimension.
            expert_parallel_rank: which rank's expert slice to select.

        Returns:
            ``[T, H]`` in the seam's own dtype.

        Raises:
            Glm5NextBlockQuantRouteError: if the load-time prep has not run, and
                on any extent disagreement :meth:`block_quant_expert_mm`
                refuses. Both are raised by the methods that own them, which is
                why this one raises nothing.
        """
        return self.block_quant_expert_mm(
            hidden_states=hidden_states,
            expert_affinities=expert_affinities,
            gate_up_proj_weight=self._prepared_kernel_operand(
                "gate_up_proj_weight"
            ),
            down_proj_weight=self._prepared_kernel_operand("down_proj_weight"),
            gate_up_consumer_scales=self._prepared_kernel_operand(
                "gate_up_consumer_scales"
            ),
            down_consumer_scales=self._prepared_kernel_operand(
                "down_consumer_scales"
            ),
            quant_config=quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
        )


# ``inc-glm53f-027``'s named refusal, at module level because an exception a
# caller catches belongs in the module namespace and not nested in the class
# that raises it. It is the second and last hunk of this increment in this file,
# and it is a pure insertion: no line of ``Glm5NextRoutedExperts`` above it or
# ``Glm5NextSharedExperts`` below it moves.
class Glm5NextBlockQuantRouteError(ValueError):
    """A block-quant expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a
    call site that reaches the substrate's ``QuantizationType.NONE`` default by
    OMISSION and computes a different function while every shape check passes.
    """


class Glm5NextSharedExperts(nn.Module):
    """The always-on shared expert at ``mlp.shared_experts``.

    Declared only when ``n_shared_experts`` is nonzero, mirroring the map's
    own condition (``weight_loaders_fp8.py:481``).
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_shared_experts = int(text_config.n_shared_experts)
        # The checkpoint's SwiGLU bound, resolved HERE and not at the call.
        # ``inc-glm53f-033`` repair round 2. The reference clamps both MLP
        # projections with this value before their product
        # (``modeling_glm5_next.py:102-104``), so the bound is a model value like
        # any other and belongs on the object the config builds. This is the same
        # construction-time read ``inc-glm53f-080`` made for the decoder's
        # epsilon in ``Glm5NextKDAAttention.__init__``, on the same kind of
        # scalar, one line below the read this method already made for
        # ``n_shared_experts``.
        self.swiglu_limit = float(text_config.swiglu_limit)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    # ── shared-expert path -- D14 owner: ``inc-glm53f-033`` ───────────────
    #
    # SCOPE. This section adds ONE method, and ``forward`` below is ``-013``'s
    # stub and stays byte-identical.
    #
    # ``__init__`` ABOVE IS NO LONGER BYTE-IDENTICAL, and the reason is recorded
    # rather than left to a diff. Round 1 of this increment's repair made the
    # SwiGLU bound a required argument of the two methods below, because
    # ``Glm5NextTextConfig`` did not model ``swiglu_limit`` at all and there was
    # no config value to read. Round 2 lifted the field
    # (``config.py``'s ``swiglu_limit``), so the bound now has a home on the
    # config, and the lead's round-2 bound directs it to be read where the object
    # is built. One line was added to ``__init__`` and nothing in it was changed,
    # so no landed reading of the three declared parameters or of
    # ``num_shared_experts`` can move.
    #
    # The residual add lives on ``Glm5NextMoEBlock`` -- this increment's second
    # section -- because adding the routed and shared halves needs both children
    # and neither child owns the other.
    #
    # THE COUNT IS 3, AND IT IS WHY THE STRUCTURE BELOW IS THREE CALLS. The
    # route predicate (plan L934-935, revision 33) reads ``-026``'s dispatch
    # counter as EXACTLY 3 -- one per projection site -- per shared-expert call.
    # Three sequential entries into ``blockwise_fp8_mm`` is therefore the
    # criterion and not an implementation convenience: a reading of 3 proves all
    # three projections crossed the block-quant seam, so no projection can
    # silently reach the substrate's non-blockwise MLP path and ignore the block
    # scales, and a bypassed projection reads 2 and fails. ``-033`` attempt 1
    # measured why the earlier declared ``1`` was structurally unreachable
    # (``increments/evidence-033.md``): ``silu`` is non-linear, so ``down``
    # cannot fold into either predecessor, and ``blockwise_fp8_mm`` takes a 2-D
    # weight and adds exactly ``+1`` per invocation at a single unlooped site.
    #
    # NO FUSION IS AUTHORED (plan L938 and the Substrate bullet's own words).
    # ``-013`` landed ``gate_proj_weight``, ``up_proj_weight`` and
    # ``down_proj_weight`` as THREE UNFUSED parameters, matching the weight map
    # (``weight_loaders_fp8.py:481-489``) and the fork's own dense precedent
    # (``Glm5NextDenseMLP`` below: "Gate and up stay **separate** parameters").
    # Concatenating gate and up into one ``[H, 2I]`` operand would author a
    # scale-grid concatenation this increment may not author, and it would read
    # 2 rather than the declared 3.
    #
    # WHY THE WEIGHTS AND SCALES ARE ARGUMENTS -- ``-027``'s measured shape
    # contract (``increments/evidence-027.md`` §2.3), inherited rather than
    # re-litigated. ``-013``'s ``__init__`` declares the three projections by
    # ``register_parameter(name, None)`` and NO scale parameter. Producing the
    # block scales is the weight loader's step, not this section's, so every
    # WEIGHT-LOADER PRODUCT this site consumes is threaded in at the call --
    # ``-032``'s and ``-027``'s precedent for the quant-config argument.
    # **`inc-glm53f-090` NARROWS THE SENTENCE ABOVE and does not delete it: the
    # public grids are still threaded in at a call, but the DERIVED KERNEL
    # OPERAND is built from them once and held on this module. See the
    # `-090` paragraph below, which states the new rule in full.**
    #
    # THE SWIGLU BOUND IS NOT ONE OF THOSE, and the distinction is the whole
    # reason it moved in round 2. A block scale is produced per checkpoint load
    # by the weight loader; ``swiglu_limit`` is a scalar the checkpoint declares
    # and the config models, exactly like ``n_shared_experts`` one line up and
    # like ``rms_norm_eps`` on the KDA attention. Model scalars resolve at
    # construction, weight-loader products arrive at the call, and the two rules
    # do not compete.
    #
    # THE SCALE OPERAND IS THE PUBLIC GRID, SO NEITHER ``to_kernel_scale_layout``
    # IS IMPORTED HERE. The campaign carries two helpers of that name at
    # different arities (``functional/blockwise_fp8_mm.py:309`` takes
    # ``(weight_scale, rows, cols)``; ``functional/moe/moe_blockwise_fp8.py:170``
    # takes ``(consumer_scales, num_experts, rows, cols, projection)``).
    # ``blockwise_fp8_mm`` applies the dense one ITSELF at
    # ``blockwise_fp8_mm.py:491``, so this site passes the public
    # ``[K//256, N//256]`` grid and imports neither -- the disambiguation
    # hazard is removed rather than navigated.
    # **`inc-glm53f-090` REVERSES THE PARAGRAPH ABOVE. The dense helper IS now
    # imported here, by ``prepare_scale_operands`` below, and the seam applies it
    # itself ONLY when no prebuilt operand arrives. The disambiguation hazard is
    # therefore navigated rather than removed, and it is navigated explicitly:
    # the import names its module in full and sits next to this note.**
    #
    # ── `inc-glm53f-090`: WHERE THE SCALE OPERAND IS BUILT ────────────────────
    #
    # WHY IT MOVED, measured rather than preferred. ``to_kernel_scale_layout``
    # allocates and then scatters ONE ELEMENT AT A TIME
    # (``blockwise_fp8_mm.py:340-342`` and ``:343-347``). At this campaign's dense
    # geometry that is 128 scatters per projection and 384 per shared-expert
    # call, on every layer that takes the route, every forward step -- review
    # finding B26-M2. The operand is a pure function of a weight-loader product,
    # so it is built ONCE, by ``prepare_scale_operands`` below, and held on this
    # module; the three sites pass it to the seam by keyword.
    #
    # THE RULE THAT REPLACES THE TWO ABOVE, stated as one sentence: model scalars
    # resolve at construction, PUBLIC weight-loader products arrive at the call,
    # and an operand DERIVED from a weight-loader product that never changes is
    # built once at load time and held. The public grids are still arguments --
    # ``shared_expert_mm``'s signature does not move -- because the seam's
    # torch-oracle fallback consumes the public grid and a prebuilt kernel
    # operand cannot stand in for it.
    #
    # WHY A CACHE INSIDE THE BRIDGE WAS NOT THE REMEDY, and this is measured too:
    # ``-076`` r1 tried exactly that and the capture route refused it -- a dict
    # keyed on a traced tensor is not expressible on the traced path, and the
    # capture route feeds ``meta`` tensors, which carry no values to key on
    # (``increments/evidence-076-r1.md`` §3). The operand has to arrive as a graph
    # INPUT, which is what building it before capture makes it.
    #
    # WHAT THIS DOES NOT WEAKEN. Skipping the bridge also skips the bridge's own
    # grid check, and the seam's replacement check is weaker: it pins the element
    # COUNT, so a transposed grid would pass it. The strong check still runs, at
    # build time, because ``prepare_scale_operands`` builds THROUGH
    # ``to_kernel_scale_layout`` rather than around it -- so the public grid is
    # still compared against ``(rows, cols)`` before it is ever flattened, once
    # per load instead of once per forward step.
    #
    # Imports are FUNCTION-LOCAL, following the landed ``-023``, ``-031``,
    # ``-032`` and ``-027`` precedent in this file: the module import block is
    # ``-013``'s D14 section.

    #: Where :meth:`prepare_scale_operands` leaves its three operands. A class
    #: attribute for the same reason ``PREPARED_WEIGHTS_ATTR`` is one on the
    #: projections section: the name is part of the contract between the builder
    #: and the reader, and neither should spell it twice.
    PREPARED_SCALE_OPERANDS_ATTR = "_prepared_scale_operands"

    #: Where :meth:`retile_checkpoint_scale_grids` records what the coarsening
    #: cost, per projection, and which projections it left alone. This block's
    #: acceptance reads it; no consumer does.
    SHARED_RETILE_HEALTH_ATTR = "_shared_expert_retile_health"

    # ── the load-path retile -- ``inc-glm53f-054a`` hand-off item (iv) ─────
    #
    # WHAT IT FIXES. The checkpoint stores one scale per ``128 x 128`` tile and
    # :meth:`prepare_scale_operands` above consumes the PUBLIC grid, one scale per
    # ``256 x 256`` block. Nothing on this module's load path bridged the two, so a
    # real load reached the prep with a ``(4, 2)`` grid where it wanted ``(2, 1)``
    # and refused. ``inc-glm53f-101`` attempt 2 recorded that refusal by name
    # (``BlockwiseFp8MmError: weight_scale has shape (4, 2), expected (2, 1) for a
    # [K=512, N=256] weight``) and DECISIONS §84 placed the bridge here.
    #
    # THE WEIGHT MOVES WITH THE GRID, and that is the whole reason this is a
    # retile and not a grid rewrite. Coarsening keeps ONE of the four tile scales
    # per block, so the other three tiles' values are wrong against the retained
    # scale until they are rescaled by their own ratio -- which is what
    # ``retile_block_scales`` does to the weight it returns
    # (``blockwise_fp8_retile.py:414-424``). Publishing the coarser grid beside the
    # original weight would change no shape and every number, and the seam's
    # element-count check would pass it. So both are replaced or neither is.
    #
    # IT MOVES NO LANDED COUNT, and that is measured rather than hoped. A module of
    # this class exists only where a MoE block does, and of the fixtures in
    # ``test_load_weights.py`` that complete a load, every one either sets
    # ``n_shared_experts=0`` (``_stacked_config``, ``_shard_config``,
    # ``_grid_shard_config`` -- ``_shard_config``'s docstring records why) or is
    # all-dense and so builds no MoE block at all (``_dense_config``). The two
    # fixtures that DO build this module both refuse today, and ``-101``'s records
    # the flip in its own words. So no green load changes what it reports.
    #
    # THE SKIP IS RECORDED, NEVER SILENT. A weight whose extents are not whole
    # ``256`` blocks has no public grid to build, so this method leaves that
    # projection exactly as the loader left it and says so in the health record.
    # Skipping quietly is how a missing retile would look like a working one.

    def retile_checkpoint_scale_grids(self) -> int:
        """Coarsen this module's checkpoint grids onto the public grid, ONCE.

        Returns how many projections were retiled -- ``3`` on a checkpoint whose
        extents are whole ``256`` blocks, ``0`` on a miniature that has no public
        grid to build.

        Raises:
            Glm5NextSharedExpertRouteError: if a weight or grid is not 2-D, or a
                grid is not at the checkpoint's own ``128``-tile granularity. An
                already-public grid is refused rather than passed over, because a
                second retile of an already-retiled grid would rescale the weight
                twice and no shape would object.
        """
        from vllm_neuron.functional.moe.blockwise_fp8_retile import (
            BLOCK_QUANT_SIZE,
            DOWN,
            TILE_SIZE,
            retile_block_scales,
        )

        health: dict[str, dict[str, object]] = {}
        retiled = 0
        for leaf in _scale_prep_leaves(self):
            grid_name = Glm5NextForConditionalGeneration._sibling_scale_grid_name(
                leaf
            )
            weight = getattr(self, leaf)
            grid = getattr(self, grid_name)
            if weight.dim() != 2:
                raise Glm5NextSharedExpertRouteError(
                    f"{leaf} must be 2-D to give the retile its extents, got "
                    f"shape {tuple(weight.shape)}"
                )
            if grid.dim() != 2:
                raise Glm5NextSharedExpertRouteError(
                    f"{grid_name} must be 2-D, got shape {tuple(grid.shape)}"
                )
            rows, cols = int(weight.shape[0]), int(weight.shape[1])
            if rows % BLOCK_QUANT_SIZE or cols % BLOCK_QUANT_SIZE:
                # No public grid exists for these extents. Recorded, not silent.
                health[leaf] = {
                    "retiled": False,
                    "extents": (rows, cols),
                    "reason": (
                        f"[{rows},{cols}] is not a whole number of "
                        f"{BLOCK_QUANT_SIZE}x{BLOCK_QUANT_SIZE} blocks"
                    ),
                }
                continue
            checkpoint_grid = (rows // TILE_SIZE, cols // TILE_SIZE)
            if tuple(grid.shape) != checkpoint_grid:
                raise Glm5NextSharedExpertRouteError(
                    f"{grid_name} has shape {tuple(grid.shape)}; this retile "
                    f"consumes the checkpoint's own {TILE_SIZE}-tile grid "
                    f"{checkpoint_grid} for a [{rows},{cols}] weight. A grid "
                    f"already at {BLOCK_QUANT_SIZE} granularity is refused here "
                    f"rather than passed over: retiling twice rescales the "
                    f"weight twice and no shape check would see it."
                )
            # ``DOWN`` selects the CONSUMER flattening, and this method reads
            # neither consumer field -- only ``block_scales`` (the retained scale
            # per block) and ``retiled_weights``. DOWN is named because its
            # flattening writes every emitted slot exactly once, so the two health
            # counters below stay readable; GATE_UP leaves half its slots NaN by
            # design and would make them say nothing here.
            result = retile_block_scales(
                weight.data.unsqueeze(0).contiguous(),
                grid.unsqueeze(0).contiguous(),
                DOWN,
            )
            # ``block_scales`` is ``(E, i_256, h_256)``
            # (``blockwise_fp8_retile.py:371``, written at ``:382``), so the
            # transpose puts it back in the weight's own ``(rows, cols)`` frame,
            # which is the frame ``to_kernel_scale_layout`` compares against.
            public = result.block_scales[0].t().contiguous()
            # ``.data`` ASSIGNMENT, not a rebind. ``setattr(self, leaf, tensor)``
            # would drop the ``nn.Parameter`` and with it every landed reading
            # that counts ``named_parameters()``. The device is carried over
            # explicitly because the producer allocates its grid with
            # ``torch.full`` and no device (``:371``), which lands on the CPU.
            weight.data = result.retiled_weights[0].to(
                device=weight.device, dtype=weight.dtype
            )
            setattr(self, grid_name, public.to(device=grid.device))
            health[leaf] = {
                "retiled": True,
                "extents": (rows, cols),
                "checkpoint_grid": checkpoint_grid,
                "public_grid": tuple(public.shape),
                "emitted_unsupplied": result.emitted_unsupplied,
                "input_scales_dropped": result.input_scales_dropped,
                "inexact_rescales": result.inexact_rescales,
            }
            retiled += 1
        setattr(self, self.SHARED_RETILE_HEALTH_ATTR, health)
        return retiled

    def prepare_scale_operands(
        self,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
    ) -> int:
        """Build the three kernel scale operands ONCE. Returns how many.

        `inc-glm53f-090`. The seam's bridge scatters one element at a time, so
        building the operand inside the per-forward path costs 128 device writes
        per projection at this campaign's dense geometry. A block scale is a
        weight-loader product that never changes after a load, so the operand it
        implies is built here, once, and held on this module.

        The argument list mirrors :meth:`shared_expert_mm`'s own -- same operands,
        same order -- because the two methods consume the same things and a
        divergent order at one of them is a mis-wiring no shape check would see.

        Args:
            gate_proj_weight: ``[H, I]`` fp8-e4m3. Read for its extents only.
            up_proj_weight: ``[H, I]`` fp8-e4m3.
            down_proj_weight: ``[I, H]`` fp8-e4m3.
            gate_proj_scale: ``[H//256, I//256]`` fp32, the PUBLIC grid.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[I//256, H//256]`` fp32, for ``down_proj_weight``.

        Returns:
            How many operands were built -- ``3`` on every successful call.

        Raises:
            Glm5NextSharedExpertRouteError: if an operand is missing or a weight
                is not 2-D. The grid's own agreement with the weight extents is
                checked by ``to_kernel_scale_layout`` below rather than here, so
                there is one authority for it and not two that can drift.

        THE EXTENTS COME FROM THE WEIGHTS, and that is the load-bearing choice.
        Deriving them from the grid's own shape instead would make this method
        unable to detect a transposed grid, because the seam's replacement check
        pins only the element count. Building THROUGH
        ``to_kernel_scale_layout`` keeps the landed grid check -- public grid
        against ``(rows, cols)``, before any flattening -- and simply moves it
        from once per forward step to once per load.
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
            prepared[name] = to_kernel_scale_layout(scale, rows, cols)
        setattr(self, self.PREPARED_SCALE_OPERANDS_ATTR, prepared)
        return len(prepared)

    def _prepared_scale_operand(self, name: str) -> torch.Tensor:
        """One prebuilt scale operand, or a refusal naming what was not done.

        The same form the projections section uses for its prepared weights, and
        for the same reason: refusing is what makes "never per forward step"
        checkable. Falling back to building on demand would put the per-call
        scatter back silently and nothing would report it.
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
        """Run the always-on shared expert through ``-026``'s dense block GEMM.

        The SwiGLU the checkpoint stores, **and it clamps both projections before
        the product**: ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))``, where
        ``L`` is ``self.swiglu_limit``, the checkpoint's own bound resolved from
        the config when this object was built. Each of the three projections
        is a separate entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`.

        The clamp is the checkpoint's, not a guard this code invented; the repair
        note at the activation step below cites the reference line for line and
        says what the omission cost.

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. ``T`` must be a
                whole number of ``TILE_SIZE`` rows -- the seam tiles ``M`` over
                the PSUM partition axis and does not pad
                (``blockwise_fp8_mm.py:239-245``), so padding is the caller's.
            gate_proj_weight: ``[H, I]`` fp8-e4m3, expressed against
                ``gate_proj_scale``.
            up_proj_weight: ``[H, I]`` fp8-e4m3.
            down_proj_weight: ``[I, H]`` fp8-e4m3.
            gate_proj_scale: ``[H//256, I//256]`` fp32, the PUBLIC block-scale
                grid :func:`~vllm_neuron.functional.blockwise_fp8_mm.scale_grid_shape`
                declares -- one scale per ``256 x 256`` weight block.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[I//256, H//256]`` fp32, for ``down_proj_weight``.
            quant_config: the resolved per-model quantisation policy, and the
                route selector; see the section note above.

        The SwiGLU bound is NOT an argument. It is ``self.swiglu_limit``,
        resolved from ``text_config.swiglu_limit`` in ``__init__``, so this method
        cannot be called with a bound that is not the one the checkpoint declared
        for this model. Round 1 of the ``B22-M1`` repair did pass it in, because
        the config did not model the key yet; round 2 lifted the field and moved
        the read to construction. **No literal bound appears on this path**, which
        is the half of ``B22-M1-shared-expert-swiglu-clamp-omitted`` that asked
        for the value to come from the checkpoint.

        Returns:
            ``[T, H]`` **fp32** -- the seam's own return dtype, not re-cast here.
            The layer forward decides the residual dtype and that is
            ``inc-glm53f-054``'s section, exactly as ``-027`` left it.

        Raises:
            Glm5NextSharedExpertRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one the
                retile bridges, or when two operand extents contradict each
                other. Named rather than coerced, so a mis-wired call site fails
                where it is wrong instead of computing a different function.
        """
        from torch.nn.functional import silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm
        from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

        # ---- ROUTE SELECTION. The certifying component (D1.4). ----------- #
        # Same shape as ``-027``'s: the route is selected by WHICH FUNCTION IS
        # CALLED and the not-block-quant branch RAISES BY NAME, so it cannot
        # fall through to the substrate's ``QuantizationType.NONE`` default
        # (``functional/mlp.py:81``, ``:249`` -- a default reached by OMISSION).
        # No quantisation enum member is named or added (plan section 11
        # constraint B.6). The counter clause DETECTS that silent fallback; this
        # raise makes it impossible.
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
                f"route consumes scales retiled from ({TILE_SIZE}, {TILE_SIZE}) "
                f"checkpoint blocks onto BLOCK_QUANT_SIZE granularity and has no "
                f"path for any other checkpoint block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        # Only the cross-operand agreements the seam cannot see are checked
        # here. ``blockwise_fp8_mm`` already refuses a K mismatch
        # (``blockwise_fp8_mm.py:467-471``), a mis-sized or non-fp32 scale grid
        # (``:328``, ``:335``) and every blocking condition
        # (``_require_blocked``), and repeating those would create a second
        # authority that can drift from the first.
        #
        # `inc-glm53f-090` CORRECTED THE K-MISMATCH CITE, and it was already
        # wrong before this increment: it read line 420, which at `0880c8f` was
        # the docstring's closing quotes and not a refusal at all -- the check
        # has always been two lines below. This increment's insert then moved
        # the real check down, so the span above is the live one. The historical
        # numbers in this note are deliberately NOT backticked: a backticked
        # number is a cite to the checker, and a quoted dead one would read as
        # drift for good. Naming the file also stops the checker inheriting the
        # wrong module for the two bare cites above, which it did at `0880c8f`.
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

        # ---- The three projection sites. The counted seam entries. ------- #
        # Each passes the operand ``prepare_scale_operands`` built at load time,
        # by keyword (`inc-glm53f-090`). The public grid is still passed too: the
        # seam's torch-oracle fallback consumes it, and the prebuilt operand
        # cannot stand in for it. The dispatch count is untouched at 3 -- this
        # changes what each entry CARRIES, not how many entries there are.
        # ENTRY 1 of 3 -- gate.
        gate = blockwise_fp8_mm(
            hidden_states,
            gate_proj_weight,
            gate_proj_scale,
            prebuilt_scale_t=self._prepared_scale_operand("gate_proj"),
        )
        # ENTRY 2 of 3 -- up.
        up = blockwise_fp8_mm(
            hidden_states,
            up_proj_weight,
            up_proj_scale,
            prebuilt_scale_t=self._prepared_scale_operand("up_proj"),
        )

        # ---- THE CHECKPOINT'S TWO CLAMPS. `B22-M1`, rounds 1 and 2. -------- #
        # WHAT WAS WRONG. This step read `activated = silu(gate) * up`, with no
        # clamp, so it computed a DIFFERENT FUNCTION from the one the checkpoint
        # stores. The correctness reference this campaign declares -- transformers
        # v5.16.1, `Glm5NextTextMLP.forward`
        # (`models/glm5_next/modeling_glm5_next.py:98-104`, and `:196-197` builds
        # `Glm5NextTextMoE.shared_experts` as that same MLP) -- bounds the gate
        # ABOVE and the up operand on BOTH sides before multiplying:
        #
        #     gate = gate.clamp(min=None, max=self.swiglu_limit)   # `:102`
        #     up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)  # `:103`
        #
        # The two lines below are that transliteration, `min=None` on the gate
        # included: the gate's bound is ONE-SIDED in the reference and copying it
        # as a two-sided clamp would be a second wrong function, not a tidier
        # one.
        #
        # WHAT THE OMISSION COST, measured rather than argued. At this section's
        # own declared fixture the pre-activations are gate `[67.94, 184.89]` and
        # up `[84.30, 166.43]`, so every element of both operands sits outside
        # the checkpoint's `[-10, 10]` box, and the clamped and unclamped results
        # differ by `max_rel_error=1.437580e+02` against a declared `rtol=3e-2`
        # (`increments/probe-R7-clamp-and-config-lift.out`). On the real
        # checkpoint that error entered the residual stream on each of the 42 MoE
        # layers, silently: no shape moves, nothing raises, and the route counter
        # still reads `nki_dispatch=3, torch_fallback=0`.
        #
        # WHERE THE BOUND COMES FROM. `self.swiglu_limit`, resolved from
        # `text_config.swiglu_limit` in `__init__`. Round 1 passed it in as a
        # required argument because `Glm5NextTextConfig` did not model the key at
        # all; round 2 lifted the field (`config.py`'s `swiglu_limit`, defaulting
        # to the checkpoint's own `10.0`) and moved the read to construction, so a
        # production caller resolves nothing and `inc-glm53f-054` will find this
        # method needing only its operands. Reading it off `self` also removes the
        # last way a caller could hand this path a bound the checkpoint never
        # declared. See `increments/evidence-033-r2.md`.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # The SwiGLU wiring. This is call-site plumbing, not authored numerics:
        # ``silu`` is torch's own, the product is elementwise, and both run in
        # the seam's fp32 return dtype so no precision is thrown away between
        # the projections. This is also the exact reason the declared count
        # cannot be 1 -- ``silu`` is non-linear, so ``down`` cannot fold into
        # either predecessor and the product must be materialised here.
        activated = silu(gate) * up

        # The down projection re-enters the seam, whose declared input dtype is
        # ``bfloat16`` (``blockwise_fp8_mm.py:446``), so the fp32 intermediate is
        # cast back to the activation dtype. The cast is named rather than
        # implicit because it is a real precision step and the acceptance's torch
        # reference mirrors it at the same point.
        # ENTRY 3 of 3 -- down.
        return blockwise_fp8_mm(
            activated.to(hidden_states.dtype),
            down_proj_weight,
            down_proj_scale,
            prebuilt_scale_t=self._prepared_scale_operand("down_proj"),
        )

    # ── the shared expert's forward -- ``inc-glm53f-054a`` item 3 of 7 ────
    #
    # WHAT THIS METHOD IS: the operand lookup, and nothing else.
    # :meth:`shared_expert_mm` above is the whole compute path and is landed and
    # separately accepted; it takes its six operands as arguments, on the
    # recorded ground that a weight-loader product is threaded in at the call.
    # This method is the ``nn.Module`` entry point, so it is where "at the call"
    # resolves to "off this module": it reads the three weights the
    # declaration tuple names and the three grids the loader attached beside
    # them, and hands them over.
    #
    # WHY THE GRID NAMES ARE DERIVED AND NOT SPELLED. The rule lives once, in
    # ``_sibling_scale_grid_name``, and the retile above already reaches it that
    # way. Spelling ``gate_proj_weight_scale_inv`` here would be a second copy of
    # a naming convention that the loader, the prep loop and the retile all read
    # from that one definition.
    #
    # WHY IT RAISES ON A MISSING GRID AND ON NOTHING ELSE. A missing grid is not
    # something the callee can check -- it receives grids, so an absent attribute
    # reaches it as ``AttributeError`` from inside a path that is not at fault,
    # and the dense MLP's forward refuses on exactly this for exactly this
    # reason. Every EXTENT agreement is the callee's, which checks each one
    # already; a second check here is how two authorities on one extent come to
    # disagree.
    #
    # WHAT IT DOES NOT DO. It does not add a residual, which is
    # :meth:`Glm5NextMoEBlock.combine_routed_and_shared`'s one add, and it does
    # not cast the seam's fp32 return -- the layer forward decides the residual
    # dtype, which is this block's later item and not this one.
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
                route selector. An ARGUMENT rather than a field, matching the
                three landed methods of this family and
                :meth:`Glm5NextDenseMLP.forward`; no module in this file holds a
                policy.

        Returns:
            ``[T, H]`` **fp32**, exactly what :meth:`shared_expert_mm` returns.

        Raises:
            Glm5NextSharedExpertRouteError: when a scale grid the loader should
                have attached is absent. Refusing rather than running an
                unscaled matmul, which returns plausible numbers.
        """
        return self.shared_expert_mm(
            hidden_states, *self.scale_route_operands(), quant_config
        )

    #: The three projections this module routes, in the order
    #: :meth:`shared_expert_mm` and :meth:`prepare_scale_operands` both declare.
    #: A class attribute for the reason the two operand-attribute names above are
    #: class attributes: the order is part of a contract between several readers
    #: and none of them should spell it a second time.
    SCALE_ROUTE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

    def scale_route_operands(self) -> tuple[torch.Tensor, ...]:
        """The six operands the shared route takes: three weights, then three grids.

        ``inc-glm53f-054a``. ONE definition of the lookup, because two callers
        need it: this module's own :meth:`forward` above, and
        :meth:`Glm5NextMoEBlock.forward`, which must hand the same six to the
        landed :meth:`Glm5NextMoEBlock.combine_routed_and_shared` -- the single
        place in this file where a shared contribution is added to a routed one.
        A second copy of the lookup in the parent is how the two come to disagree
        about which grid belongs to which weight.

        THE ORDER IS THE TWO LANDED METHODS' OWN -- weights first, then grids,
        each triple in declaration order -- so a caller can splat this straight
        into either signature. A divergent order at one call site is a mis-wiring
        no shape check would see, which is the reason
        :meth:`prepare_scale_operands` gives for mirroring the same list.

        Returns:
            ``(gate_w, up_w, down_w, gate_grid, up_grid, down_grid)``.

        Raises:
            Glm5NextSharedExpertRouteError: when a scale grid the loader should
                have attached is absent. The grid names are DERIVED from the
                weight leaves by ``_sibling_scale_grid_name``, which is the single
                definition of that convention; spelling one here would be a
                second copy of a rule the loader, the prep loop and the retile all
                read from that one place.
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
                    f"each declared weight, and this route consumes the PUBLIC "
                    f"grid at BLOCK_QUANT_SIZE granularity that "
                    f"retile_checkpoint_scale_grids publishes. Refusing rather "
                    f"than running an unscaled matmul, which returns plausible "
                    f"numbers."
                )
            weights.append(getattr(self, leaf))
            grids.append(grid)
        return (*weights, *grids)


# ``inc-glm53f-033``'s named refusal, at module level for the same reason
# ``-027``'s is: an exception a caller catches belongs in the module namespace
# and not nested in the class that raises it. It is a pure insertion between two
# classes -- no line of ``Glm5NextSharedExperts`` above it or
# ``Glm5NextMoEBlock`` below it moves.
class Glm5NextSharedExpertRouteError(ValueError):
    """A shared-expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a
    call site that reaches the substrate's ``QuantizationType.NONE`` default by
    OMISSION and computes a different function while every shape check passes.
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
        # ``world_size`` is a trailing optional addition by ``inc-glm53f-031``,
        # threaded to the routed bank only. ``_build_mlp``'s call site is
        # unchanged and stays outside this increment's surface.
        #
        # ``ep_degree`` is the same shape, added by ``inc-glm53f-087``: also
        # trailing, also optional, also threaded to the routed bank only. It is
        # the EXPERT-PARALLEL degree the bank divides by; ``world_size`` stays the
        # tensor-parallel one. ``_build_mlp``'s signature still does not move.
        self.experts = Glm5NextRoutedExperts(
            text_config, world_size=world_size, ep_degree=ep_degree
        )
        if text_config.n_shared_experts:
            self.shared_experts = Glm5NextSharedExperts(text_config)

    # ── the residual add -- D14 owner: ``inc-glm53f-033`` ─────────────────
    #
    # SCOPE. This section adds ONE method and edits no line above or below it.
    # ``__init__`` above and ``forward`` below are ``inc-glm53f-013``'s landed
    # code and stay byte-identical.
    #
    # WHY THE ADD LIVES HERE AND NOT ON EITHER CHILD. It needs the routed half
    # and the shared half, and neither ``Glm5NextRoutedExperts`` nor
    # ``Glm5NextSharedExperts`` owns the other. ``Glm5NextMoEBlock`` holds both
    # (``__init__`` above), so it is the one place the two halves meet. D14's
    # sub-class rule sends the compute to the child that owns the weights --
    # which is why ``-027`` put the routed matmul on the routed bank and this
    # increment put the shared matmul on the shared bank -- and it sends the
    # combination to their parent.
    #
    # "ADDED EXACTLY ONCE" IS STRUCTURAL HERE, NOT ASSERTED. The plan's second
    # acceptance conjunct (L933) proves the shared contribution enters the layer
    # output once rather than twice. That property is made true by construction
    # below: this method contains EXACTLY ONE call to ``shared_expert_mm`` and
    # EXACTLY ONE ``+``, and it is the only place in this file that adds a shared
    # contribution to a routed one. The acceptance measures the property
    # numerically; this structure is what makes the measurement reproducible
    # rather than incidental.
    #
    # WHAT THIS SECTION DOES NOT DO, deliberately: it does not assemble the layer
    # forward (``-013`` / ``-054``), it does not call the routed path (``-027``'s
    # ``block_quant_expert_mm``, landed and separately accepted), and it authors
    # no numerics. ``routed_output`` arrives as an argument precisely so this
    # method composes the two halves without owning either.

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
    ) -> torch.Tensor:
        """``routed_output + shared_expert(hidden_states)`` -- the layer output.

        Args:
            routed_output: ``[T, H]`` the routed experts' contribution, as
                ``inc-glm53f-027``'s ``block_quant_expert_mm`` returns it. Passed
                in rather than computed here; see the section note above.
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

        The SwiGLU bound is not an argument here either, and it is not forwarded.
        The shared expert this block built holds it
        (:meth:`Glm5NextSharedExperts.__init__`), so the bound reaches the clamp
        without passing through this method at all. ``inc-glm53f-033`` repair
        round 2; round 1 forwarded a required argument because the config did not
        model the key yet.

        Returns:
            ``[T, H]`` fp32 -- the sum, in the shared half's fp32 seam dtype.
            The residual dtype for the decoder layer is ``inc-glm53f-054``'s.

        Raises:
            Glm5NextSharedExpertRouteError: when this block declares no shared
                expert, or when ``routed_output`` and the shared contribution
                disagree in extent. A silent broadcast is the failure this
                refuses: ``[T, H] + [1, H]`` and ``[T, H] + [T, 1]`` both
                broadcast without error and both compute a different layer.
        """
        # ``n_shared_experts == 0`` leaves the attribute undeclared -- ``-013``'s
        # ``__init__`` mirrors the weight map's own condition
        # (``weight_loaders_fp8.py:481``). Checked rather than assumed, because
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

        # THE ONE call to the shared path.
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

        # Extents compared EXACTLY, before the add. ``torch`` would broadcast a
        # disagreeing extent silently and return a plausible tensor of the wrong
        # shape, which no downstream shape check would catch.
        if tuple(routed_output.shape) != tuple(shared_output.shape):
            raise Glm5NextSharedExpertRouteError(
                f"routed_output has shape {tuple(routed_output.shape)} and the "
                f"shared contribution has shape {tuple(shared_output.shape)}; "
                f"they must agree exactly. Refusing to add: torch would "
                f"broadcast these and compute a different layer without error."
            )

        # THE ONE add. The shared contribution enters the layer output here and
        # nowhere else in this file.
        return routed_output + shared_output

    # ── the sparse MLP's forward -- ``inc-glm53f-054a`` item 4 of 7 ───────
    #
    # WHAT THIS METHOD IS: the three landed pieces in the reference's own order,
    # and nothing else. The router is ``route_tokens`` on the bank, the routed
    # half is the bank's own forward, and the add is
    # :meth:`combine_routed_and_shared` above. This method authors no numerics
    # and no refusal: every extent is checked by the method that owns it.
    #
    # WHY IT TAKES THE ACTIVATIONS TWICE, and this is the load-bearing reading of
    # the whole method. The reference's MoE block receives ONE tensor, already
    # normalised by the layer, and hands that same tensor to the router and to
    # both expert halves (``modeling_glm5_next.py:200-207``: ``self.gate(hidden_states)``
    # and ``self.experts(...)`` and ``self.shared_experts(residuals)``, where
    # ``residuals`` is this block's own input). This fork's router is FUSED: the
    # RMSNorm is inside ``inc-glm53f-032``'s kernel, so ``route_tokens`` must be
    # handed the PRE-norm activations together with the norm's gain
    # (``route_tokens``'s own signature and docstring). Handing it the normalised
    # tensor would normalise twice and compute a different router; computing the
    # norm here for the experts instead would put a second authority on the
    # layer's own FFN norm. So the layer normalises once, and passes both what it
    # started with and what it produced.
    #
    # WHY THE ROUTER'S GAIN IS AN ARGUMENT AND NOT A PARAMETER HERE. It is the
    # decoder layer's ``post_attention_layernorm_weight``: the checkpoint declares
    # no router-norm tensor at all (``weight_loaders_fp8.py:660-665`` maps the
    # router's weight and its correction bias and nothing else), so the gain the
    # fused kernel needs belongs to the layer, which is where this block's caller
    # sits.
    #
    # WHY ``text_config`` ARRIVES AT THE CALL. ``-031``'s ``__init__`` retains no
    # config and ``route_tokens`` needs the routing hyperparameters, so ``-032``
    # threads it in at the call; this method is a caller and follows that.
    #
    # THE NO-SHARED-EXPERT BRANCH IS THE LANDED METHOD'S OWN WORDS. A block built
    # with ``n_shared_experts == 0`` declares no shared module, and
    # :meth:`combine_routed_and_shared` refuses such a call by name because "the
    # routed output is already the layer output on such a block". So this method
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
        expert_parallel_rank: int = 0,
    ) -> torch.Tensor:
        """One sparse layer's MLP: route, run this rank's experts, add the shared.

        Args:
            hidden_states: ``[T, H]`` the layer's PRE-norm activations, for the
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
            expert_parallel_rank: which rank's expert slice to select.

        Returns:
            ``[T, H]`` in the seams' own dtype. The residual dtype is the layer
            forward's decision, not this method's.

        Raises:
            Glm5NextBlockQuantRouteError: whatever the bank refuses.
            Glm5NextSharedExpertRouteError: whatever the shared route refuses,
                including the extent disagreement the add checks.
        """
        # THE ROUTER. ``route_tokens`` declares ``[B, S, H]`` and the seam
        # flattens ``B`` and ``S`` into one token axis, so a 2-D activation
        # tensor is spelled ``[1, T, H]`` here rather than reshaped inside the
        # callee. Only the affinities are consumed: the logits are the oracle's
        # and the index set is the affinities' own support.
        _logits, _expert_index, expert_affinities = self.experts.route_tokens(
            hidden_states.unsqueeze(0), router_gamma, text_config
        )

        routed_output = self.experts(
            normed_hidden_states,
            expert_affinities,
            quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
        )

        shared_experts = getattr(self, "shared_experts", None)
        if shared_experts is None:
            return routed_output
        return self.combine_routed_and_shared(
            routed_output,
            normed_hidden_states,
            *shared_experts.scale_route_operands(),
            quant_config,
        )


class Glm5NextDenseMLP(nn.Module):
    """The dense MLP on the first ``first_k_dense_replace`` layers.

    Gate and up stay **separate** parameters, matching the map, which follows
    the fork's own dense precedent rather than fusing them
    (``weight_loaders_fp8.py:429-431``).
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.intermediate_size = int(text_config.intermediate_size)
        # The checkpoint's SwiGLU bound, resolved HERE and not at the call --
        # the same construction-time read ``inc-glm53f-033`` repair round 2 made
        # for :class:`Glm5NextSharedExperts` (``:1857``), on the same scalar,
        # from the same config field.
        #
        # WHY THE DENSE MLP CARRIES IT TOO, read off the reference rather than
        # off a comment about the reference. The reference has ONE MLP class,
        # ``Glm5NextTextMLP`` (``modeling_glm5_next.py:86``), and it is
        # CONSTRUCTED at exactly two sites: ``self.shared_experts`` at ``:196``
        # and the dense ``else`` arm of ``self.mlp`` at ``:1271``. That one class
        # stores the bound at ``:96`` and clamps with it at ``:102-103``, so the
        # dense MLP clamps with the same value from the same field -- there is no
        # separate dense reading to get wrong. Measured against the reference at
        # the digest this campaign pins, 56 of 56 checks, in
        # ``../../../artifacts/campaigns/glm-5.3-flash-port/increments/probe-054a-swiglu-clamp-r2.out``.
        #
        # AN UNCLAMPED DENSE MLP WOULD BE THE ``-033`` DEFECT AGAIN. That repair
        # (``B22-M1-shared-expert-swiglu-clamp-omitted``) was for a path that
        # computed a different function from the checkpoint's on every token
        # leaving the bound's box, with no shape moving and nothing raising.
        # ONE LINE IS ADDED AND NOTHING ABOVE IT CHANGES, so no landed reading of
        # ``intermediate_size`` or of the three declared parameters can move.
        self.swiglu_limit = float(text_config.swiglu_limit)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    # ── the dense-MLP compute path -- D14 owner: ``inc-glm53f-054a`` ───────
    #
    # SCOPE. This section replaces ``forward`` below and adds the one
    # ``__init__`` line above. It adds no method, touches no other class, and
    # calls no landed method of another class -- the shared expert's
    # ``shared_expert_mm`` is the same arithmetic on the same seam, and it is
    # NOT called from here because it reads that class's prepared scale
    # operands off ``self`` (``_prepared_scale_operand``, ``:2053``) and this
    # class has no prep. Reaching into it would either move that landed method
    # or bind this forward to another module's instance state.
    #
    # WHY NO PREBUILT SCALE OPERAND HERE. ``blockwise_fp8_mm``'s
    # ``prebuilt_scale_t`` is optional and keyword-only, and omitting it makes
    # the call behave exactly as it did before ``inc-glm53f-090``
    # (``blockwise_fp8_mm.py:441``, ``:449-456``). ``-090``'s load-time prep is
    # reached by ``_run_load_time_preps`` through
    # ``hasattr(type(module), "prepare_scale_operands")`` (``:5999``), which is
    # a per-class opt-in this class does not take. Adding that prep is a
    # separate decision with its own acceptance, and NOT something to smuggle
    # into a forward: it would change what the load path does.
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """One dense layer's MLP. THREE dispatches, and a clamped SwiGLU.

        ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))``, where ``L`` is
        ``self.swiglu_limit``, the checkpoint's own bound resolved from the
        config when this object was built. Each of the three projections is a
        separate entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`, which
        is the dense blockwise route this campaign registered for the dense MLP
        and the shared expert alike (DECISIONS §77).

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. ``T`` must be a
                whole number of ``TILE_SIZE`` rows -- the seam tiles ``M`` over
                the PSUM partition axis and does not pad
                (``blockwise_fp8_mm.py:239-245``), so padding is the caller's,
                exactly as it is for the shared expert.
            quant_config: the resolved per-model quantisation policy, and the
                route selector. An ARGUMENT rather than a field, because that is
                what all three landed methods of this family do (``:1539``,
                ``:2078``, ``:2373``) and no module in this file holds a policy.

        Returns:
            ``[T, H]`` **fp32** -- the seam's own return dtype, not re-cast here.
            The layer forward decides the residual dtype, which is the same
            division :meth:`Glm5NextSharedExperts.shared_expert_mm` records.

        Raises:
            Glm5NextDenseMLPRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one this
                route consumes, when a scale grid the loader should have
                attached is absent, or when two operand extents contradict each
                other. Named rather than coerced, so a mis-wired call site fails
                where it is wrong instead of computing a different function.
        """
        from torch.nn.functional import silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm
        from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

        # ---- ROUTE SELECTION, the same two refusals the shared expert makes.
        # There is no unquantised dense-MLP path at this site, and continuing
        # anyway is what would reach the substrate's ``QuantizationType.NONE``
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
                f"route consumes scales retiled from ({TILE_SIZE}, {TILE_SIZE}) "
                f"checkpoint blocks onto BLOCK_QUANT_SIZE granularity and has "
                f"no path for any other checkpoint block shape."
            )

        # ---- THE OPERANDS. The grid name is DERIVED by the rule the landed
        # prep loop uses (``_sibling_scale_grid_name``, ``:5542``) rather than
        # spelled out here, so the two cannot drift. The grids are plain
        # attributes and not declared parameters, for the reason recorded on
        # ``_load_out_of_band_scales``, which is why this is a ``getattr``.
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

        # ---- EXTENTS. Only the agreements the seam cannot see: it reads its
        # own extents off each pair of operands separately, so nothing checks
        # that gate and up are the same shape or that down transposes them.
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

        # ---- THE TWO PARALLEL PROJECTIONS. Two entries, two dispatches.
        gate = blockwise_fp8_mm(
            hidden_states, gate_proj_weight, scale_grid("gate_proj_weight")
        )
        up = blockwise_fp8_mm(
            hidden_states, up_proj_weight, scale_grid("up_proj_weight")
        )

        # ---- THE CLAMP. The checkpoint's, not a guard this code invented:
        # ``modeling_glm5_next.py:102`` clamps gate from ABOVE ONLY and ``:103``
        # clamps up on BOTH SIDES, and only then are they multiplied. The
        # asymmetry is the reference's and copying it as a two-sided clamp on
        # both would be a second wrong function, not a tidier one -- the note
        # at ``:2238`` records that reasoning for the shared expert.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # ---- THE SwiGLU WIRING. Call-site plumbing, not authored numerics:
        # ``silu`` is torch's own, the product is elementwise, and both run in
        # the seam's fp32 return dtype so no precision is thrown away between
        # the projections. This is also why the dispatch count cannot be 2 --
        # ``silu`` is non-linear, so ``down`` folds into neither predecessor.
        activated = silu(gate) * up

        # ---- THE DOWN PROJECTION re-enters the seam, whose declared input
        # dtype is ``bfloat16`` (``blockwise_fp8_mm.py:446``), so the fp32
        # intermediate is cast back to the caller's activation dtype here.
        return blockwise_fp8_mm(
            activated.to(hidden_states.dtype),
            down_proj_weight,
            scale_grid("down_proj_weight"),
        )


class Glm5NextDenseMLPRouteError(ValueError):
    """A dense-MLP call this route refuses, named rather than coerced.

    ``inc-glm53f-054a``. At module level, and not nested in the class that
    raises it, because an exception a caller catches belongs in the module
    namespace -- the shape ``inc-glm53f-027`` set for
    :class:`Glm5NextBlockQuantRouteError` and ``-033`` for
    :class:`Glm5NextSharedExpertRouteError`.

    Raised in preference to continuing, because each failure it closes returns
    plausible numbers rather than an error: an unquantised route reaches the
    substrate's ``QuantizationType.NONE`` default by omission, a missing scale
    grid runs an unscaled matmul, and contradicting extents multiply tensors
    the reference never multiplies.
    """


def _build_mlp(text_config: Glm5NextTextConfig, layer_idx: int) -> nn.Module:
    """Dense below ``first_k_dense_replace``, sparse at and above it.

    The same predicate the map uses (``weight_loaders_fp8.py:341-344``), so
    the two sides cannot disagree about which layers carry experts.
    """
    if layer_idx < int(text_config.first_k_dense_replace):
        return Glm5NextDenseMLP(text_config)
    return Glm5NextMoEBlock(text_config)


# ---------------------------------------------------------------------------
# The KDA (``linear_attention``) half. D14 owner: ``inc-glm53f-038`` (M3),
# whose acceptance runs "a 3-layer KDA stack" -- so ``Glm5NextKDALayer`` is
# the decoder layer, and the gated-delta module it holds sits at the map's
# ``self_attn`` path. RE-GROUNDED BY ``inc-glm53f-082``: this header used to
# call ``linear_attn`` the map's path, which it stopped being when
# ``inc-glm53f-078`` measured the family off the published checkpoint index.
# ``linear_attn`` survives below only as ``CACHE_NAME_SUFFIX``, which names
# the KV-cache entry and is not a module path.
# ---------------------------------------------------------------------------


class Glm5NextKDAAttention(nn.Module):
    """Gated-delta linear attention at ``self_attn``.

    Every parameter name here is the landed map's ``KDA_PROJECTIONS`` plus
    ``KDA_BARE_LEAVES``, as ``weight_loaders_fp8.py``'s ``_add_kda_attention``
    emits them: thirteen ``<leaf>_weight`` names and the two bare state tensors
    ``A_log`` and ``dt_bias``, **fifteen in all and not one scale companion**.
    RE-GROUNDED BY ``inc-glm53f-082``: ``inc-glm53f-078`` measured that set off
    the published checkpoint index and retired the six fused ``linear_attn.*``
    names this class used to declare. ``conv1d_bias`` is gone with them because
    the index carries no conv1d bias of any spelling. Cites here name symbols
    rather than line numbers, because the line numbers are what went stale.

    THE CACHE GEOMETRY THIS CARRIES, STATED EXACTLY. A linear-attention layer
    holds a **recurrent state**, not a key/value history. The pin's
    ``LayerSpec`` has no vocabulary for that state -- adding it is
    ``inc-glm53f-015``'s declared surface at M1 -- so the spec this layer
    reports describes its per-head state extent in the pin's existing fields
    and asserts nothing about recurrent-state layout. That limit is recorded
    rather than papered over.
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
        # raises here instead of reaching the gate seam as a default. The seam
        # requires it and holds no copy of it (``gate_clamp.py:253-259``).
        self.gate_lower_bound = float(
            _linear_attn_field(text_config, "gate_lower_bound")
        )
        # The decoder's RMSNorm epsilon, from the config rather than from a local
        # default, on ``inc-glm53f-080``'s precedent at line 889 of this file.
        self.rms_norm_eps = float(text_config.rms_norm_eps)
        # ``NeuronConfig.kda_state_chunk_size`` is declared a tuning dial with
        # "None = let the layer choose" (``neuron_config.py:185-188``), and
        # ``LayerSpec.chunk_size`` already exists at the pin, so the dial is
        # passed through rather than dropped. None by default.
        self.cache_chunk_size = getattr(
            text_config.neuron_config, "kda_state_chunk_size", None
        )

        # The four recurrent-state values ``get_kv_spec`` reports for this
        # family. ``inc-glm53f-015`` declared the four field names and
        # ``inc-glm53f-016`` certified the runner's translation of them; D14
        # gives the VALUES to ``inc-glm53f-038``, because they come from this
        # layer's own state geometry.
        #
        # BOTH PAIRS ARE DERIVED FROM vLLM's OWN CALCULATORS AND NEITHER IS
        # WRITTEN AS A LITERAL. The conv state's extent ORDER is chosen by the
        # environment (``VLLM_SSM_CONV_STATE_LAYOUT``), so a hand-written pair
        # would be right under one layout and silently transposed under the
        # other -- and a transposition survives a byte reconciliation, which is
        # why ``test_kv_cache_spec.py:31-40`` records the orientation term as
        # load-bearing. The import is function-local because this module holds
        # no vLLM import at module level.
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
        # passed in (``vllm/utils/torch_utils.py:291-295``), so passing
        # ``self.cache_dtype`` keeps ``NeuronConfig.kda_state_dtype``'s override
        # honoured on the conv carrier while the recurrent carrier's float32
        # still comes from the vendor rather than from a local constant.
        (
            self.kda_conv_state_dtype,
            self.kda_recurrent_state_dtype,
        ) = MambaStateDtypeCalculator.kda_state_dtype(self.cache_dtype, "auto")

        # Which axis of ``kda_conv_state_shape`` is the channel axis, read from
        # the same authority that ordered it. The forward slices the conv
        # carrier, so it must not infer the order from the extents.
        # ``is_conv_state_dim_first`` IS the predicate ``_orient_conv_shape``
        # branched on (``vllm/model_executor/layers/mamba/mamba_utils.py:46-48``
        # -- path-qualified because this tree carries a second ``mamba_utils.py``
        # at ``vllm/v1/worker/``), so this is the same reading and
        # not a second inference from the extents. The layout string is kept
        # beside it because a transcript naming ``"SD"`` is checkable and a bare
        # bool is not.
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

    # ----------------------------------------------------------------- #
    # The conv carrier's extent order, handled in one place.             #
    # ----------------------------------------------------------------- #
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
        """The chunk width the chunked seams are entered with.

        ``NeuronConfig.kda_state_chunk_size`` is declared a dial whose ``None``
        means "let the layer choose", so this is where the layer chooses. The
        choice is DERIVED from the two declared bounds it has to satisfy rather
        than picked: the intra-chunk seam needs a power of two in ``[2, 128]``,
        and both chunked seams refuse a chunk-local cumulative gate above
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
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        is_prefill: bool,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """One KDA layer's gated-delta linear attention over ``[T, hidden]``.

        The five landed seams are entered in this order, and each entry is one
        counted dispatch: the short convolution (``inc-glm53f-034``), the gate
        clamp (``inc-glm53f-084``'s re-authored form), then either the two
        chunked seams (``inc-glm53f-035a`` and ``-035b``) or the single-token
        decode seam (``inc-glm53f-036``). This function composes them; it
        implements none of their arithmetic, which is why its substrate class is
        non-kernel-class.

        Args:
            hidden_states: ``[T, hidden]``. One rank's slice of one step's
                tokens.
            conv_state: this layer's conv carrier from the runner bank, shaped
                as ``get_kv_spec`` reports it. READ for the left context of the
                convolution and WRITTEN IN PLACE with the new tail.
            recurrent_state: this layer's recurrent carrier from the runner
                bank, ``[heads, V, K]``. Index 1 is the VALUE extent and index 2
                the KEY extent -- the orientation ``-035b``'s ``final_state``
                and ``-036``'s ``state`` both use. ``V == K == head_dim`` here,
                so that choice cannot be read off the shape and is stated rather
                than implied. Written in place.
            is_prefill: whether this call is a prefill. It selects the
                recurrence route and nothing else.
            chunk_size: overrides the resolved chunk width, for a test that
                needs to name it.

        Returns:
            ``[T, hidden]`` at the input dtype.

        THE PREFILL ARM ENTERS THE CHUNKED SEAMS WITH A ZERO STATE, BY
        CONSTRUCTION. Neither chunked seam accepts an entering state, so a
        prefill starts the recurrence at zero and ``recurrent_state`` is
        OVERWRITTEN by this call rather than read by it. That is this block's
        declared NON-GOAL held as code, not an oversight: carrying a state into
        a chunked prefill is what chunked prefill, prefix caching or
        multi-token prediction would need, and none of the three exists at the
        pin.

        WHOLE CHUNKS GO THROUGH THE CHUNKED PAIR AND THE REMAINDER WALKS. A
        prefill of ``T = n * chunk + r`` tokens takes one intra-chunk and one
        inter-chunk dispatch for the ``n`` whole chunks together -- their inputs
        carry the chunk axis, so ``n`` chunks cost one dispatch each -- and then
        ``r`` single-token decode dispatches for the remainder. A decode call
        takes no chunked dispatch at any token count.

        The imports are function-local because this module's import block is
        another increment's section (D14).
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

        # --- seam 1: the short convolution, ONE dispatch for q, k and v ------
        # The three streams are convolved together as one channel block, which
        # is the same channel extent the state calculator reports
        # (``conv_dim = proj + 2 * proj_k``). Padding is carried by the state
        # rather than by the seam, which refuses non-zero width padding.
        conv_in = torch.cat((q_in, k_in, v_in), dim=-1)
        padded = torch.cat((self._conv_history(conv_state), conv_in), dim=0)
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
        self._store_conv_history(conv_state, padded[padded.shape[0] - state_rows :])

        # --- seam 2: the gate clamp, one dispatch per head per token tile -----
        # The seam takes ONE head per call: it refuses an ``a_log`` holding more
        # than one value, because the decay rate is per head while the bias and
        # the gate are per key channel.
        #
        # It also refuses more than ``GATE_MAX_TILE`` tokens in one call, because
        # both of its axes pass through a transpose that serves that width. So a
        # prompt longer than one tile is handed over in tiles and reassembled
        # here, in the caller. The seam keeps its refusal rather than growing a
        # quiet fallback: tiling is a caller's concern, and a kernel that stretched
        # its own bound would be the torch-level fallback the substrate rule
        # forbids.
        #
        # Tiling is exact here, not approximate. The gate applies a per-channel
        # bias, one scalar decay rate and a sigmoid, with NO reduction along the
        # token axis, so tile boundaries cannot move a value: a tiled call and a
        # whole call agree bit for bit.
        a_log = self.A_log.to(torch.float32).reshape(-1)
        dt_bias = self.dt_bias.to(torch.float32).reshape(-1)

        def clamp_one_head(h: int) -> torch.Tensor:
            """One head's gate over the whole prompt, in tiles the seam accepts."""
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
            # A prompt that already fits is returned as the single tile it is, so
            # it still costs exactly one dispatch and no concatenation.
            return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)

        gate_parts = [clamp_one_head(h) for h in range(heads)]
        beta = torch.sigmoid(raw_beta)

        # --- seams 3 to 5: the recurrence, per head --------------------------
        chunk = self._resolve_chunk_size(chunk_size)
        n_chunks = tokens // chunk if is_prefill else 0
        chunked = n_chunks * chunk
        core = torch.empty(tokens, width, dtype=torch.float32)
        for h in range(heads):
            span = slice(h * kdim, (h + 1) * kdim)
            q_h = q_conv[:, span].contiguous()
            k_h = k_conv[:, span].contiguous()
            v_h = v_conv[:, span].contiguous()
            gk_h = gate_parts[h]
            beta_h = beta[:, h].contiguous()

            if is_prefill:
                state = torch.zeros(kdim, kdim, dtype=torch.float32)
            else:
                state = recurrent_state[h].to(torch.float32)

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
                core[t, span] = step.o.reshape(-1)
                state = step.state

            recurrent_state[h] = state.to(recurrent_state.dtype)

        # --- gated output norm, then the output projection -------------------
        # ``rmsnorm(core) * sigmoid(out_gate)``, normalised over the head extent
        # because the norm gain is one value per key channel. The reference
        # builds this half as a gated RMSNorm whose activation is sigmoid
        # (``kimi_gdn_linear_attn.py:219``), which is why the raw gate is passed
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
        return attn_out.to(hidden_states.dtype)


class Glm5NextKDALayer(nn.Module):
    """Decoder layer on the ``linear_attention`` half of the hybrid stack.

    Section reserved for ``inc-glm53f-038`` (KDA layer + runner state
    plumbing).
    """

    #: Attribute the family's attention module is bound to -- the map's own
    #: module path, which ``weight_loaders_fp8.py``'s ``_add_kda_attention``
    #: builds as ``f"{param_prefix}.self_attn"``. RE-GROUNDED BY
    #: ``inc-glm53f-082``, which moved this off ``linear_attn`` because
    #: ``declared_parameter_names`` builds its paths from ``named_modules()``.
    #: **NOT the same string as ``Glm5NextKDAAttention.CACHE_NAME_SUFFIX``**,
    #: which stays ``linear_attn`` and names a KV-cache entry, not a module.
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
            # The six mHC weights sit FLAT ON THE LAYER because that is where
            # the map puts them: ``MHC_LEAVES``, emitted for every layer by an
            # unconditional ``_add_mhc`` as ``f"{param_prefix}.{leaf}"`` -- no
            # ``.weight`` leaf, no scale companion, no submodule. A later
            # increment that binds a ``Glm5NextHyperConnection`` instance keeps
            # them here at layer level; moving them under a submodule attribute
            # reddens the map equality, and re-opening the map is the lead's
            # call rather than that increment's.
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

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-attention RMSNorm: ``x / sqrt(mean(x**2) + eps) * gain``.

        A method rather than a module-level helper, because this file's
        module-level region is another increment's D14 section. The epsilon is
        the checkpoint's ``rms_norm_eps``, resolved at construction on the same
        ground ``inc-glm53f-080`` states at line 889.
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
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        is_prefill: bool,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Pre-norm, then the linear-attention half, then the residual add.

        WHAT THIS FORWARD DELIBERATELY DOES NOT DO, AND WHY IT IS NOT A GAP.
        A finished decoder layer also runs its feed-forward half and its
        hyper-connection mixing. Neither is reachable at this milestone and
        neither is this increment's to write:

        * ``self.mlp`` raises whichever branch ``_build_mlp`` chose --
          ``Glm5NextDenseMLP.forward`` and ``Glm5NextMoEBlock.forward`` are both
          still stubs -- and those sections belong to ``inc-glm53f-031`` through
          ``inc-glm53f-033``.
        * the six mHC weights sit flat on this layer but no
          ``Glm5NextHyperConnection`` instance is bound to it yet, and that
          wiring is ``inc-glm53f-030``'s section.

        D14 tells an implementer whose increment would have to touch a class
        outside its own section to raise that rather than widen its surface, so
        this forward stops at the attention half and ``inc-glm53f-054`` joins the
        halves when it writes the 45-layer forward.

        Args and returns are the attention module's, passed through unchanged;
        see :meth:`Glm5NextKDAAttention.forward` for what the two carriers mean.
        """
        residual = hidden_states
        normed = self._input_norm(hidden_states)
        attn_out = self.attention(
            normed,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=is_prefill,
            chunk_size=chunk_size,
        )
        return residual + attn_out


# ---------------------------------------------------------------------------
# The DSA (``deepseek_sparse_attention``) half. D14 owners:
# ``inc-glm53f-039`` (MLA projections) and ``-042`` (MLA decode path) inside
# ``Glm5NextMLAAttention``; ``-051`` for ``Glm5NextDSALayer``, whose acceptance
# runs "a 3-layer DSA stack" -- so that class is the decoder layer and the MLA
# module sits at ``self_attn``.
#
# THIS COMMENT USED TO SAY "layer + sequence tiling", and the second half was
# wrong rather than merely stale: there is NO DSA sequence-tiling seam in this
# fork to integrate. Removed at design entry ``design-20260905-am`` (plan
# revision 223), which cut ``-051``'s acceptance from eight counted runs to
# two. The tiling question belongs to ``-052``'s untiled-S ladder.
#
# THE RUNNER HALF IS ALSO NOT ``-051``'s, as of design entry
# ``design-20260905-an`` (plan revision 224): ``-051`` DECLARES the layer-side
# interface and ``-054`` threads it, because two ``-054``-owned stubs sit
# between the runner and ``Glm5NextDSALayer.forward``.
# ---------------------------------------------------------------------------


class Glm5NextDSAIndexerError(ValueError):
    """Raised when the DSA indexer is asked for something it cannot serve.

    ``inc-glm53f-051``. Declared beside the class that raises it, the convention
    every sibling route error in this file already follows. NO COUNT OF THOSE
    SIBLINGS IS WRITTEN HERE ON PURPOSE: ``Glm5NextMLADecodeError`` below says
    "the three sibling route errors" and there are more than three now, which is
    what a typed count does as a file grows.

    A DISTINCT TYPE RATHER THAN A BARE ``ValueError``, for the reason the sibling
    ``Glm5NextMLADecodeError`` gives: the acceptance asserts that a refusal is
    raised BY NAME, and a bare ``ValueError`` cannot tell a refusal this section
    owns from one torch raised on its way through. The indexer's refusals are
    geometry and materialisation checks on operands it did not load, so they must
    be distinguishable from a defect inside a seam.
    """


class Glm5NextDSAIndexer(nn.Module):
    """The DSA sparse indexer at ``self_attn.indexer``.

    LEAF NAMES PROVISIONAL AT THIS INCREMENT, and not any more: the map now
    records ``dsa_indexer`` as GROUNDED (``weight_loaders_fp8.py:84-86``) because
    ``inc-glm53f-078`` read the real shard index. The dials it is sized from,
    ``index_n_heads`` / ``index_head_dim``, USED to live only in the fixtures and
    never in fork Python; ``inc-glm53f-051`` made all seven ``index_*`` keys typed
    ``Glm5NextTextConfig`` fields (entry ``design-20260905-ad``), so this class
    reads them from the config like every other width in this file.

    WHAT THIS CLASS OWNS, since it now owns arithmetic. ``inc-glm53f-051`` gives it
    the four projections of the indexer chain, its key normalisation and its
    forward. The four projections run on the landed ``mla_projection`` seam and
    their weights are prepared ONCE per instance, which is the same division of
    labour the sibling ``Glm5NextMLAAttention`` PROJECTIONS section states and the
    same one the reference's own indexer applies -- it caches its transposed
    weights-projection half on the indexer module rather than transposing per call
    (``vllm/models/glm5next/nvidia/attention.py:326-332`` at pin ``878631b6``).
    """

    #: Attribute the prepared weights are cached on. A plain attribute and NOT a
    #: buffer, for the reason the sibling section gives: a buffer enters
    #: ``state_dict()``, which would double every prepared weight in a saved
    #: checkpoint.
    PREPARED_WEIGHTS_ATTR = "_prepared_indexer_weights"

    #: Site name -> the parameter attribute that site's weight arrives on.
    #:
    #: WHY THIS MAP EXISTS AT ALL, and why the sibling section needs no such thing.
    #: There, every site's weight is ``f"{name}_weight"`` and the convention can be
    #: left implicit. Here THREE of the four follow that convention and the fourth
    #: does not: ``index_kpool_compress_gate`` is a bare checkpoint tensor with no
    #: ``.weight`` leaf (``weight_loaders_fp8.py``, the ``_add_dsa_attention``
    #: mapping), so a name-plus-suffix rule would look for a parameter that does
    #: not exist. Stating all four here makes the exception visible instead of
    #: hiding it in a fallback.
    #:
    #: ONE CONSEQUENCE, DISCLOSED. The model's production prep walk pre-flights
    #: each operand by building ``name + _WEIGHT_LEAF_SUFFIX`` and CONTINUES past a
    #: ``None``, so it covers three of these four sites and silently skips the
    #: fourth. It does not fail. :meth:`prepare_projection_weights` below therefore
    #: performs that site's own absent and placeholder checks, so the skip does not
    #: become a gap.
    PROJECTION_PARAMETERS: dict[str, str] = {
        "wq_b": "wq_b_weight",
        "wk": "wk_weight",
        "weights_proj": "weights_proj_weight",
        "index_kpool_compress_gate": "index_kpool_compress_gate",
    }

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        # The four dials this class computes with, each read from the config
        # rather than transcribed. ``inc-glm53f-051`` added them as fields; before
        # that the adapter dropped them and there was nothing to read.
        self.hidden_size = int(text_config.hidden_size)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.index_n_heads = int(text_config.index_n_heads)
        self.index_head_dim = int(text_config.index_head_dim)
        self.index_kpool = int(text_config.index_kpool)
        self.index_topk = int(text_config.index_topk)
        # BOTH compress dials are read, and neither is a switch: the forward
        # REFUSES any value but True for either (:meth:`require_dials`). They are
        # preconditions, so no branch in this class serves False.
        self.index_kpool_compress = bool(text_config.index_kpool_compress)
        self.index_kpool_always_select_tail = bool(
            text_config.index_kpool_always_select_tail
        )

        # The map's seven, in the map's own order: the four scaled projections,
        # then ``k_norm_bias``, then the two bare compress tensors.
        # ``inc-glm53f-082`` replaced the provisional ``wq_weight`` with the
        # ``wq_b_weight`` the checkpoint actually carries.
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
        code used -- the reason the sibling section's own method gives.

        ``index_n_heads * index_head_dim`` and ``hidden_size`` are both 4096 on this
        checkpoint AND THEY ARE NOT THE SAME QUANTITY: the first is the indexer's
        total head width and the second is the model's residual width. Each is
        written from its own dial below so the coincidence cannot be mistaken for
        an identity by a later reader or by a checkpoint that separates them.

        ``index_kpool_compress_gate`` is a projection here because that is what it
        is: the reference computes the per-token pool gate as
        ``F.linear(hidden_states, index_kpool_compress_gate)`` with the weight
        shaped ``[index_head_dim, hidden_size]``
        (``vllm/models/glm5next/nvidia/attention.py:380`` at pin ``878631b6``), so
        it contracts the hidden width exactly as the other three do.
        """
        return (
            ("wq_b", self.q_lora_rank, self.index_n_heads * self.index_head_dim),
            ("wk", self.hidden_size, self.index_head_dim),
            ("weights_proj", self.hidden_size, self.index_n_heads),
            ("index_kpool_compress_gate", self.hidden_size, self.index_head_dim),
        )

    def prepare_projection_weights(self) -> int:
        """Transpose the four indexer weights ONCE. Returns how many.

        SAME REASON AS THE SIBLING SECTION'S, restated only where this class
        differs. ``mla_projection`` needs each weight contraction-major,
        ``[in_features, out_features]``; a checkpoint stores the torch orientation,
        ``[out_features, in_features]``. A projection weight never changes, so it
        is transposed here once rather than on every call.

        THIS METHOD IS FOUND BY DUCK TYPE, not by a new call site. The model's
        production prep walk calls ``prepare_projection_weights`` on any module
        whose TYPE declares it, so binding this class under the attention module is
        all the wiring it needs and no load-path code is touched.

        IT CARRIES ITS OWN ABSENT AND PLACEHOLDER CHECKS, and that is not
        duplication. The walk's pre-flight builds each attribute name by appending
        ``_weight`` and continues past a ``None``, so it cannot see this class's
        bare ``index_kpool_compress_gate`` leaf at all. Checking here means all
        four sites are checked, whoever called.

        Each weight is checked against the closed-form widths BEFORE it is
        transposed, so a checkpoint whose indexer geometry differs fails here with
        the site named -- rather than reaching a seam that would accept the
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
            # ``.t()`` alone is a view and the kernel loads from memory, so the
            # copy is forced here -- once -- rather than left for the seam.
            prepared[name] = weight.to(torch.float32).t().contiguous()
        setattr(self, self.PREPARED_WEIGHTS_ATTR, prepared)
        return len(prepared)

    def _prepared_weight(self, name: str) -> torch.Tensor:
        """One prepared weight, or a refusal naming what was not done.

        Refusing is what makes "never per call" checkable, the reason the sibling
        section's own accessor gives: a fallback that transposed on demand would
        bring back the per-call copy this preparation exists to remove, and nothing
        would report it.
        """
        prepared = getattr(self, self.PREPARED_WEIGHTS_ATTR, None)
        if not prepared:
            raise Glm5NextDSAIndexerError(
                "prepare_projection_weights() has not run; the indexer's "
                "projection weights are transposed once at load time, never per "
                "call"
            )
        return prepared[name]

    #: Epsilon of the indexer's key normalisation. A MODULE CONSTANT and not a
    #: config read, because this fork's config carries no field for it: the
    #: adapter keeps only declared dataclass fields and drops the rest
    #: (``config.py:103`` and ``:121``), and no indexer dial survives that filter.
    #: The value is the reference's, read at the pin rather than defaulted here --
    #: ``LayerNorm(self.head_dim, eps=1e-6)`` at upstream
    #: ``vllm/models/glm5next/nvidia/attention.py:267`` (blob at ``878631b6``),
    #: which its own DeepSeek-V3.2 ancestor states identically at
    #: ``vllm/models/deepseek_v32/attention.py:84``. NOT ``rms_norm_eps``: that is
    #: this checkpoint's 1e-05 for the RMS norms, a different constant on a
    #: different norm, and reusing it would be a silent substitution.
    #: Authorised as a module constant by the lead at design entry
    #: ``design-20260905-ab``, on the ``_latent_norm`` precedent.
    KEY_NORM_EPS = 1e-6

    def _key_norm(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """LayerNorm on the indexer key: ``[tokens, index_head_dim]`` in, same out.

        TORCH GLUE ON PURPOSE, and the ruling says which kind. This is elementwise
        work with one reduction over a 128-wide axis, no landed ``functional/``
        seam performs a standalone LayerNorm, and P13 keeps torch legitimate for
        glue. The sibling ``_latent_norm`` in this file is the precedent for the
        shape of this method: a method rather than a module-level helper, because
        this file's module-level region is another increment's D14 section.

        A LAYERNORM AND NOT AN RMSNORM, and the checkpoint is what says so. The
        weight map binds ``k_norm_bias`` beside ``k_norm_weight``
        (``weight_loaders_fp8.py``), and an RMS norm has no bias to bind. So the
        mean is subtracted here; treating this as the sibling RMS norm would drop
        that subtraction and compute a different function at exactly the right
        shape.

        THE FP32 CAST IS THE REFERENCE'S, not a local preference. Upstream
        normalises in float32 and casts back to the input dtype in one fused step
        (``_fused_indexer_k_norm``, ``attention.py:54-58`` at the pin). The cast
        back matters because the value returned here feeds seams whose gates admit
        bf16 only, and a float32 key would take a torch oracle instead of the NKI
        route while every shape still checked out.
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
        """The single constant folded into ``weights`` before the score GEMM.

        Upstream folds TWO factors into one constant and says so in the helper's
        own comment -- *"scale folds softmax_scale (head_dim**-0.5) and
        n_head**-0.5 into a single constant"*
        (``_fused_indexer_weight_scale``, ``attention.py:62-68`` at pin
        ``878631b6``, applied ``:373-375``). Both come from this class's own
        dials, so nothing is minted here and no caller passes a scale in.

        NOT THE LAYER'S ``softmax_scale``, and the two must not be confused.
        That one is the MLA attention's and reaches ``attend()`` as a caller
        argument; this one is the INDEXER's, and upstream's ``softmax_scale``
        at the cite above is its own ``index_head_dim ** -0.5``.

        THE QUERY SCALE IS ABSENT AND THAT IS THE FORK'S CHOICE, not an
        omission here. Upstream's third factor is the fp8 quantisation scale
        from ``fwht128_quant_fp8`` (``:369``); this fork's rotation seam
        ``dsa_hadamard128`` returns a plain tensor, and
        ``kpool_hadamard.py:57`` and ``decode_tail_update.py:85`` each record
        fp8 and the ue8m0 scale as out of scope with no plan block owning
        them. So a reference for this chain must fold two factors, not three.
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

        ``q_latent`` is the NORMALISED query latent, which upstream's indexer
        also takes as an argument rather than computing
        (``attention.py:317``, consumed ``:319``). In this fork it comes from
        ``Glm5NextMLAAttention.project_query_latent``.

        FIVE DISPATCHES, and which seam takes each. Four go to
        ``mla_projection`` -- one per site in :meth:`projection_widths` -- and
        one to ``dsa_hadamard128`` for the rotation. The four are the ruled
        substrate (design entry ``design-20260905-ab``): the increment stays
        NON-KERNEL-CLASS because every accelerator step here reaches a landed
        kernel, and none of them authors a torch substitute for kernel work.

        THE DTYPE LADDER IS FORCED BY THE SEAMS' OWN GATES, not chosen.
        ``mla_projection`` returns float32 by contract. ``dsa_hadamard128``
        and the pooling seams take the NKI route on bf16 only
        (``kpool_hadamard.py:147``), and ``dsa_score_gemm`` wants bf16 queries
        with float32 weights (``score_gemm.py:149`` and ``:158``). So the key,
        the query and the gate are cast to bf16 and the weights stay float32 --
        a float32 key would pass every shape check and silently take a torch
        oracle instead of the kernel.

        WHAT IS DELIBERATELY ABSENT, both because this checkpoint does not
        reach it. RoPE: upstream skips the split, the rotary and the cat
        entirely when ``rope_dim == 0`` (``:339-361``) and this checkpoint's
        ``qk_rope_head_dim`` is 0. Head zero-padding: upstream pads only when
        ``n_head < 32`` (``:383-389``) and this checkpoint has exactly 32.
        """
        # Fully-qualified module imports, which is this file's own form for a
        # `functional/` seam and NOT a package-level one: the DSA package's
        # `__init__.py` is EMPTY (0 bytes at this tip), so a package-level
        # import of a seam name would raise ImportError at the first call.
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

        # 1. The query. Upstream views the projection to [-1, n_head, head_dim]
        #    (`:319-320`); the rotation seam takes a 2-D [rows, 128], so the
        #    view is flat for the call and restored after it.
        query = mla_projection(q_latent.to(torch.float32), self._prepared_weight("wq_b"))
        query = dsa_hadamard128(
            query.reshape(tokens * self.index_n_heads, self.index_head_dim).to(
                torch.bfloat16
            )
        ).reshape(tokens, self.index_n_heads, self.index_head_dim)

        # 2. The key, then its LayerNorm. The cast to bf16 happens BEFORE the
        #    norm so the norm's own cast-back lands on bf16, which is
        #    upstream's order: its `k` leaves a bf16 linear and
        #    `_fused_indexer_k_norm` normalises in float32 and casts back
        #    (`:335-337`, helper `:54-58`).
        key = self._key_norm(
            mla_projection(hidden_f32, self._prepared_weight("wk")).to(torch.bfloat16),
            self.k_norm_weight,
            self.k_norm_bias,
        )

        # 3. The weights, with the single constant folded in. float32 all the
        #    way: `dsa_score_gemm` requires it and applies no scale of its own.
        weights = mla_projection(
            hidden_f32, self._prepared_weight("weights_proj")
        ) * self.projection_scale()

        # 4. The kpool gate. Upstream's `F.linear(hidden_states,
        #    index_kpool_compress_gate)` (`:380`), which is why
        #    `projection_widths` carries this site as a projection.
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
        position, and ``write_mask`` is ``[tokens]`` bool marking which of
        those positions actually completed a pool. The caller writes only the
        masked rows, at ``slot_mapping``'s slots.

        TRANSCRIBED FROM THE REFERENCE THIS INCREMENT IS POINTED AT.
        ``_kpool_compress_insert`` (``sparse_attn_indexer_kpool.py:54-99`` at
        pin ``878631b6``) is named as this increment's integration surface by
        the plan's rev-169 rider, and its shape is reproduced rather than
        improved on: EVERY position is treated as a pool-completion candidate
        and the non-completions are masked, because -- in the reference's own
        words -- compacting the valid rows first *"costs two device syncs on
        the eager prefill path and buys nothing numerically"*.

        ``slot_mapping`` IS POOL-GRANULAR, which is the whole reason the mask
        exists: only the LAST token of each complete pool carries a
        non-negative slot and intra-pool positions carry ``-1``
        (``:64-68``). The second term, ``pos >= index_kpool - 1``, drops a pool
        whose start falls before the batch -- leading padding, whose gate and
        key data the reference calls undefined.

        THE SEAM SEES NO PARTIAL POOL, and that is a landed contract rather
        than a convention: ``dsa_kpool_hadamard`` asserts complete pools, and
        the plan's rev-169 correction records that the caller masks
        non-completions and persists the raw remainder to the tail cache. The
        remainder is the decode ring's, ``dsa_decode_tail_update``, not this
        method's.

        ONE DISPATCH, on ``dsa_kpool_hadamard``. The sliding window is index
        arithmetic and carries no dispatch of its own.
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
                f"(sparse_attn_indexer_kpool.py:76-77) and this caller refuses "
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
        position: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Advance the decode ring by one token. Returns ``(pooled, new_tail)``.

        ``pooled`` is ``[1, index_head_dim]`` when this token COMPLETED a pool
        and ``None`` when it did not; ``new_tail`` always has ``tail``'s shape.

        THE RING IS STATE AND THIS METHOD DOES NOT OWN IT. The seam's own
        docstring is explicit -- *"The caller threads ``new_tail`` into the next
        step -- the ring is state, and this function does not mutate its
        argument"* -- so the returned ring goes back to the caller, exactly as
        ``attend()`` takes its latent cache as an argument rather than holding
        one. This method allocates nothing.

        WHY THIS IS THE DECODE COUNTERPART OF :meth:`pool_window` AND NOT A
        SPECIAL CASE OF IT. A prefill sees whole pools and pools them in one
        call; a decode step sees ONE token and must remember the pool it is
        part-way through, which is what the ring holds -- half 0 keys, half 1
        gate scores, upstream's own layout. The plan's rev-169 correction
        assigns the raw remainder to this ring rather than to the pooling seam,
        which *"NEVER SEES A PARTIAL POOL"*.

        ``position`` IS THE TOKEN'S ABSOLUTE POSITION and a python int, not a
        tensor, for the reason the seam's module docstring gives; it decides
        which ring row is written and whether the pool completes.

        ONE DISPATCH, on ``dsa_decode_tail_update``, and it is the only seam in
        this chain that fires on the decode leg and not the prefill leg.
        """
        from vllm_neuron.functional.dsa.decode_tail_update import (
            dsa_decode_tail_update,
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
        if int(position) < 0:
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

        return dsa_decode_tail_update(
            tail, key, gate_score, ape.to(torch.float32), int(position)
        )

    def score_pools(
        self,
        query: torch.Tensor,
        candidate_keys: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate pool against every token. ``[tokens, cands]`` fp32.

        ``query`` is ``[tokens, index_n_heads, index_head_dim]`` bf16 and
        ``candidate_keys`` is ``[cands, index_head_dim]`` -- ONE key per
        candidate, shared across heads, which is this checkpoint's MQA shape.

        THE CANDIDATE AXIS IS POOL-GRANULAR, not token-granular, and the seam
        says so from upstream: *"logits are pool-granular (compress_ratio ==
        index_kpool)"* (``sparse_attn_indexer_kpool.py:551-554``, quoted in
        ``score_gemm.py``). That is why the next stage selects POOLS and why an
        expansion is needed afterwards to reach tokens.

        ``weights`` ARRIVES ALREADY SCALED and this method adds nothing. The
        seam applies no scale of its own by contract, and the constant is
        :meth:`projection_scale`, folded in by :meth:`project_stage`. Folding it
        twice would square it, which is exactly the kind of defect a
        scale-carrying argument invites -- so the fold has ONE site and this
        docstring names it.

        ONE DISPATCH, on ``dsa_score_gemm``.
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
        """How many POOLS the selection keeps. Derived in ONE place.

        ``index_topk`` counts TOKENS, while the selection on this checkpoint is
        pool-granular -- ``sparse_attn_indexer_kpool.py:551-554`` at pin
        ``878631b6`` reads *"logits are pool-granular (compress_ratio ==
        index_kpool)"*. So the token budget divides by the pool width before it
        reaches ``dsa_topk_select``. Neither number is spelled here; both are
        config dials, and the campaign's earlier rounds show what typing a count
        instead of computing it costs.
        """
        return self.index_topk // self.index_kpool

    def require_dials(self) -> None:
        """Refuse the two compress dials this class does not serve.

        WHY A REFUSAL AND NOT A BRANCH. The plan's rev-167 rider, re-issued at
        entry ``design-20260905-af``, owns ``index_kpool_always_select_tail``
        here as a PRECONDITION and a READING and never as a step: the landed
        ``dsa_index_expand`` IS upstream's fused expand-plus-tail and appends the
        tail itself, deriving ``tail_start`` internally
        (``kpool_compress.py:762``, ``:860-871``;
        ``sparse_attn_indexer_kpool.py:598``, ``:883``). So there is no
        force-include step for this class to author, and a ``False`` would need a
        different expansion that no landed seam provides.

        WHAT THE TAIL ACTUALLY IS, because an earlier draft of this docstring
        called it "the tail pooled block" and that was loose on two counts.
        (1) The appended entities are RAW TOKEN indices, not a pool id: the seam
        adds ``pool_size - 1`` columns holding ``tail_start + t`` with
        ``tail_start = seq_len // pool_size * pool_size``, one column per
        possible position in the incomplete final pool (``index_expand.py:9``,
        ``:48-50``, ``:428-439``). Every other column of the row carries pool
        ids expanded to tokens; these carry individual tokens directly.
        (2) The append is EMPTY whenever ``seq_len % pool_size == 0``, because
        ``tail_count = seq_len - tail_start`` is then zero and every tail column
        masks to ``-1``. So "force-includes" never means "always adds a token" --
        a sequence that divides evenly into pools contributes no tail token at
        all, and that is the common case at a pool-aligned length rather than an
        edge case. The dial still binds: it selects an expansion SHAPE that
        reserves those columns, not a guarantee that they are populated.

        UPSTREAM REFUSES THE SAME TWO VALUES, so this is a transcription and not
        a local policy: ``transformers_utils/configs/glm5_next.py:118-126`` at
        pin ``878631b6`` raises ``NotImplementedError`` for each. Upstream tests
        ``is not True`` rather than falsiness; both dials are stored through
        ``bool()`` at construction, so the two tests cannot diverge here.

        AND WRITING A TORCH PATH FOR ``False`` WOULD BE A P13 DEFECT -- a
        torch-level fallback for work the kernel owns.
        """
        if not self.index_kpool_compress:
            raise Glm5NextDSAIndexerError(
                "index_kpool_compress must be True; this indexer serves only "
                "the compressed, pool-granular path, because the landed "
                "dsa_index_expand seam expands POOL ids and no landed seam "
                "selects token-granular candidates. Upstream refuses the same "
                "value (transformers_utils/configs/glm5_next.py:118-126)"
            )
        if not self.index_kpool_always_select_tail:
            raise Glm5NextDSAIndexerError(
                "index_kpool_always_select_tail must be True; the landed "
                "dsa_index_expand seam appends the tail itself -- pool_size - 1 "
                "columns carrying the raw token indices of the incomplete final "
                "pool, empty when seq_len % pool_size == 0 -- and derives "
                "tail_start internally, so False would need an expansion no "
                "landed seam provides. Upstream refuses the same value "
                "(transformers_utils/configs/glm5_next.py:118-126)"
            )

    def select_pools(self, scores: torch.Tensor) -> torch.Tensor:
        """Keep the ``select_k`` highest-scoring pools. ``[rows, select_k]`` int32.

        THE ONE CAST BETWEEN TWO SEAMS THAT DISAGREE, and it belongs here.
        ``dsa_topk_select`` returns int64 indices *"to match ``torch.topk``"*
        (``topk_select.py:302-317``), while ``dsa_index_expand`` admits int32
        ONLY and says why -- *"int64 indices would double the SBUF traffic for a
        range no sequence length reaches"* (``index_expand.py:139-141``). So
        exactly one cast is needed and this is its single site.

        WHY THIS METHOD DOES NOT RE-CHECK THE STRICT BOUND. ``dsa_topk_select``
        needs ``0 < k < width`` STRICTLY, and at ``k == width`` its gate RETURNS
        FALSE rather than raising (``topk_select.py:294``) -- which takes the
        torch route and increments ``torch_fallback``, breaking a declared zero
        SILENTLY, unlike the loud raise at ``k > width`` (``:320-323``). That is
        finding F10, and both entry points ROUTE the regime away before a score
        exists -- ``_require_serviceable`` reports ``selects=False`` and they
        return ``inc-glm53f-099``'s causal bypass -- so the bound already holds
        by the time this runs. Checking it in two places would let the two
        disagree. Until ``inc-glm53f-099`` the same guarantee came from a refusal
        here rather than a route; the guarantee is the same one and this method
        is unchanged by the swap.

        ONE DISPATCH, on ``dsa_topk_select``.
        """
        from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

        if scores.ndim != 2:
            raise Glm5NextDSAIndexerError(
                f"scores must be [rows, cands] from score_pools; got "
                f"{tuple(scores.shape)}"
            )
        _values, indices = dsa_topk_select(scores, self.select_k())
        return indices.to(torch.int32)

    def expand_indices(
        self, pool_ids: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Selected pools expanded to token indices, tail appended. ONE dispatch.

        Returns ``[rows, select_k * index_kpool + index_kpool - 1]`` int32 token
        indices in upstream's column order, where ``-1`` means *"this column
        selects no token"* and is a VALUE rather than an out-of-bounds index.

        THE ``-1`` LEAVES THIS CLASS UNTOUCHED, which is a ruling and not a
        preference. Entry ``design-20260905-af`` took route (a): the mask goes
        INSIDE the kernel, where ``inc-glm53f-098`` masks ``-1`` in
        ``mla_sparse_attention`` and narrows its gate to ``lo < -1``. So this
        class applies no filler, no compaction, no clamp and no mask, and hands
        the sentinel-bearing tensor to ``attend()`` unchanged.

        FINDING F8 IS WHY THE RULING WAS NEEDED. The fork's
        ``mla_sparse_attention`` refuses ANY negative index by name
        (``mla_sparse.py:1259-1267``), while upstream bounds its kernel with a
        SEPARATE per-row valid count instead -- ``sparse_mla_top_k_lens =
        seq_lens.clamp(min=1)``, ``flashinfer_mla_sparse.py:467`` -- a parameter
        the fork's seam does not have. The sentinel is upstream's DESIGN, not an
        accident: it initialises its whole index buffer to ``-1``
        (``sparse_attn_indexer_kpool.py:435``) and writes the ``-1``-bearing
        expansion straight into it (``:606``, ``:892``).

        AND NO FILLER WOULD HAVE BEEN FREE, which is why route (a) is not merely
        tidier. Softmax normalises over the columns it is given, so any in-range
        filler duplicates a real cache row and takes real probability mass.
        Upstream can point its empty rows at slot 0 ONLY because it also clamps
        their length to 1 and zeroes the output afterwards
        (``flashinfer_mla_sparse.py:467``, ``:497-499``).
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

        ONE IMPLEMENTATION FOR BOTH ENTRY POINTS, which is the point of extracting it:
        :meth:`forward` and :meth:`forward_ragged` must agree about what is serviceable,
        and two copies of a refusal are two things that can drift apart. F10's earlier
        laps were about exactly this class of divergence. ``inc-glm53f-099`` keeps that
        property and adds one answer to it: ``selects`` is the ONE decision, and the two
        entry points make two calls on it.

        ``selects`` IS ``candidates > select_k()``, AND IT IS STRICT. ``dsa_topk_select``
        needs ``0 < k < width`` STRICTLY, and at ``k == width`` its gate RETURNS FALSE
        rather than raising (``topk_select.py:294``) -- taking the torch route and
        breaking the declared ``torch_fallback == 0`` SILENTLY, unlike the loud raise at
        ``k > width`` (``:320-323``). ``width`` is the WIDEST row's complete-pool count,
        because that row sizes the shared candidate axis.

        WHY THIS REPORTS WHERE IT USED TO REFUSE (``inc-glm53f-099``, entry
        ``design-20260906-as``). Below the strict bound there is nothing to select from,
        and upstream does not clamp ``k`` there -- it bypasses selection entirely and
        attends causally (``sparse_attn_indexer_kpool.py:203-217``). ``inc-glm53f-098``'s
        landed refusal named that route and left it unbuilt; it is built now, so the
        regime is SERVED rather than refused and this method reports which of the two
        answers the caller owes. The refusal is gone from here and no refusal replaced it,
        because a served regime has nothing to refuse.

        WHY THE BYPASS CANNOT LIVE IN THIS METHOD, which is the whole shape of the fix.
        Both entry points call this BEFORE they write the pooled-key store and advance the
        decode ring, so returning an answer here would skip the write and corrupt every
        later step. So the DECISION is here, once, and the two DISPATCHES are at the two
        entry points, each after its own write stage.
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
        """The width the short-sequence bypass emits: SELECTION'S OWN width, derived.

        The bypass and the selecting path must hand the sparse attention kernel the same
        shape, or the emitted width would vary by regime and every consumer would have to
        branch. ``dsa_index_expand`` allocates
        ``index_expand_width(n_groups, pool_size)`` (the allocation at
        ``index_expand.py:316``, the helper itself at ``index_expand.py:255-267``), so that
        is read from the seam's own helper here rather than recomputed -- 128 at this file's
        tiny test dials and 2176 at the checkpoint's, both measured.
        """
        from vllm_neuron.functional.dsa.index_expand import index_expand_width

        return int(index_expand_width(self.select_k(), self.index_kpool))

    def _bypass_indices(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """The short regime's answer: each row's plain causal prefix, at :meth:`bypass_width`.

        ONE DISPATCH, called from both entry points, so the two cannot answer the short
        regime differently. Row ``i`` holds ``0 .. seq_lens[i] - 1`` and then ``-1``, which
        is what "attend causally, select nothing" means as an index tensor; the ``-1`` is
        the sentinel ``inc-glm53f-098``'s sparse kernel masks.

        EVERY CAUSAL COLUMN FITS, with no clamp, and that is arithmetic rather than luck:
        the bypass holds only while ``max_seq_len // pool <= select_k``, so the largest
        position is ``select_k * pool + pool - 2``, which is one below the RAW expansion
        width and therefore below the rounded-up emitted width. Read as a value at both
        dial sets in ``probe-099-widths-host.out`` -- 125 spare columns at the
        checkpoint's, 117 at the tiny dials, and zero admissible lengths that overflow.
        """
        from vllm_neuron.functional.dsa.causal_fill import dsa_causal_fill

        return dsa_causal_fill(seq_lens.to(torch.int32) - 1, self.bypass_width())

    def _gather_candidates(
        self, pool_cache: torch.Tensor, candidates: int, page_size: int
    ) -> torch.Tensor:
        """The candidate pooled keys. ONE dispatch on ``dsa_paged_gather``.

        The candidate axis is every complete pool of the widest row, SHARED across the
        batch -- which is upstream's own shape, a ``[tokens, max_seq_len_pooled]`` logits
        grid rather than a per-row candidate set. Per-row differences enter later, at the
        expansion, where ``dsa_index_expand`` derives each row's tail from that row's own
        ``seq_len``.

        THE PAGE TABLE IS ARITHMETIC, NOT A LOOKUP, and only because of the serving
        constraint: at one sequence per call (the sibling attention's G1) pool ``j`` lives
        at a fixed page and slot, which is the same contract ``attend()`` states as *"slot
        0 through the last written slot"*. A batched page table would be a real lookup and
        is not this increment's.
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
        position: int | None = None,
    ) -> torch.Tensor:
        """The whole indexer chain. Returns ``topk_indices`` and NOTHING ELSE.

        INDICES ALONE IS A RULING. Entry ``design-20260905-af`` route (a) keeps
        the ``-1`` mask inside ``inc-glm53f-098``'s kernel, so no per-row valid
        length leaves this class -- returning one would give a caller a second,
        contradictory way to bound the same kernel. :meth:`expand_indices` states
        the finding behind it.

        Args:
            hidden_states: ``[tokens, hidden_size]``.
            q_latent: ``[tokens, q_lora_rank]`` fp32, from the sibling
                attention's ``project_query_latent``.
            pool_cache: ``[slots, index_head_dim]`` bf16, the pooled-key store,
                WRITTEN IN PLACE for this call's completed pools and then read
                back as the candidate set. Its LAST row is reserved as write
                trash; see below.
            seq_lens: ``[rows]`` int32, upstream's shape, consumed on device by
                the expansion seam.
            max_seq_len: the batch's longest sequence, as a PYTHON INT.
            page_size: rows per page in ``pool_cache``.
            slot_mapping: ``[tokens]`` int32, pool-granular. Prefill only.
            tail: ``[2, index_kpool, index_head_dim]`` bf16 ring, WRITTEN IN
                PLACE. Passing it selects the decode leg.
            position: the decode token's absolute position, a python int.

        WHY ``max_seq_len`` IS A PYTHON INT AND NOT READ OFF ``seq_lens``. The
        obvious ``int(seq_lens.max())`` is a host read of tensor DATA inside a
        region the runner compiles with ``fullgraph=True``, which is a GRAPH
        BREAK -- ``ragged_pack.py`` measured and recorded exactly this and took
        python ints for the same reason (``neuron_model_runner.py:1457-1462``).
        The candidate count also sizes an ``arange``, so it must be a trace-time
        constant regardless. The cost is that consistency with ``seq_lens`` is
        DECLARED by the caller rather than measured here, and the tests assert it
        host-side, outside the traced region.

        WHY THE RING AND THE POOL STORE ARE WRITTEN IN PLACE. ``attend()`` states
        this file's own contract for carried state -- *"The cache is WRITTEN in
        place for those tokens and then READ from slot 0 through the last written
        slot"* -- and following it is what lets this method return indices alone.
        ``dsa_decode_tail_update`` is functional by design (*"the ring is state,
        and this function does not mutate its argument"*), so its new ring is
        copied into the caller's buffer here, at one site.

        WHY THE POOL WRITE USES A TRASH ROW INSTEAD OF A BOOLEAN MASK. Indexing
        with a bool mask produces a DATA-DEPENDENT shape, which is the same graph
        break in another costume. So every position writes, and the positions
        that completed no pool are redirected to ``pool_cache``'s last row. That
        is ``inc-glm53f-045``'s own measured pattern -- it *"sends every padding
        row to a REAL in-range trash row"* rather than masking by going out of
        bounds. Duplicate trash destinations resolve in unspecified order and
        that is harmless: the row is never addressed as a candidate.

        THE SUBSTRATE (P13). Every arithmetic stage is one of the eight landed
        kernel-class DSA seams or the landed ``mla_projection`` seam. What is
        torch here is orchestration and named so a reviewer can check it: shape
        validation, index arithmetic for the gather, one ``index_copy_`` per leg,
        and one ring copy. No torch path computes an indexer value.
        """
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
        query, key, weights, gate_score = self.project_stage(hidden_states, q_latent)

        if is_decode:
            pooled, new_tail = self.tail_step(tail, key, gate_score, int(position))
            # The seam does not mutate its argument, so the ring is threaded
            # back into the caller's buffer here.
            tail.copy_(new_tail)
            if pooled is not None:
                # `pooled is not None` is decided from `position`, a python int,
                # so this branch is a trace-time choice and not a data read.
                pool_slot = torch.full(
                    (1,), int(position) // pool, dtype=torch.int64,
                    device=pool_cache.device,
                )
                pool_cache.index_copy_(0, pool_slot, pooled.to(pool_cache.dtype))
        else:
            pooled, write_mask = self.pool_window(key, gate_score, slot_mapping)
            destination = torch.where(
                write_mask, slot_mapping.to(torch.int64), torch.tensor(
                    trash, dtype=torch.int64, device=pool_cache.device
                )
            )
            pool_cache.index_copy_(0, destination, pooled.to(pool_cache.dtype))

        if not selects:
            # inc-glm53f-099. The write stage above has already landed -- the pool row is
            # stored and, on the decode leg, the ring has advanced -- so returning here
            # loses nothing. Below the strict bound there is nothing to select from.
            return self._bypass_indices(seq_lens)

        candidate_keys = self._gather_candidates(pool_cache, candidates, int(page_size))
        scores = self.score_pools(query, candidate_keys, weights)
        pool_ids = self.select_pools(scores)
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
        """The NON-UNIFORM decode arm. Returns dense ``[tokens, width]`` int32.

        WHY THIS ARM EXISTS AT ALL, and it is not a second code path for its own sake.
        Finding F3: the ragged-pack family is UNREACHABLE at batch one, because upstream
        guards its whole pack region with ``if decode_metadata.requires_padding:``
        (``sparse_attn_indexer_kpool.py:744``) and a batch of one is uniform by
        construction. Entry ``design-20260905-aa`` route (i) answers that with two cases
        whose UNION carries the family readings, and this is the second of them. It is
        called on the indexer DIRECTLY, so no layer and no ``attend()`` is involved.

        WHAT IT PACKS, AND WHY THAT IS EXACTLY ONE THING. Only the query is packed, and
        only because ``dsa_score_gemm`` needs its rows dense and aligned with the gate.
        Packing anything else would be authoring a dispatch to move a counter, which this
        seat has refused three times now and refuses here. Finding F11 is the reason the
        obvious second pack is impossible rather than merely unnecessary: the gate weights
        are fp32 by ``dsa_score_gemm``'s contract (``score_gemm.py:158``) and
        ``dsa_ragged_pack`` admits bfloat16 ALONE (``ragged_pack.py:121``), so a weights
        pack would take the torch route and break the standing ``torch_fallback == 0``.
        The weights are moved by ``index_select`` instead -- torch glue on a narrow fp32
        gate, which preserves every bit of the fold that a bf16 round trip would lose.

        AND NOTHING IS UNPACKED, which is a ruling. Entry ``design-20260905-aj`` route (b)
        makes ``dsa_ragged_unpack``'s reading on this arm a DECLARED 0, covered by
        ``inc-glm53f-045``'s own landed bit-identical round-trip item. The arm's result is
        int32 by ``dsa_index_expand``'s contract, the pack module admits bf16 only, and the
        consumer wants the DENSE form anyway: ``mla_sparse_attention`` gates
        ``topk_indices.ndim != 2`` and raises by name (``mla_sparse.py:1214``, ``:1216``),
        so a padded rank-3 tensor would be refused. There is no dense-to-padded step to
        author.

        MIND THE NAMES, because upstream's are inverted. Upstream's ``pack_seq_triton`` is
        dense-to-padded (its result is named ``padded_q_quant_decode_tokens``,
        ``sparse_attn_indexer_kpool.py:745-760``) and ``unpack_seq_triton`` is
        padded-to-dense (``:889``). The fork's ``dsa_ragged_pack`` is padded-to-dense and
        ``dsa_ragged_unpack`` dense-to-padded. This method is written to the FORK's
        contract; reading either name by upstream habit inverts the arm silently.

        Args:
            hidden_states: ``[batch, max_len, hidden_size]`` -- the PADDED batch.
            q_latent: ``[batch, max_len, q_lora_rank]`` fp32, padded to match.
            pool_cache: ``[slots, index_head_dim]`` bf16, READ ONLY here.
            seq_lens: ``[tokens]`` int32, one per PACKED row.
            lengths: valid rows per request, as python ints.
            max_seq_len: the batch's longest sequence, a python int.
            page_size: rows per page in ``pool_cache``.

        WHY ``lengths`` IS A SEQUENCE OF PYTHON INTS: ``dsa_ragged_pack``'s own contract,
        for the reason its module docstring gives -- the packed length is DERIVED from
        them, and deriving it from tensor data would be a host read inside a
        ``fullgraph=True`` region and so a graph break.

        WHY THIS ARM DOES NOT WRITE THE CACHE OR ADVANCE THE RING, declared rather than
        omitted. Its job is the selection chain on a non-uniform batch; the pooled-key
        store arrives already populated and the ring is the uniform case's business, which
        already reads ``dsa_decode_tail_update`` and ``dsa_kpool_hadamard``. So both of
        those families read 0 here. THE ARM'S DECLARED READINGS, one line per entry point
        because two of the nine share a module and a family name cannot say which half
        moved: ``dsa_ragged_pack`` 1, ``dsa_ragged_unpack`` 0, ``dsa_hadamard128`` 1,
        ``dsa_kpool_hadamard`` 0, ``dsa_paged_gather`` 1, ``dsa_score_gemm`` 1,
        ``dsa_topk_select`` 1, ``dsa_index_expand`` 1, ``dsa_decode_tail_update`` 0. The
        rotation the projection needs is the moving half; the pooling entry point is the
        one this arm never calls, and an earlier draft of this list named the shared module
        instead, which read as a claim about the pooling. Nine named readings, no family
        shorthand, so ``probe-051-arm-authored`` can compare each one to a measurement.

        THOSE NINE ARE THE SELECTING PATH'S, and ``inc-glm53f-099`` added a second path
        that reads differently. On the SHORT regime this arm returns the causal bypass
        after the pack, so ``dsa_paged_gather``, ``dsa_score_gemm``, ``dsa_topk_select``
        and ``dsa_index_expand`` all read 0 and ``dsa_causal_fill`` reads one NKI dispatch
        with no torch fallback. ``dsa_ragged_pack`` 1 and ``dsa_hadamard128`` 1 are
        unchanged, which is the placement's whole point.

        ONE COST, DISCLOSED. :meth:`project_stage` computes the key and the gate score
        that this arm never uses, because it is one dispatch-counted unit whose reading of
        exactly 4 ``mla_projection`` dispatches per indexer call is the declared
        substrate-binding measurement. Two unused projections is the price of that count
        staying stable across both entry points, and it is the right trade -- but it is a
        real cost and it is stated rather than hidden.
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
                f"requires_padding (sparse_attn_indexer_kpool.py:744) -- so "
                f"serving it here would move the pack counter on a case that "
                f"does not need one. Use forward() instead"
            )
        if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != tokens:
            raise Glm5NextDSAIndexerError(
                f"seq_lens must be [tokens] = {(tokens,)}, one per PACKED row, "
                f"which is what the lengths sum to; got {tuple(seq_lens.shape)}"
            )

        # The projections are ROW-WISE, so they commute exactly with row selection:
        # projecting the padded grid and then packing gives bit-for-bit what packing
        # and then projecting would. Padding rows compute values that the pack drops
        # and nothing downstream ever sees. Flattening is a reshape, not a move.
        heads, head_dim = self.index_n_heads, self.index_head_dim
        query, _key, weights, _gate_score = self.project_stage(
            hidden_states.reshape(batch * max_len, -1),
            q_latent.reshape(batch * max_len, -1),
        )

        # THE ONE PACK. The query carries `heads * head_dim` per row, and the seam takes
        # a 3-D `[batch, max_len, width]`, so the head axis folds into the width for the
        # move and unfolds after -- upstream reshapes around its own pack for the same
        # reason (`padded_weights = pack_seq_triton(...).reshape(...)`, `:759-760`).
        packed_query = dsa_ragged_pack(
            query.reshape(batch, max_len, heads * head_dim), checked
        ).reshape(tokens, heads, head_dim)

        # The fp32 gate moves by index_select, NOT through the seam: F11. The row index
        # is built from the lengths, which are python ints, so its shape is a trace-time
        # constant and no tensor data is read on the host.
        keep = [
            b * max_len + r for b, n in enumerate(checked) for r in range(n)
        ]
        packed_weights = weights.index_select(
            0, torch.tensor(keep, dtype=torch.int64, device=weights.device)
        )

        if not selects:
            # inc-glm53f-099, placed AFTER the pack rather than before it, deliberately.
            # This arm has no write stage to protect, so the only thing the placement
            # decides is the pack's counter reading -- and the pack is this arm's whole
            # reason for existing (design-20260905-aa's seventh family). Bypassing before
            # it would zero that reading on the short regime. The packed query it computes
            # is then unused, which is the same disclosed cost as the two unused
            # projections above and is stated for the same reason.
            return self._bypass_indices(seq_lens)

        candidate_keys = self._gather_candidates(pool_cache, candidates, int(page_size))
        scores = self.score_pools(packed_query, candidate_keys, packed_weights)
        pool_ids = self.select_pools(scores)
        return self.expand_indices(pool_ids, seq_lens)


class Glm5NextMLADecodeError(ValueError):
    """Raised when the MLA decode path is asked for something it cannot serve.

    ``inc-glm53f-042``. Declared beside the class that raises it, the convention
    the three sibling route errors in this file already follow.

    A DISTINCT TYPE RATHER THAN A BARE ``ValueError``, for one reason the
    acceptance depends on. The ``B == 1`` serving constraint has to be
    distinguishable from a shape typo, and a test that caught ``ValueError``
    would pass on either -- so the constraint would be asserted without being
    measured. The seams this path calls raise their own named errors for the same
    reason (``MlaAbsorbError``, ``MlaSparseAttentionError``).
    """


class Glm5NextMLAAttention(nn.Module):
    """Multi-head latent attention at ``self_attn``, NoPE on this checkpoint.

    Parameter names are the landed map's
    (``weight_loaders_fp8.py:366-389``). No ``*_rope_*`` projection exists:
    ``mla_use_nope`` with ``qk_rope_head_dim == 0`` means there is no rotary
    head slice at all.

    THE CACHE GEOMETRY. MLA caches **one compressed latent vector per token
    per layer**, not one entry per attention head. The map's own projection
    name says so -- ``kv_a_proj_with_mqa``, multi-query, a single KV head --
    so ``num_kv_heads`` is 1 and is **not** tensor-parallel sharded: a single
    latent is replicated across ranks rather than split. The width is
    ``kv_lora_rank + qk_rope_head_dim``, which is 512 on this checkpoint.
    """

    #: The map's module path for this family (``weight_loaders_fp8.py:366``).
    CACHE_NAME_SUFFIX = "self_attn"

    #: MLA compresses KV to one latent per token; the latent is replicated
    #: across tensor-parallel ranks, so this is 1 at every world size.
    NUM_LATENT_KV_HEADS = 1

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_attention_heads = int(text_config.num_attention_heads)
        self.kv_lora_rank = int(text_config.kv_lora_rank)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.qk_nope_head_dim = int(text_config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(text_config.qk_rope_head_dim)
        self.v_head_dim = int(text_config.v_head_dim)
        self.mla_use_nope = bool(text_config.mla_use_nope)
        # ``inc-glm53f-039b`` adds these two scalars, because the projections
        # section below needs them and neither existed on the skeleton. The
        # hidden size is the input width of two of the five sites and the output
        # width of a third; the epsilon is the checkpoint's, read through the
        # config on the same ground stated at line 889 rather than defaulted
        # locally.
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
        # -- inc-glm53f-085 (WP5 repair) declares the four blockwise-FP8 scale
        #    parameters, in its OWN call so the seven names above stay exactly as
        #    ``inc-glm53f-039b`` and ``inc-glm53f-013`` landed them.
        #
        #    DERIVED FROM THE MAP'S OWN LIST, never a second literal: the four
        #    leaves come from ``DSA_SCALED_PROJECTIONS`` and the suffix from
        #    ``FP8_SCALE_SUFFIX``, both in ``weight_loaders_fp8.py``. That is what
        #    makes the declared-name set and the weight map's parameter set unable
        #    to drift apart -- the bijection ``test_kv_spec.py`` asserts.
        #
        #    ``kv_b_proj`` is absent BECAUSE it is absent from that list: this
        #    checkpoint keeps it in BF16 (``inc-glm53f-078``), so it carries no
        #    scale companion and takes no dequant.
        _declare_parameters(
            self,
            *(f"{leaf}_{FP8_SCALE_SUFFIX}" for leaf in DSA_SCALED_PROJECTIONS),
        )
        self.indexer = Glm5NextDSAIndexer(text_config)

    # -- inc-glm53f-100 -- THE ONE PLACE THIS CLASS TURNS HEADS INTO ITS HEADS.
    #
    # It sits here, in the class preamble ahead of both section banners, because
    # both sections below read it: the projections' widths and the absorb
    # operands' widths are the same head count seen from two sides. Putting it in
    # either section would make the other section's reader cross a boundary to
    # find out what a head count means.

    def _heads_per_rank(self) -> int:
        """This rank's share of the attention heads. Floored at 1.

        WHY THIS IS A METHOD AND NOT A CONSTRUCTOR VALUE. ``self.num_attention_
        heads`` stays the MODEL's count for the life of the module -- it is what
        the checkpoint, the config and every closed-form comment mean by "heads" --
        and this is the per-rank view of it. Overwriting the attribute instead
        would have made every remaining reader of it silently per-rank, including
        the ones in other increments' sections, and would have needed the world
        size in ``__init__``, which this class is not given: it is constructed at
        ``Glm5NextDSALayer.__init__`` with the text config alone, and that call
        site belongs to another increment.

        WHERE THE WORLD SIZE COMES FROM. :func:`_resolve_world_size`, the module's
        own reader, which is also what the root binds ``self.world_size`` to
        (``:4945``) and therefore what the shard table's width callbacks are
        handed. One source, so the width a rank's weight is sliced to and the
        width this class expects cannot come from two different answers. It is
        read per call rather than cached because tests inject the world size by
        monkeypatching that function (``test_load_weights.py:3180``) and a value
        cached in ``__init__`` would have frozen the answer before the injection.

        FLOORED AT 1 by ``_per_rank``, reused rather than re-derived so this class
        and the shard table floor identically: the table's three MLA widths all go
        through ``_mla_head_count``, which calls that same function on the same two
        inputs.

        ``_per_rank`` HAS EXACTLY THREE CALLERS IN THIS FILE, read rather than
        assumed -- this method, ``_mla_head_count``, and ``Glm5NextKDAAttention``'s
        ``__init__``, which divides ONCE into ``num_kv_heads_per_rank``. That third
        caller is why the KDA width functions divide nowhere: their attribute is
        already per-rank. MLA needs the division here because its attribute is not.
        """
        return _per_rank(self.num_attention_heads, _resolve_world_size())

    # -- the PROJECTIONS section -- D14 owner: ``inc-glm53f-039b`` (M3) -------
    #
    # WHAT THIS SECTION IS FOR. The five low-rank projections had no substrate
    # member left to call: ``inc-glm53f-072`` measured both candidates and both
    # REFUSE this checkpoint's widths. ``inc-glm53f-039a`` therefore wrote the
    # projection as a NKI kernel, and this section is the call site that reaches
    # it. Every number below is computed from the config; not one is read from a
    # weight's shape, so a mis-shaped checkpoint is caught rather than adopted.
    #
    # WHY THERE IS NO TORCH MATMUL ANYWHERE BELOW. The sibling linear-attention
    # class projects with ``x @ w.t()`` and is right to: its widths are small and
    # no kernel refuses them. Here a torch matmul would be a fallback for work a
    # kernel now does, so this section's acceptance counts occurrences of the
    # torch matmul forms in this class and requires ZERO. That is also why the
    # weights are transposed once, below, rather than per call.
    #
    # WHAT THIS SECTION DOES NOT DO, deliberately:
    #   * it does NOT implement ``forward`` -- D14 gives the forward stubs to
    #     ``inc-glm53f-013`` and then to ``inc-glm53f-054``, and the decode path
    #     that would call these methods to ``inc-glm53f-042``. Both methods below
    #     are entry points those increments call; neither runs on its own.
    #   * it allocates NO rotary parameter and computes no rotary slice, because
    #     ``qk_rope_head_dim`` is 0 on this checkpoint and that 0 is a value
    #     rather than a placeholder. The absence is counted by the acceptance.

    #: Attribute the transposed projection weights are cached on. A plain
    #: attribute and NOT a buffer on purpose: a buffer enters ``state_dict()``,
    #: which would double every projection weight in a saved checkpoint.
    PREPARED_WEIGHTS_ATTR = "_prepared_projection_weights"

    def projection_widths(self) -> tuple[tuple[str, int, int], ...]:
        """The five sites as ``(name, in_features, out_features)``, closed form.

        Each width is derived from the config and named where it comes from, so
        the expectation a test compares against is not a transcription of the
        same literal the code used.

        On this checkpoint the rotary head width is 0, so the query head width
        is the nope width alone and the latent width is the rank alone. Both
        sums are written out anyway: on a config that had a rotary slice the
        bare value would be short by exactly that slice, which is the same
        reason ``_resolve_mla_head_size`` sums rather than takes the rank.

        THE HEAD COUNT IS THIS RANK'S (``inc-glm53f-100``), so these are the five
        widths of the weights THIS rank holds, which is what they have to be: a
        checkpoint's projection is sliced to the rank's heads at load time, and
        this method is what ``prepare_projection_weights`` checks the loaded weight
        against. The three widths that carry the head count therefore narrow with
        the world size and the other two do not -- the latent projections are
        replicated, because MLA compresses KV to one latent per token rather than
        one per head. At world size 1 every width is byte-identical to what it was
        before that increment.
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
        """Transpose the five projection weights ONCE. Returns how many.

        THE ONE-TIME TRANSPOSE IS THIS INCREMENT'S OWED WORK, and it is owed
        because of a hardware fact on the other side of the seam. A matmul
        contracts the partition axis, so the kernel needs each weight
        contraction-major, ``[in_features, out_features]``. A checkpoint stores
        the torch orientation, ``[out_features, in_features]``. Transposing per
        call would copy up to 64 MB every time; a projection weight never
        changes, so it is transposed here, once, when the weights are loaded.
        The kernel's own record states the same division of labour
        (``../../../artifacts/campaigns/glm-5.3-flash-port/increments/evidence-039a.md``).

        Each weight is checked against the closed-form widths before it is
        transposed. A checkpoint whose projection is the wrong shape fails here,
        with the site named, instead of reaching the kernel as a geometry it
        would accept and quietly compute the wrong thing with.
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
            # -- inc-glm53f-085 owns THIS ONE STEP and nothing else here. The
            #    dequant runs in the checkpoint's own [out_features, in_features]
            #    orientation, where the scale grid matches by construction, and
            #    only THEN is the weight transposed by the line below -- so
            #    ``inc-glm53f-039b``'s one-time transpose keeps both its position
            #    and its ground.
            weight = self._dequantised_projection_weight(name, weight)
            # ``.t()`` alone is a view, and the kernel loads from memory, so the
            # copy is forced here -- once -- rather than left for the seam.
            prepared[name] = weight.to(torch.float32).t().contiguous()
        setattr(self, self.PREPARED_WEIGHTS_ATTR, prepared)
        return len(prepared)

    def _dequantised_projection_weight(
        self, name: str, weight: torch.Tensor
    ) -> torch.Tensor:
        """One projection weight in a real dtype, dequantised if it arrived as fp8.

        WHY THIS EXISTS. The published checkpoint stores four of the five MLA
        projections as blockwise-FP8 bytes with a ``weight_scale_inv`` companion
        per 128x128 tile. Before this method the bytes were transposed and used
        as if they were real numbers, so those four projections computed the
        wrong function at exactly the right shapes -- a defect no shape check
        can see. ``inc-glm53f-085`` repairs it.

        THE TEST IS THE DTYPE, NEVER A CONFIG FLAG. A quantisation flag can be
        wrong, absent, or stale, and a weight that is fp8 bytes is fp8 bytes
        whatever a flag says. So the three cases are decided by what actually
        arrived:

        * a real dtype is returned UNCHANGED, exactly as ``inc-glm53f-039b``
          landed it -- this method adds nothing to that path;
        * fp8 bytes with their scale materialised are dequantised;
        * fp8 bytes with NO scale materialised RAISE, naming the site and the
          parameter that is missing.

        THE THIRD CASE IS THE POINT. Silently treating unscaled fp8 bytes as
        numbers is the defect being repaired, so the one thing this method must
        never do is continue quietly when the scale is absent. A refusal that
        names the site is also distinguishable from a skipped test, which is
        what makes it checkable.

        The dequant itself is the fork's own ``dequantise_blockwise``; this
        method chooses no arithmetic of its own. It returns fp32, so the
        caller's following ``.to(torch.float32)`` is already satisfied and
        stays a no-op rather than a second conversion.
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

    # -- the ABSORB section -- D14 owner: ``inc-glm53f-042`` -------------------
    #
    # WHAT THIS SECTION IS FOR. The projections above work in the HEAD width 256;
    # the sparse attention seam works in the LATENT rank 512. Two per-head
    # matmuls cross between them, and ``inc-glm53f-097`` authored the NKI seam
    # that computes them. This section prepares that seam's two weight operands
    # once, at load time, and does nothing else.
    #
    # WHERE THE OPERANDS COME FROM, and why no new weight is read. Both are
    # halves of ONE weight this class already prepares, ``kv_b_proj``: it expands
    # a latent into a per-head key and value pair, and ``inc-glm53f-039b``
    # already splits its OUTPUT on that boundary (``key_value[..., :256]`` and
    # ``[..., 256:]``). This section splits the WEIGHT on the same boundary
    # instead, which is what lets the two matmuls be absorbed into the query and
    # into the attention output. No second projection is read, no new parameter
    # is declared, and no checkpoint tensor is touched.
    #
    # WHY ONCE AND NOT PER CALL. ``inc-glm53f-039b``'s one-time transpose exists
    # because copying a projection weight per call costs up to 64 MB every call.
    # The same argument applies with more force here: the split reshapes, slices
    # and permutes a 512 x 32,768 weight, and it never changes. So it runs at
    # load time behind the same device pre-flight, and the accessor REFUSES
    # instead of building on demand -- refusing is what makes "never per call"
    # checkable, exactly as ``_prepared_weight`` argues just above.
    #
    # THE TWO ORIENTATIONS ARE NOT SYMMETRIC, and that asymmetry is the design.
    # ``nc_matmul`` contracts the partition axis, so each operand must present
    # its own contraction extent there:
    #   absorb-in  turns ``query [S, H, 256]`` into ``[S, H, 512]``, contracting
    #              the HEAD width, so ``W_UK`` is ``[H, 256, 512]``;
    #   absorb-out turns the seam's ``[S, H, 512]`` into ``[S, H, 256]``,
    #              contracting the LATENT, so ``W_UV`` is ``[H, 512, 256]``.
    # ``W_UV`` is therefore the value half as it already sits in the prepared
    # weight, and ``W_UK`` is the key half TRANSPOSED. Storing ``W_UK`` the other
    # way round would put a transpose back on the per-forward path, which is the
    # one cost this section exists to avoid.

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

        THE HEAD COUNT IS THIS RANK'S (``inc-glm53f-100``, a second writer in this
        section BY CONCERN: it moves widths and touches no absorb algebra and no
        numeric). It has to be, and not as a convenience: both operands are split
        from the prepared ``kv_b_proj``, which under tensor parallelism holds only
        this rank's heads, so an expectation written in the model's full head count
        would refuse a correctly loaded weight. The two contraction extents --
        ``qk_nope_head_dim`` and ``kv_lora_rank`` -- are NOT touched, because
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
        """Split the prepared ``kv_b_proj`` into the two absorb operands. ONCE.

        Returns how many operands were built, so a caller can tell this ran from
        a call that had nothing to do.

        THE INPUT IS THE PREPARED WEIGHT, NOT THE CHECKPOINT PARAMETER, and the
        difference is load-bearing. ``_prepared_weight`` returns ``kv_b_proj``
        after ``inc-glm53f-085``'s dequantisation chain and
        ``inc-glm53f-039b``'s one-time transpose, so it is
        ``[kv_lora_rank, heads * (nope + v)]`` in a real dtype. Splitting the raw
        parameter instead would, on a checkpoint that stored this weight as fp8
        bytes, split bytes rather than numbers and compute the wrong function at
        exactly the right shapes -- the defect class ``-085`` was raised to fix.

        THE SPLIT IS A REARRANGEMENT AND NOTHING ELSE. Every element of both
        operands is an element of the prepared weight, in the prepared weight's
        own dtype: no cast, no scale, no arithmetic. The acceptance asserts that
        head by head as item (iv-a) per ``../../../artifacts/campaigns/
        glm-5.3-flash-port/approvals/DECISIONS.md`` §15a.5, which is what makes
        "the split picks the right bytes for the right head" a measurement
        instead of a claim.

        ``inc-glm53f-100`` MOVES ONE BINDING AND NOTHING ELSE IN THIS METHOD: the
        head count below is now this rank's. Every line after it is untouched --
        the closed form, the reshape, the two permutes, the width comparison -- and
        that is the point: the split is per head, so once "heads" means this rank's
        heads the whole chain is already per-rank arithmetic. The refusal it raises
        is the one that matters under tensor parallelism, because a full-count
        expectation against a rank's weight would stop a correct load.
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
        operands = {
            # [latent, heads, nope] -> [heads, nope, latent]: the key half,
            # transposed, because absorb-in contracts the HEAD width.
            "W_UK": per_head[:, :, :nope].permute(1, 2, 0).contiguous(),
            # [latent, heads, v] -> [heads, latent, v]: the value half as it
            # already stands, because absorb-out contracts the LATENT.
            "W_UV": per_head[:, :, nope:].permute(1, 0, 2).contiguous(),
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

        Refusing instead of building on demand is what makes "once, at load
        time" checkable -- the argument ``_prepared_weight`` makes, and the
        failure it prevents is the same one: a silent per-call rebuild that
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

        A method rather than a module-level helper, because this file's
        module-level region is another increment's D14 section -- the same
        reason the sibling layer's own norm gives.

        The two latent norms sit BETWEEN the projections, so they belong to this
        section and to no other. They are applied here rather than left for a
        later increment: a projection chain that emitted un-normalised latents
        would be numerically wrong, and correcting it later would put a second
        writer inside this section.
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        return normed * gain.to(torch.float32)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, no-rotary key and value from hidden states. FOUR dispatches.

        ``hidden_states`` is ``[tokens, hidden_size]``. The three returns are
        ``[tokens, heads, qk_nope_head_dim]``, ``[tokens, heads,
        qk_nope_head_dim]`` and ``[tokens, heads, v_head_dim]``.

        The chain is the low-rank one this checkpoint declares: compress to a
        rank, normalise the latent, expand to the head width. The key and value
        come out of ONE expansion and are split, which is why this is four
        dispatches and not five.

        THE HEAD COUNT IS THIS RANK'S (``inc-glm53f-100``). This method has no
        production caller -- it is the DENSE path, kept as the oracle the absorbed
        decode path is compared against, which is why ``inc-glm53f-042`` wrote a
        sibling rather than reuse it -- so it needs no reduction and gets no
        acceptance item of its own. The binding still moves, for two reasons that
        are not tidiness: it reads ``projection_widths`` two lines above, so a
        full-count head here against per-rank widths would make ONE method
        internally inconsistent; and the key-value reshape below would then refuse
        a correctly loaded weight under tensor parallelism, turning the class's
        comparison oracle into the one thing that cannot run.
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

    # -- RIDER on the projections section above -- ``inc-glm53f-042``, design
    #    entry ``design-20260905-p`` ruling (iii). DISCLOSED, not silent.
    #
    #    WHAT THE RIDER IS. The absorbed decode path needs the NORMALISED LATENT
    #    that ``project_qkv`` computes as a local and then consumes. The design
    #    offered two ways to expose it: a ``return_latent`` keyword on
    #    ``project_qkv``, or a sibling method reusing its first three dispatches.
    #
    #    I CHOSE THE SIBLING, and the reason is not style. ``project_qkv``'s
    #    FOURTH dispatch is the ``kv_b_proj`` expansion, 512 -> 32,768, and the
    #    entire point of absorbing ``W_UK`` and ``W_UV`` is that the decode path
    #    never performs that expansion. A ``return_latent`` keyword would have
    #    handed back the latent while still paying for the expansion on every
    #    decode step, making the absorbed path SLOWER than the path it replaces.
    #    That is not a micro-optimisation; it would defeat the increment.
    #
    #    WHAT IT COSTS. The three dispatches below repeat ``project_qkv``'s first
    #    three lines rather than being factored out of them. Factoring would edit
    #    a landed method's body, and ``-039b``'s acceptance asserts that body's
    #    behaviour; a behaviour-identical refactor is still a rewrite of code
    #    another increment's items stand on. Repetition that is visible beats a
    #    refactor that is invisible, so ``project_qkv`` above is untouched --
    #    byte-for-byte, including its three returns.

    def project_query_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The NORMALISED query latent. ONE dispatch. ``[tokens, q_lora_rank]``, float32.

        ADDITIVE, AND THAT IS THE WHOLE OF IT. ``inc-glm53f-051`` needs this exact
        value and no caller could obtain it: the sibling below computed it as its
        first two statements and returned only the query and the KV latent. This
        method is those two statements, named, and the sibling now calls it -- so
        the value has one producer instead of two copies, and the sibling's
        signature, returns and dispatch count are all unchanged. Authorised at
        design entry ``design-20260905-ad``; the D14 owner of this section is
        ``inc-glm53f-039b`` and this is the only line of it that moves.

        WHO NEEDS IT AND WHY IT IS NOT AN INTERNAL. The DSA indexer's ``wq_b``
        projection contracts ``q_lora_rank``, so the indexer's input IS this
        latent -- exactly as the reference passes it in as an argument
        (``vllm/models/glm5next/nvidia/attention.py:317``, consumed at ``:319``,
        blob at pin ``878631b6``). The DSA layer calls this once per phase and
        hands the result to the indexer.

        FLOAT32 ON PURPOSE, and the caller is told rather than left to infer. The
        seam returns float32 by contract and the norm runs in float32; the sibling
        casts back to the model dtype only at its own return, because that is what
        its callers consume. This method hands back what the seam and the norm
        produced, so a caller that needs another dtype casts once, itself, at the
        point it knows about.

        RECORDED DEBT, not a defect and not this increment's to pay: ``attend()``
        recomputes this latent inside ``project_query_and_latent``, so a DSA layer
        that calls the indexer and then ``attend()`` computes it twice. Threading
        the latent into ``attend()`` changes a landed signature with its own
        callers, which is ``inc-glm53f-054``'s to do when it writes the 45-layer
        forward. The cost is one dispatch of a ``[tokens, 4096] x [4096, 1536]``
        projection per layer per phase, and it is declared in this increment's
        predictions rather than absorbed.
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
        """Query and the NORMALISED KV latent. THREE dispatches, not four.

        ``hidden_states`` is ``[tokens, hidden_size]``. The returns are
        ``query [tokens, heads, qk_nope_head_dim]`` and
        ``kv_latent [tokens, kv_lora_rank]``.

        This is the absorbed path's entry into the projections: the query at the
        head width, and the compressed KV latent BEFORE any expansion. The
        expansion is what ``W_UK`` and ``W_UV`` absorb, so performing it here
        would be paying for the work this increment exists to remove.

        The latent is returned NORMALISED because that is what the cache stores
        and what the seam contracts against; returning the un-normalised
        compression would put a norm on every reader of the cache instead of one
        norm on the writer.

        THE FIRST TWO STATEMENTS NOW LIVE IN :meth:`project_query_latent`, and
        nothing else about this method moved (``inc-glm53f-051``, entry
        ``design-20260905-ad``). Same signature, same two returns, same THREE
        dispatches -- the extracted method performs the first of the three. The
        extraction exists because the DSA indexer needs that intermediate value
        and no caller could reach it while it was a local here.

        THE HEAD COUNT IS THIS RANK'S (``inc-glm53f-100``), so the returned query
        is ``[tokens, heads_on_this_rank, qk_nope_head_dim]``. This was the one
        site in the class where a full-count head reshape would NOT have raised:
        the reshape below divides the projection's own output width by the head
        count, so on two ranks of a 64-head model a rank's ``[tokens, 8192]``
        query would have reshaped cleanly to ``[tokens, 64, 128]`` instead of
        ``[tokens, 32, 256]`` -- right element count, wrong heads, no error, and a
        head width of 128 the config never mentions. Reading the count from the
        same reader the widths use is what removes that.
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

    def project_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        """The output projection. ONE dispatch.

        ``attn_out`` is ``[tokens, heads, v_head_dim]`` or the same flattened to
        ``[tokens, heads * v_head_dim]``; the return is ``[tokens,
        hidden_size]``. Both input forms are accepted because the decode path
        that calls this is another increment's and its layout is its own choice,
        not something this section should dictate.

        THIS IS THE ROW-PARALLEL SITE, AND IT REDUCES (``inc-glm53f-100``). Under
        tensor parallelism the head width is this projection's INPUT, so each rank
        holds a slice of the weight's columns, contracts it against its own heads,
        and produces a PARTIAL SUM at the full output width -- every rank's result
        is the same shape and none of them is the answer. The answer is their sum,
        so the collective is not an optimisation here: without it the model returns
        one rank's fraction of every attention output, at exactly the right shape,
        which is the failure mode that does not announce itself.

        THE GROUP AND THE FORM ARE THE FORK'S, not a second convention. All eight
        shipped model files reduce a row-parallel result by calling ``all_reduce``
        on ``get_tp_group()``'s coordinator, in the statement form used below
        (``qwen3/model.py:535-537``, and 18 sites across ``llama3``, ``gpt_oss``,
        ``qwen3`` and ``qwen3_vl``); the convention is written down at
        ``parallel/DESIGN.md:149-151``. :func:`_resolve_tp_group` is where this
        file resolves it, and it returns ``None`` at world size 1, so a
        single-rank run takes exactly the path it took before this increment --
        no collective, no import of vllm, no behaviour change.

        TWO READINGS WORTH STATING, because both could be wrong in a way tests
        would not catch. First, the reduction is IN PLACE: the statement form
        discards a return value, so it assumes ``all_reduce`` writes through its
        argument. That assumption is inherited rather than invented -- all 18
        shipped row-parallel sites make it -- and it is safe against aliasing here
        because ``mla_projection`` returns a fresh tensor rather than a view of a
        cached weight. Second, the sum happens BEFORE the cast back to the input
        dtype: partial sums are reduced in the float32 the seam computed them in,
        because rounding each rank's fraction to bfloat16 first and adding after
        would round the parts instead of the whole.
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
        # Sum this rank's partial with every other rank's. ``None`` means one
        # rank, where the partial already is the whole sum.
        group = _resolve_tp_group()
        if group is not None:
            group.all_reduce(projected)
        return projected.to(attn_out.dtype)

    # -- the DECODE section -- D14 owner: ``inc-glm53f-042`` -------------------
    #
    # WHAT THIS SECTION IS FOR. It is the chain that turns hidden states into
    # attention output for this layer, and it is ONE method rather than a prefill
    # method and a decode method. The acceptance compares a decode step against
    # the matching slice of a prefill run, and that comparison is only worth
    # taking if BOTH sides are the same code at different token counts. Two
    # methods would let the comparison pass while the two paths diverged, which
    # is the failure the item exists to catch.
    #
    # THE CHAIN, and every step's owner:
    #   project_query_and_latent  three dispatches, this increment's rider
    #   the cache write           this increment
    #   the cache read            this increment
    #   absorb-in                 ``inc-glm53f-097``'s seam, W_UK
    #   sparse attention          ``inc-glm53f-040``/``-041``/``-093``'s seam
    #   absorb-out                ``inc-glm53f-097``'s seam, W_UV
    #   project_output            ``inc-glm53f-039b``'s method, called unchanged
    #
    # WHY THE EXPANSION NEVER HAPPENS HERE. ``kv_b_proj`` expands a 512 latent to
    # 32,768 per token. The absorbed chain multiplies the QUERY by ``W_UK``
    # instead and the attention OUTPUT by ``W_UV``, so the expansion is never
    # computed on the per-token path at all. That is the whole reason this section
    # is not simply ``project_qkv`` followed by dense attention.
    #
    # WHAT THIS SECTION DOES NOT OWN, and both absences are the design's:
    #   * TOP-K SELECTION. The sparse seam requires ``topk_indices`` and no block
    #     on the plan produces them -- the DSA indexer is a ``-013`` stub that
    #     raises. So the indices are a caller's argument here. Recorded as a plan
    #     gap by the lead rather than improvised into this method.
    #   * THE SOFTMAX SCALE. No block registers a value for it, so it is a
    #     caller's argument too. Deriving one here would mint a registered value
    #     this increment has no authority to mint.

    def attend(
        self,
        hidden_states: torch.Tensor,
        latent_cache: torch.Tensor,
        start_position: int,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """One layer's MLA attention. Returns ``[tokens, hidden_size]``.

        ``hidden_states`` is ``[tokens, hidden_size]``: a prefill passes all its
        tokens at once and a decode step passes one. ``latent_cache`` is this
        layer's latent cache in the model's own declared spec layout --
        ``[slots, NUM_LATENT_KV_HEADS, head_size]``, one latent per token, bf16 --
        and ``start_position`` is the slot the first of these tokens occupies.
        The cache is WRITTEN in place for those tokens and then READ from slot 0
        through the last written slot, which is the context this attention sees.

        WHY ``batch_size`` IS A PARAMETER AND NOT INFERRED. ``latent_cache`` here
        is ONE sequence's slots, because MLA caches one latent per token per
        layer and this checkpoint's serving constraint G1 is ``B == 1``. A caller
        serving more than one sequence would have to loop, and the named refusal
        below says so rather than letting a second sequence's tokens land in the
        first one's slots. The parameter exists to make that constraint a
        measurement: the acceptance asserts the refusal is raised BY NAME and
        that no seam counter moves, which distinguishes a refusal from a silent
        skip.
        """
        if int(batch_size) != 1:
            raise Glm5NextMLADecodeError(
                f"the MLA decode path serves one sequence at a time (constraint "
                f"G1, batch_size == 1); got batch_size={batch_size}. The latent "
                f"cache passed here is a single sequence's slots, so a larger "
                f"batch would write one sequence's latents into another's slots"
            )
        if hidden_states.ndim != 2 or int(hidden_states.shape[1]) != self.hidden_size:
            raise Glm5NextMLADecodeError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(hidden_states.shape[0])
        latent = self.kv_lora_rank
        want_cache = (self.NUM_LATENT_KV_HEADS, self.head_size)
        if latent_cache.ndim != 3 or tuple(latent_cache.shape[1:]) != want_cache:
            raise Glm5NextMLADecodeError(
                f"latent_cache must be [slots, {self.NUM_LATENT_KV_HEADS}, "
                f"{self.head_size}] -- this layer's own declared cache spec; got "
                f"{tuple(latent_cache.shape)}"
            )
        slots = int(latent_cache.shape[0])
        start = int(start_position)
        if start < 0 or start + tokens > slots:
            raise Glm5NextMLADecodeError(
                f"these {tokens} token(s) at start_position={start} do not fit "
                f"the cache's {slots} slot(s); a write past the end would "
                f"silently wrap onto another sequence's rows"
            )

        query, kv_latent = self.project_query_and_latent(hidden_states)

        # THE CACHE WRITE. One latent per token, into this layer's single KV
        # head. Done before the read below, so a decode step attends to its own
        # token as well as its context -- the same set a prefill of the same
        # tokens would see, which is what makes the two comparable.
        latent_cache[start : start + tokens, 0, :] = kv_latent.to(latent_cache.dtype)

        # THE CACHE READ. Slot 0 through the last slot written is this sequence's
        # context. ``[S_kv, latent]`` is the shape the sparse seam contracts.
        c_kv = latent_cache[: start + tokens, 0, :]

        from vllm_neuron.functional.attention.mla_absorb import mla_absorb
        from vllm_neuron.functional.attention.mla_sparse import mla_sparse_attention

        # ABSORB-IN. ``[S, H, 256] x [H, 256, 512] -> [S, H, 512]``: the query
        # moves into the latent space the seam works in.
        q_lift = mla_absorb(query, self._absorb_weight("W_UK"))
        if int(q_lift.shape[2]) != latent:
            raise Glm5NextMLADecodeError(
                f"absorb-in produced a width of {int(q_lift.shape[2])}; the "
                f"sparse seam contracts {latent}"
            )

        attended = mla_sparse_attention(q_lift, c_kv, topk_indices, softmax_scale)

        # ABSORB-OUT. ``[S, H, 512] x [H, 512, 256] -> [S, H, 256]``: back to the
        # head width ``project_output`` consumes. The cast is here rather than
        # inside the seam because the seam returns float32 by contract and this
        # layer's chain carries the model dtype.
        reduced = mla_absorb(
            attended.to(hidden_states.dtype), self._absorb_weight("W_UV")
        )
        return self.project_output(reduced)

    def forward(
        self,
        normed_hidden_states: torch.Tensor,
        *,
        latent_cache: torch.Tensor,
        pool_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        start_position: int,
        softmax_scale: float,
        max_seq_len: int,
        page_size: int,
        slot_mapping: torch.Tensor | None = None,
        tail: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        """This module's whole contribution to one layer: select, then attend.

        Returns ``[tokens, hidden_size]`` -- the attention half's output, with no
        residual added and nothing normalised, because both belong to the layer.

        THE INPUT ARRIVES NORMALISED, and the parameter name says so. The layer
        owns its pre-attention norm and its residual; this module owns the three
        calls between them. Every one of the three consumes the same normalised
        tensor, which is why one argument carries it rather than three.

        THE THREE CALLS ARE THE COMPOSITION AND THIS METHOD ADDS NOTHING TO THEM:
        the query latent the indexer's ``wq_b`` contracts, the indexer that turns
        it into selected rows, and :meth:`attend`, which ends in
        :meth:`project_output`. No numeric is authored here and no refusal is
        added: every extent and every dial is checked by the callee that owns it,
        and a second check here is how two authorities on one extent come to
        disagree.

        THE INDICES PASS THROUGH UNCHANGED. Entry ``design-20260905-af`` route
        (a) puts the ``-1`` mask inside ``inc-glm53f-098``'s kernel, so no filler,
        compaction, clamp or mask is applied between the indexer and ``attend()``
        -- this hands over exactly what the indexer returned. The sibling layer's
        forward states the same ruling, and this method is now the one place that
        realises it.

        RECORDED DEBT, unchanged and not paid here: ``attend()`` recomputes the
        query latent inside :meth:`project_query_and_latent`, so the latent this
        method computes for the indexer is computed a second time -- one
        ``[tokens, hidden_size] x [hidden_size, q_lora_rank]`` dispatch per layer
        per phase. :meth:`project_query_latent` declares that debt and names
        ``inc-glm53f-054`` as its payer; paying it means threading the latent into
        ``attend()``, which changes a landed signature whose own acceptance items
        call it positionally. That is a widening this block does not take, so the
        debt is carried forward and declared rather than absorbed in silence.

        The two carriers are the caller's and both are written in place:
        ``latent_cache`` is ``attend()``'s own contract and ``pool_cache`` and
        ``tail`` are the indexer's. See :meth:`Glm5NextDSAIndexer.forward` for
        what each means and why ``max_seq_len`` is a python int.
        """
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
        )
        return self.attend(
            normed_hidden_states,
            latent_cache,
            int(start_position),
            topk_indices,
            float(softmax_scale),
        )


class Glm5NextDSALayer(nn.Module):
    """Decoder layer on the ``deepseek_sparse_attention`` half.

    Section reserved for ``inc-glm53f-051`` (DSA runner integration and
    sequence tiling).
    """

    #: Attribute the family's attention module is bound to -- the map's own
    #: module path, which ``weight_loaders_fp8.py``'s ``_add_dsa_attention``
    #: builds as ``f"{param_prefix}.self_attn"``. RE-GROUNDED BY
    #: ``inc-glm53f-082``: the old cite pointed at a line that is now the
    #: post-attention-layernorm mapping. The value itself does not move.
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
            # ``Glm5NextKDALayer``: ``_add_mhc`` runs for EVERY layer of the
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

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-attention RMSNorm: ``x / sqrt(mean(x**2) + eps) * gain``.

        DELIBERATELY THE SAME BODY AS ``Glm5NextKDALayer._input_norm``, and the
        duplication is forced rather than lazy. Sharing it would mean either a
        module-level helper -- and that region is another increment's D14 section,
        the reason the sibling gives for it being a method at all -- or a new base
        class, which would move a landed class this increment does not own. D14
        tells an implementer to raise a widening rather than take it, so the body
        is repeated here in this increment's own section and the sibling stays
        untouched.
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
        pool_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        start_position: int,
        softmax_scale: float,
        max_seq_len: int,
        page_size: int,
        slot_mapping: torch.Tensor | None = None,
        tail: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        """Pre-norm, then the sparse attention half, then the residual add.

        THE ATTENTION HALF IS ONE CALL NOW (``inc-glm53f-054a``, a declared
        second writer in this section by the invitation two paragraphs down: the
        landed text names ``inc-glm53f-054`` as the increment that joins this
        layer's halves). The three calls this method used to inline -- the query
        latent, the indexer, ``attend()`` -- are
        :meth:`Glm5NextMLAAttention.forward`'s body, unchanged in order,
        arguments and count, and this layer calls it. Behaviour does not move:
        the same three callees run in the same order on the same operands, so
        every dispatch reading this layer's acceptance declares is the reading it
        was. What moves is that the composition has ONE definition -- a second
        copy in the parent is how the two come to disagree about which tensor
        each callee consumes, and the 45-layer forward this block also writes
        would have been the second copy's second reader.

        THE INDICES PASS THROUGH UNCHANGED, which is the ruling and the whole
        point of the shape. Entry ``design-20260905-af`` route (a) puts
        the ``-1`` mask inside ``inc-glm53f-098``'s kernel, so no filler, no
        compaction, no clamp and no mask is applied between the indexer and
        ``attend()`` -- exactly what the indexer returned is handed over. The
        callee states the same ruling, and it is now the one place that realises
        it.

        WHAT THIS FORWARD DELIBERATELY DOES NOT DO, AND WHY IT IS NOT A GAP. The
        same two absences the KDA sibling records, for the same D14 reason:

        * ``self.mlp`` raises whichever branch ``_build_mlp`` chose, and those
          sections belong to ``inc-glm53f-031`` through ``inc-glm53f-033``.
        * the six mHC weights sit flat on this layer but no
          ``Glm5NextHyperConnection`` instance is bound to it, and that wiring is
          ``inc-glm53f-030``'s section.

        So this forward stops at the attention half and ``inc-glm53f-054`` joins
        the halves when it writes the 45-layer forward.

        THE TWO CARRIERS ARE THE CALLER'S, both written in place: ``latent_cache``
        is ``attend()``'s own contract and ``pool_cache`` and ``tail`` are the
        indexer's. See :meth:`Glm5NextDSAIndexer.forward` for what each means and
        why ``max_seq_len`` is a python int.
        """
        residual = hidden_states
        normed = self._input_norm(hidden_states)
        attn_out = self.attention(
            normed,
            latent_cache=latent_cache,
            pool_cache=pool_cache,
            seq_lens=seq_lens,
            start_position=int(start_position),
            softmax_scale=float(softmax_scale),
            max_seq_len=int(max_seq_len),
            page_size=int(page_size),
            slot_mapping=slot_mapping,
            tail=tail,
            position=position,
        )
        return residual + attn_out


def _build_layer(
    text_config: Glm5NextTextConfig, layer_idx: int, layer_type: str, world_size: int
) -> nn.Module:
    """One decoder layer, family chosen by EQUALITY on ``layer_types``.

    Never by substring: ``"attention"`` is a substring of both family names
    (``config.py:36-40``), so a substring test would silently mis-partition
    the stack -- and mis-partitioning it is exactly what the 34/11 split
    would then fail to detect.
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
# The tree. D14 owner for this section: ``inc-glm53f-013`` (creator), then
# ``inc-glm53f-054`` at M4, which replaces the stubs with the full 45-layer
# forward.
# ---------------------------------------------------------------------------


class Glm5NextModel(nn.Module):
    """The decoder stack, named ``model`` because the map's paths say so.

    Every mapped parameter outside the layer stack hangs here or on the root
    (``weight_loaders_fp8.py:315-319``).
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

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextModel.forward is a stub created by inc-glm53f-013; the "
            "full 45-layer forward lands with inc-glm53f-054"
        )


# ---------------------------------------------------------------------------
# ``inc-glm53f-091`` -- weight loading support.
#
# D14: this constant and ``load_weights`` plus its two private helpers on the
# class below are ``inc-glm53f-091``'s. ``inc-glm53f-054`` owns the 45-layer
# forward and every stub it replaces; nothing here touches one.
# ---------------------------------------------------------------------------

#: The dtype a quantised weight's checkpoint bytes are stored in. OCP
#: ``float8_e4m3fn``.
#:
#: **Declared here rather than imported, which is the fork's own convention for
#: this constant:** three shipped modules each declare their own
#: (``llama3/model_mx_fp8.py:101``, ``llama3/model_static_fp8.py:119``,
#: ``llama3/weight_pack_mx_fp8.py:21``). The sibling ``weight_loaders_fp8.py``
#: holds a private ``_FP8_DTYPE`` of this same value, and importing a private
#: name across modules is not that convention -- nor is it available under this
#: increment's partition of that file, which is the loader wiring only.
_FP8_DTYPE = torch.float8_e4m3fn

#: The leaf suffix every weight parameter in this tree carries, so a scale grid
#: can be named from its weight. ``inc-glm53f-011``'s naming convention, read
#: rather than restated: the map's own parameter names end in it
#: (``weight_loaders_fp8.py:295-298``).
_WEIGHT_LEAF_SUFFIX = "_weight"


def _scale_prep_leaves(module: nn.Module) -> list[str]:
    """The declared weight leaves whose scale grid is actually ON the module.

    ``inc-glm53f-095b``. Factored out of
    :meth:`Glm5NextForConditionalGeneration._run_load_time_preps`, where the same
    derivation read the declaration tuple and nothing else.

    IT READS PRESENCE BECAUSE PRESENCE IS WHAT THE LOOP DOES NEXT. For every leaf
    this returns, the caller takes a bare ``getattr`` of the sibling grid with no
    default, so a leaf kept here on the strength of a DECLARATION would raise
    ``AttributeError`` the moment that grid had not arrived. The routed expert
    bank is where the two answers part: it declares ``router_weight``, and no
    ``router_weight_scale_inv`` exists anywhere in this tree. The unfactored
    derivation yielded four leaves and eight operands for a bank; this one yields
    the three leaves that have grids.

    AND A DECLARATION TEST WOULD BE WRONG IN THE OTHER DIRECTION TOO. Scale grids
    in this tree are deliberately plain attributes rather than registered
    parameters, for the reason recorded on
    :meth:`Glm5NextForConditionalGeneration._load_out_of_band_scales`, so NO
    module in this file declares one. A membership test against
    ``declared_param_names`` therefore returns nothing at all for the shared
    expert, whose six-operand prep then fails on six missing positional
    arguments.

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


class Glm5NextWeightLoadError(ValueError):
    """A checkpoint could not be read onto this model tree.

    ``inc-glm53f-091``. A ``ValueError`` subclass to match the three named
    errors this file already raises (``Glm5NextHyperConnectionError``,
    ``Glm5NextBlockQuantRouteError``, ``Glm5NextSharedExpertRouteError``) rather
    than to make a claim about the failure being a file-system one -- the
    checkpoint argument may name a local directory or a remote repository, and
    :meth:`Glm5NextForConditionalGeneration.load_weights` must refuse both the
    same way.

    Every message this class carries names the checkpoint path, because the
    argument is passed straight through from the runner and a refusal that did
    not name it would send a reader to the wrong place.
    """


class Glm5NextForConditionalGeneration(nn.Module):
    """The blockwise-FP8 GLM-5.3-Flash implementation.

    The module path, this class name and the ``from_configs`` signature are
    **pinned by landed code**: ``factory.py:340`` already reads ``from
    .model_fp8 import Glm5NextForConditionalGeneration as Model`` and
    ``factory.py:342`` calls
    ``Model.from_configs(hf_config, text_neuron_config=..., vision_neuron_config=...)``.
    The name is duplicated with ``factory.py:268`` on purpose -- that is the
    plugin's selector-to-implementation convention, the same pair
    ``llama3/factory.py:42`` uses -- so neither side is renamed here.
    """

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
        # Mirrors the map's own condition: a tied head has no separate
        # checkpoint key and therefore no separate parameter
        # (``weight_loaders_fp8.py:382-383``).
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
        """Build from an HF config, the signature ``factory.py:342`` calls."""
        config = Glm5NextConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    # ── declared parameter names ─────────────────────────────────────────

    def declared_parameter_names(self) -> tuple[str, ...]:
        """Every parameter attribute path this tree declares, in tree order.

        This is the enumerable form of the derivation the lead ruling
        requires: the set returned here is compared against
        ``build_weight_mappings(text_config)``'s keys in
        ``test_kv_spec.py``. ``named_parameters()`` cannot be used, because
        declared-but-unmaterialised parameters are ``None`` and torch skips
        ``None`` entries in both ``named_parameters()`` and ``state_dict()``.
        """
        names: list[str] = []
        for module_path, module in self.named_modules():
            for leaf in getattr(module, "declared_param_names", ()):
                names.append(f"{module_path}.{leaf}" if module_path else leaf)
        return tuple(names)

    # ── weight loading (``inc-glm53f-091``) ──────────────────────────────

    def _checkpoint_block_size(self) -> tuple[int, int]:
        """The quantisation block shape THIS CHECKPOINT declares, as two ints.

        ``inc-glm53f-101``, remedy part 1. Read off
        ``quantization_config.weight_block_size``, which ``config.py:458`` already
        lifts verbatim, so the rule that converts a weight shard into grid rows is
        told the checkpoint's own tile instead of taking the parser's fallback
        (``quantization.py:114``). The fallback is right for this checkpoint and
        would be silently wrong for the next one, which is the whole reason this
        is threaded rather than defaulted.

        REFUSED RATHER THAN DEFAULTED when the field is absent or malformed. The
        only caller reaches this after finding a scale grid in the checkpoint, so
        a checkpoint that ships scale grids and declares no block shape is a
        contradiction, and guessing 128 there would scale real weight blocks by
        the wrong grid row.
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

        Load-bearing rather than cosmetic. The checkpoint reader takes its
        target dtype OFF THE PLACEHOLDER (``utils/checkpoints.py:437``) and
        CASTS the tensor it built to that dtype after a warning
        (``:570-576``), so a placeholder typed one step too wide does not merely
        log -- it changes the values that reach the model.

        The four cases come from
        :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.classify_mapped_keys`,
        which is the one classifier this and the loader choice share, so the
        dtype and the loader cannot disagree about what a key is.

        A STACKED EXPERT BANK IS FP8 FOR THE SAME REASON A LONE QUANTISED WEIGHT
        IS, and is NAMED in the arm below rather than left to fall through.
        ``inc-glm53f-095`` made a bank its own classifier kind; a kind this
        method did not name would fall past the ``quantised_weight`` arm, miss the
        sibling clause (a bank has no sibling scale ENTRY -- its scales are
        interleaved inside its own) and reach ``text_config.torch_dtype``. All 126
        bank placeholders on the real configuration would then be bf16 while
        their loader delivers fp8, and the reader warns on exactly that mismatch
        (``utils/checkpoints.py:437``, ``:570-575``). Settled from source before
        the fourth kind was added, not after -- ``inc-glm53f-095``'s
        refusal-collision reading, recorded with that increment.
        This is the second consumer in "one classifier, two consumers", and it
        moves whenever the classifier gains a kind.

        THE SIBLING CLAUSE, AND THE DEFECT IT REPAIRS. A quantised weight whose
        scale travels in the SAME map entry classifies ``quantised_weight`` and
        takes fp8 from the case above. But ``inc-glm53f-085`` gave each of the
        four scaled MLA projections its own scale-grid entry, which left each of
        those weights ALONE in its entry -- and a lone weight key classifies
        ``plain``. Without the clause below they took the config dtype, the
        reader narrowed the checkpoint's fp8 bytes to bf16, and
        :meth:`Glm5NextMLAAttention._dequantised_projection_weight` then saw a
        real dtype and returned the weight UNCHANGED. The dequant did nothing,
        silently, on every scaled projection of every sparse-attention layer,
        and the transpose that follows it operated on narrowed bytes -- exactly
        the defect ``inc-glm53f-085`` exists to repair, arriving through the
        placeholder dtype instead of through the transpose. Measured before this
        clause was written, with the dense-MLP two-key entry as the control that
        still reached the dequant:
        ``../../../artifacts/campaigns/glm-5.3-flash-port/increments/probe-091b-dsa-dtype.out``
        (``DEQUANT_BRANCHES_REACHED=0`` of 4, four cast lines logged).

        The clause asks the MAP whether a sibling scale grid exists for this
        weight, which is the same question
        ``_dequantised_projection_weight`` asks of the module at ``:2959`` --
        one question, asked of the two places that have to agree. It adds no
        second classifier of the three cases: ``classify_mapped_keys`` still
        decides, and this only distinguishes the two kinds of ``plain``.
        """
        kind = classify_mapped_keys(checkpoint_keys)
        if kind == MAPPED_KEY_SCALE_GRID:
            return torch.float32
        if kind in (MAPPED_KEY_QUANTISED_WEIGHT, MAPPED_KEY_STACKED_BANK):
            return _FP8_DTYPE
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

        WHY THIS STEP EXISTS AT ALL, measured rather than assumed. The
        checkpoint reader decides what to load by iterating
        ``list(model.named_parameters())`` (``utils/checkpoints.py:348``,
        consumed at ``:402``), and torch omits a ``register_parameter(name,
        None)`` declaration from that list. This tree is every parameter
        declared and none materialised, so before this step the list is EMPTY
        and a load that skipped it would read no weight at all and report
        success. Measured on the host's torch:
        ``increments/probe-091-assign-shape.out``,
        ``NAMED_PARAMS_COUNT_ON_NONE_TREE=0``.

        WHY THE PLACEHOLDER CLAIMS NO SHAPE. A shaped placeholder would have to
        carry the per-rank sharded shape, which the modules in this file already
        know and which would become a second place to get it wrong. It does not
        have to: ``UninitializedParameter`` is torch's own
        parameter-without-a-shape-yet, and torch's shape check when the tensors
        arrive is guarded by exactly that case -- ``if not is_param_lazy and
        input_param.shape != param.shape``, with no ``assign`` term in it. An
        ordinary zero-size placeholder is REFUSED there; the lazy one is
        accepted and filled from the checkpoint. Both readings are in the probe
        named above, and it also records that ``assign=False`` raises on a lazy
        parameter, which is why the load below must pass ``assign=True``.

        The placeholder still carries the only two things the reader takes off
        it: the dtype, and the weight loader attached to it.

        The walk is the same one :meth:`declared_parameter_names` does, and is
        repeated here only because registering a parameter needs the module and
        the leaf name rather than the dotted path. The two cannot drift apart
        unnoticed: the acceptance counts this method's result against
        :meth:`declared_parameter_names`, so a walk that visited a different set
        reddens.

        WHY IT IS TWO PASSES AND NOT ONE. Choosing a loader can REFUSE -- an
        expert bank has no loader in this package yet
        (:class:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.Glm5NextExpertBankNotLoadableError`),
        and ``inc-glm53f-095`` is where it gains one. A single pass that
        registered as it went would leave every parameter before the refusing one
        registered and every one after it declared, which is a tree no later load
        can tell apart from a fresh one. So nothing is registered until every
        placeholder and every loader has been built: a refusal leaves the tree
        byte-for-byte as it arrived, and ``named_parameters()`` still reads
        EXACTLY ZERO. The acceptance reads that zero.
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
                # May RAISE. Deliberately before the first registration below.
                #
                # ``owner`` is the module that DECLARED this parameter, and only
                # ``inc-glm53f-095``'s expert-bank loader reads it: the expert
                # geometry lives on ``Glm5NextRoutedExperts`` (``-031``) and
                # nowhere else, so the loader is handed the declaring module
                # rather than deriving a second partition from the key count.
                #
                # ``geometry`` is ``inc-glm53f-094``'s addition and is THIS site's
                # to resolve, because this is the only place that holds both
                # halves: the declaring module, which knows its own widths, and
                # ``self.world_size``, which no declaring class but two is told.
                # It is ``None`` for every replicated family and at world size 1,
                # so the path every landed test exercises is byte-for-byte the one
                # ``inc-glm53f-091`` measured.
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

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Read the checkpoint's weights onto ``device``.

        ``inc-glm53f-091``. This method is what the runner already calls:
        ``neuron_model_runner.py:1299`` invokes ``self.model.load_weights(...)``
        with these three arguments and nothing guards it, so before this
        increment any non-CPU-compile run of this architecture raised
        ``AttributeError`` before a single weight was read. The signature is the
        shipped one, ``qwen3/model.py:966-968``, so no runner file changes.

        Four steps, in an order two of them force:

        1. Open the checkpoint. This is first so that a checkpoint that cannot
           be opened leaves the tree exactly as it was -- every parameter still
           declared and none materialised.
        2. Build the key map at the checkpoint's own quantisation setting with
           the checkpoint's own skip list. The map is what the reader is told to
           look for, so a family missing here is a family never read.
        3. Materialise a placeholder per declared parameter and attach its
           loader. Forced to come before step 4, for the reason
           :meth:`_materialise_declared_parameters` records.
        4. Read the tensors and assign them. ``assign=True`` is required, not
           stylistic: it is what replaces each placeholder with the tensor the
           reader built, and the probe records that ``assign=False`` raises on a
           lazy placeholder.

        SHARDING LANDS WITH ``inc-glm53f-094``, named here rather than left as a
        surprise. The rank resolved here reaches no loader that uses it: neither
        loader this package defines shards, and ``blockwise_scale_loader``'s own
        docstring records that a sharded scale grid "follows the weight's own
        shard geometry and lands with the module that declares that geometry".
        So at a world size above one every rank reads the WHOLE tensor, while
        this file's modules already describe per-rank geometry (``_per_rank``
        above). That gap is the landed position of this package, inherited here
        rather than introduced, and ``inc-glm53f-094`` closes it by attaching the
        per-module shard loaders the pin already ships
        (``utils/weight_loader.py:139-146`` and ``:105-130``, which is how
        ``qwen3`` shards). The rank is resolved and passed now because the
        reader's contract takes one and hands it to every loader transform, so
        that increment needs no change here.

        A ROUTED EXPERT BANK NOW LOADS, AND WHAT STILL REFUSES IS NAMED
        (``inc-glm53f-095``). Step 3 used to raise
        :class:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.Glm5NextExpertBankNotLoadableError`
        on EVERY map entry carrying more than one scale key, because every loader
        this package defined kept ``slices[0][:]`` and dropped 287 of 288 experts
        in silence. The stacking loader serves that entry now. The same error,
        unchanged in class and still NOT caught and re-wrapped here, refuses what
        this package cannot serve honestly: a malformed entry (odd key count, or a
        scale key out of alternation), a multi-weight entry carrying no scale key,
        and a bank whose owning module declares no expert geometry. Leaving it
        unwrapped is a choice: its message already names the parameter and the
        defect, and wrapping it in this method's own error would bury both behind
        a sentence about checkpoints. It is a ``ValueError`` either way, like
        ``Glm5NextWeightLoadError``. The measurement behind the original refusal
        -- expert 0 arriving and 287 experts dropped in silence, with a green
        acceptance over it -- is recorded on the raising function.
        """
        try:
            checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
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

        # The two quantisation settings sit on the TOP-LEVEL config, not on the
        # text config: ``config.py:400`` and ``:408`` are members of
        # ``Glm5NextConfig`` because ``inc-glm53f-079`` lifted them from the
        # checkpoint's top-level ``quantization_config``. ``torch_dtype`` is the
        # text config's own (``:159``). Reading either off the wrong object
        # raises ``AttributeError`` rather than defaulting, which is why this
        # comment names the line for each.
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

        # Everything below runs AFTER the line above, and that is the whole
        # ordering contract of ``inc-glm53f-091b``: the line above is what turns
        # a shape-free placeholder into a real on-device tensor, so a scale read
        # or a prep placed before it would work on placeholders. The acceptance
        # reads the three source positions rather than trusting this comment.
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

        WHY ANY SCALE IS READ OUT OF BAND AT ALL, measured rather than
        preferred. A map entry holding a weight AND its scale is served by
        :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.wrap_with_blockwise_fp8_downscale`,
        whose base transform is ``slices[0][:]``
        (``weight_loaders_fp8.py:1342``) -- it keeps the WEIGHT slice and drops
        the companion, so the scale reaches nothing through the reader.
        Widening the map so each of those scales became its own parameter would
        move ``inc-glm53f-085``'s asserted difference set
        (``test/vllm_neuron/model/glm5_next/test_kv_spec.py:840-848``), and that
        is a lead call this block does not take. So the scales are read here
        instead, on the fork's own landed precedent for exactly this problem:
        ``qwen3/model.py:1050-1072``.

        THE INDEXING IS EXPLICIT, for the reason the precedent states in its own
        comment (``qwen3/model.py:1045-1046``): ``load_sharded_pipelined``
        indexes as a side effect, and a lookup on an unindexed checkpoint
        silently misses. This method is called after that reader has run, so the
        index is already built -- and it calls
        :meth:`~vllm_neuron.utils.checkpoints.SafetensorsCheckpoint._ensure_indexed`
        anyway, because a later caller that reads scales without the pipelined
        load would otherwise get zeros with no complaint.

        AND A MISS IS A REFUSAL, NOT A DEFAULT. The precedent falls back to
        ``torch.ones`` when a key is absent, which is right for a KV scale that
        may legitimately not be published. It is wrong here: a missing block
        scale means the fp8 bytes of that projection cannot be dequantised, and
        a silent 1.0 would make the bytes look like numbers -- pass-shaped
        failure, the same class of defect as the narrowing this increment
        repairs. So an absent key raises and names itself.

        WHERE THE SCALE IS PUT. As a plain attribute on the owning module,
        named the way the consumer already looks it up --
        ``f"{leaf}_{FP8_SCALE_SUFFIX}"``, the name
        :meth:`Glm5NextMLAAttention._dequantised_projection_weight` reads at
        ``:2959``. Plain, not a registered parameter: registering it would add a
        name to ``named_parameters()`` that the map does not carry, which is the
        map widening this method exists to avoid.

        AND A ROUTED EXPERT BANK IS READ HERE TOO, added by ``inc-glm53f-095b``.
        A bank's E scale grids are interleaved with its E weight keys inside ONE
        map entry, so the lone-grid path below cannot see them: it wants a single
        companion scale and finds E. The bank branch asks the landed classifier
        whether an entry is a bank, hands the whole key list to ``-095``'s
        stacking loader, and stores the result under the SAME plain-attribute
        name -- so the bank gains three attributes per layer while
        ``named_parameters()``, ``build_weight_mappings`` and
        ``inc-glm53f-085``'s asserted set stay where they landed. The
        kernel-side consumption of those grids is
        ``inc-glm53f-054``: no module in this file declares a
        ``prepare_scale_operands`` for a bank yet, so the prep loop does not
        visit one.
        """
        checkpoint._ensure_indexed()
        read = 0
        for param_name, keys in mappings.items():
            key_list = [keys] if isinstance(keys, str) else list(keys)
            scales = scale_keys(key_list)
            if classify_mapped_keys(key_list) == MAPPED_KEY_STACKED_BANK:
                # ``inc-glm53f-095b``. A BANK DOES REACH HERE, and it is the one
                # entry shape the lone-grid path below cannot serve: its E scale
                # grids travel interleaved with its E weight keys inside this one
                # entry, so there is no companion entry to read and no single
                # ``scales[0]`` to read it from. The classifier is asked rather
                # than re-derived, so "what is a bank" has one answer in this file
                # and not two.
                #
                # THE RANK IS RESOLVED HERE, not taken as an argument. The
                # loader's transform needs one, and both landed call sites in
                # ``load_weights`` pass this method three positionals; widening
                # its signature to thread a rank through would move landed code
                # for a value ``_resolve_rank`` already owns.
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
                # ``-095``'s landed loader is what de-interleaves and stacks
                # these grids -- this rank's experts through
                # ``local_expert_indices``, each grid compensated by
                # ``compensate_block_scales``, stacked on the leading axis in
                # expert order, so row ``[e]`` matches the weight bank's row
                # ``[e]``. It was landed "to be CALLED rather than attached"
                # (``weight_loaders_fp8.py:2136-2143``) and this is that caller.
                #
                # ``inc-glm53f-101``. The bank's geometry now reaches this call, so
                # a column-sharded bank gets column-sharded grids. ``None`` at
                # expert-parallel degree 1 and at world size 1, which is every
                # landed reading of this branch.
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
            # ``inc-glm53f-094``. A SHARDED WEIGHT'S GRID IS SHARDED WITH IT, and
            # this is the only route such a grid travels. The grid of a dense-MLP
            # projection is not a declared parameter -- it arrives in the same map
            # entry as its weight (``weight_loaders_fp8.py:628-633`` pairs them
            # through ``_quantised``), so it reaches this reader and never
            # ``loader_for_mapped_keys``. Left whole it would describe the FULL
            # weight while the weight itself is this rank's half, and the dequant
            # would scale the wrong blocks: the shard is not optional here.
            #
            # THE GEOMETRY IS THE WEIGHT'S, converted to grid rows by
            # ``shard_geometry_for_grid``, so the block boundary is checked once
            # in one place and a misaligned shard refuses by name. ``None`` at
            # world size 1 and for every replicated family, which is why the
            # landed reading below is the untouched path.
            geometry = _shard_geometry_for(module, leaf, self.world_size)
            if geometry is None:
                grid = checkpoint._get_slice(scales[0])[:]
            else:
                # The block shape is THIS CHECKPOINT'S, threaded rather than
                # defaulted (``inc-glm53f-101``, remedy part 1). Read here, inside
                # the branch that already found a scale grid, so a checkpoint with
                # no grids never has to declare one.
                grid = sharded_scale_grid_loader(
                    geometry, param_name, self._checkpoint_block_size()
                ).load([checkpoint._get_slice(scales[0])], _resolve_rank())
            setattr(module, attribute, grid.to(dtype=torch.float32, device=device))
            read += 1
        return read

    def _run_load_time_preps(self, device: torch.device) -> tuple[int, int]:
        """Call both load-time preps, each behind its own device pre-flight.

        Returns ``(projection prep calls, scale prep calls)``.

        THE SINGLE PRODUCTION CALLER. Before this method
        ``prepare_projection_weights`` and ``prepare_scale_operands`` had ZERO
        production call sites -- every mention in ``vllm_neuron/`` was a test, a
        docstring, a comment or an error-message literal. The acceptance counts
        those sites, so this method is the one place either is called.

        WHY THE PRE-FLIGHT EXISTS, and why it is authored here rather than in the
        preps. Both preps build their operands on the DEVICE OF THE TENSORS THEY
        ARE GIVEN and store them by plain ``setattr`` into a dict attribute that
        ``nn.Module._apply`` never visits. So a prep run before the weights
        reached the device would leave the operands on the CPU **permanently**,
        while every later ``.to(device)`` reported success -- measured in
        ``../../../artifacts/campaigns/glm-5.3-flash-port/increments/probe-091-device-binding.out``
        (``PLAIN_DICT_IS_LEFT_BEHIND=True``). Neither prep checks a device: the
        ``.device`` count inside ``prepare_scale_operands`` is 0 and inside
        ``to_kernel_scale_layout`` is 0. The refusal therefore has to live on the
        caller's side, and ``inc-glm53f-090``'s method and this file's
        shared-expert section stay byte-frozen.
        """
        projection_calls = 0
        scale_calls = 0
        for path, module in self.named_modules():
            if hasattr(type(module), "prepare_projection_weights"):
                names = [n for n, _, _ in module.projection_widths()]
                self._require_prep_operands_on_device(path, module, names, device)
                module.prepare_projection_weights()
                projection_calls += 1
                # ``inc-glm53f-042``. The absorb split reads the PREPARED
                # ``kv_b_proj``, so it has to run after the prep that builds it
                # -- here, immediately after, behind the same device pre-flight
                # and inside the same single production call site. A module that
                # declares no absorb split is skipped by the same ``hasattr``
                # test this loop already uses on the two preps, so the shared
                # expert path is untouched.
                #
                # IT ADDS NO RETURN VALUE ON PURPOSE. This method returns
                # ``(projection calls, scale calls)`` and ``inc-glm53f-091``'s
                # items read that pair; a third element would change a landed
                # contract for a count the acceptance can read straight off the
                # module. So the signature stays exactly as it landed.
                if hasattr(type(module), "prepare_absorb_weights"):
                    module.prepare_absorb_weights()
            # ``inc-glm53f-054a`` hand-off item (iv). The checkpoint's grids are
            # at 128-tile granularity and the shared expert's prep consumes the
            # 256 public grid, so the bridge runs HERE -- on the load path, after
            # the shards are attached and BEFORE the prep reads the grid two
            # branches below. Gated by the same ``hasattr`` test this loop already
            # uses on the preps, so a module that declares no retile is skipped and
            # the routed bank -- which retiles inside its own prep -- is untouched.
            #
            # IT ADDS NO RETURN VALUE, on the precedent recorded for
            # ``prepare_absorb_weights`` above: this method returns
            # ``(projection calls, scale calls)`` and ``inc-glm53f-091``'s items
            # read that pair, so the signature stays exactly as it landed. What the
            # retile did is on the module, in its own health record.
            if hasattr(type(module), "retile_checkpoint_scale_grids"):
                names = [
                    leaf[: -len(_WEIGHT_LEAF_SUFFIX)]
                    for leaf in _scale_prep_leaves(module)
                ]
                self._require_prep_operands_on_device(path, module, names, device)
                module.retile_checkpoint_scale_grids()
            if hasattr(type(module), "prepare_scale_operands"):
                # The projection names come off the module's OWN declaration
                # tuple, so this call cannot ask for a projection the shared
                # expert does not have -- and, since ``inc-glm53f-095b`` factored
                # the derivation into ``_scale_prep_leaves``, only for one whose
                # scale grid is present to be read two lines below. Passed by
                # KEYWORD: the prep takes six positional operands and a silent
                # reordering of three weights against three scales would compute
                # a wrong answer at exactly the right shapes.
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
        return projection_calls, scale_calls

    def _require_prep_operands_on_device(
        self,
        module_path: str,
        module: nn.Module,
        names: list[str],
        device: torch.device,
    ) -> int:
        """Every operand a prep is about to read is already on ``device``.

        Returns how many operands were checked, so a caller can tell a passing
        check from a check that had nothing to look at.

        This is case B of conjunct 8. A weight or scale that is still a
        shape-free placeholder, or sitting on another device, is named here and
        the prep is not called -- because after the prep there is nothing left to
        detect: the operands are built and stored, on whatever device they were
        built on, and no later move touches them. An ABSENT operand is skipped
        here and named by the prep itself (review finding B69-N1: this sentence
        used to claim absent operands were named here, and the loop below
        continues past ``None``).
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
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def get_kv_spec(self) -> KVSpec:
        """One ``LayerSpec`` per layer of the hybrid stack, in layer order.

        Field mapping and naming follow ``llama3/model.py:1781-1795``; the
        construction path does **not**, because that precedent reads
        instantiated submodules and this stack is never instantiated with
        weights. Geometry is read off the declared layer objects instead.

        The two families report different geometry from one uniform read:
        each layer exposes its attention module and that module declares
        ``num_kv_heads_per_rank``, ``head_size``, ``cache_dtype`` and
        ``cache_chunk_size``. ``sliding_window_size`` is ``None`` on every
        entry -- this arch declares no sliding window on either half.

        THE FOUR RECURRENT-STATE FIELDS ARE FILLED BY THE SAME UNIFORM READ.
        A linear-attention layer holds a short-convolution state and a
        recurrent state instead of a key/value history, and reports the two on
        ``LayerSpec``'s four ``kda_*`` fields (``inc-glm53f-015``'s declared
        interface). ``Glm5NextKDAAttention`` derives all four from vLLM's own
        state calculators at its own rank geometry; ``Glm5NextMLAAttention``
        declares none of them. So the read is a DEFAULTING one -- a family that
        carries no recurrent state reports ``None`` on all four by carrying no
        attribute, rather than by this method testing which family it is
        looking at. That keeps the one loop family-blind, which is the property
        ``inc-glm53f-013`` built it for, and it is why the runner recognises the
        two halves BY THE FIELDS THEY CARRY (``neuron_model_runner.py``
        ``:8720-8726``) rather than by a layer name.

        All four move together or not at all. The runner refuses a layer that
        declares part of the geometry (``neuron_model_runner.py:8727-8733``),
        because the conv and recurrent carriers are paired positionally, so a
        partial set would shorten the reported page. One ``getattr`` per field
        against one attribute-carrying class satisfies that by construction.
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
                )
            )
        return KVSpec(layers=layers)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration.forward is a stub created by "
            "inc-glm53f-013; the full 45-layer forward lands with "
            "inc-glm53f-054"
        )
