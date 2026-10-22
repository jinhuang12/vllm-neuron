# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) pair shard map
=======================================

The chunk-pair prefill runs two chunks per step on two groups of 32 ranks, one chunk
each. A group of 32 must hold the whole model, so each rank holds two of the 64
tensor-parallel shards it holds one of today: its own shard ``r``, and the shard of
its partner ``r XOR 48``, which sits in the other group. Every per-shard op then runs
once per held shard at today's widths, and decode keeps reading the own shard alone,
so the own tensors stay exactly today's.

This module is the map and reads no tensor value: which rank is a rank's partner,
which of the model's tensors differ between two ranks' shards, and what the partner
copy is called. :func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.load_pair_shards`
loads what the map lists.

What gets a partner. A tensor gets one exactly when its loader reads the rank: through
a shard geometry (``_SHARD_GEOMETRY`` in ``model_fp8.py``), or, for a routed expert
bank above expert-parallel degree 1, through the expert partition, which gives each
expert-parallel group its own block of experts. Every other tensor is replicated --
the same bytes on every rank -- so the own copy serves both shards and a second copy
would only spend HBM.

What a partner is called and where it lives. Beside its own counterpart, on the same
module, named ``partner_<leaf>``, in the same form: a registered parameter where the
own tensor is a parameter, a plain attribute where the own tensor is one of the scale
grids the out-of-band reader stores as plain attributes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import nn

from .weight_loaders_fp8 import (
    MAPPED_KEY_SCALE_GRID,
    MAPPED_KEY_STACKED_BANK,
    AnyShardGeometry,
    Glm5NextWeightMapError,
    _as_key_list,
    build_weight_mappings,
    classify_mapped_keys,
    scale_keys,
)

#: The world size the pair layout is defined on: the 64-rank trn2 host, 16 chips at
#: four ranks per chip (chip = rank // 4), split into two prefill groups of 32
#: (group = rank // 32).
PAIR_WORLD_SIZE = 64

#: Rank ``r``'s partner is ``r ^ PAIR_PARTNER_XOR``, which flips rank bits 4 and 5.
#: Bit 5 moves the rank to the other prefill group, ranks 0-31 with ranks 32-63; the
#: two bits together move it from chip ``d`` to chip ``d ^ 12``, at the same place
#: on the chip (``rank % 4`` is kept). The partners of ranks 0-31 are 48-63 then
#: 32-47, the whole other group, so each group holds all 64 shards. A process group
#: that pairs ranks for this layout pairs them by :func:`partner_rank`.
PAIR_PARTNER_XOR = 48


class Glm5NextPairLayoutError(ValueError):
    """A rank or world size the pair shard layout has no partner map for."""


#: The leaf-name prefix every partner tensor carries.
PARTNER_PREFIX = "partner_"

#: A partner that is a registered parameter, as its own counterpart is.
PARTNER_PARAMETER = "parameter"

#: A partner scale grid of one weight. The out-of-band reader stores the own grid as
#: a plain attribute rather than a parameter, and so does the partner load.
PARTNER_SCALE_GRID = "scale_grid"

#: A routed bank's partner scale grids, one per local expert, stacked on the leading
#: axis as the bank's own are.
PARTNER_BANK_SCALE_GRID = "bank_scale_grid"


def partner_rank(rank: int, world_size: int) -> int:
    """The rank whose shard ``rank`` holds beside its own.

    Raises :class:`Glm5NextPairLayoutError` for a world size the layout is not
    defined on, rather than inventing a pairing for it -- the XOR mask is a fact of
    the 64-rank mesh, not a rule that scales -- and for a rank outside the world.
    """
    if int(world_size) != PAIR_WORLD_SIZE:
        raise Glm5NextPairLayoutError(
            f"the pair shard layout is defined on {PAIR_WORLD_SIZE} ranks (two "
            f"prefill groups of {PAIR_WORLD_SIZE // 2}); world size {world_size} has "
            f"no partner map"
        )
    if not 0 <= int(rank) < PAIR_WORLD_SIZE:
        raise Glm5NextPairLayoutError(
            f"rank {rank} is outside the {PAIR_WORLD_SIZE}-rank world, so it has no "
            f"partner"
        )
    return int(rank) ^ PAIR_PARTNER_XOR


def partner_name(leaf: str) -> str:
    """The attribute a partner tensor takes on the module that holds ``leaf``."""
    return f"{PARTNER_PREFIX}{leaf}"


@dataclass(frozen=True)
class PartnerTensor:
    """One tensor a rank holds twice: where it lives, what it is read from, its form.

    ``geometry`` is the own tensor's shard geometry, unchanged: a partner is the same
    shard read at the partner's rank, never a different cut. For a scale grid it is
    the geometry of the grid's weight, which the grid loaders convert to grid rows.
    """

    module_path: str
    #: The module that holds the own tensor and will hold the partner. Excluded from
    #: equality, since a module compares by identity.
    module: nn.Module = field(compare=False, repr=False)
    #: The own tensor's attribute name on ``module``.
    leaf: str
    #: The weight map's parameter name the checkpoint keys belong to. For a parameter
    #: it is the own tensor's own path; for a scale grid it is its weight's.
    param_name: str
    checkpoint_keys: tuple[str, ...]
    geometry: AnyShardGeometry | None
    #: :data:`PARTNER_PARAMETER`, :data:`PARTNER_SCALE_GRID` or
    #: :data:`PARTNER_BANK_SCALE_GRID`.
    form: str
    #: The dtype the partner is stored in: the own parameter's placeholder dtype, which
    #: the reader casts to, or float32 for a grid, as the out-of-band reader stores one.
    dtype: torch.dtype
    #: True for a routed expert bank, whose loader also needs the partner's placement
    #: on the expert-parallel mesh.
    is_bank: bool

    @property
    def partner_leaf(self) -> str:
        return partner_name(self.leaf)

    @property
    def partner_path(self) -> str:
        """The partner tensor's dotted path in the module tree."""
        if not self.module_path:
            return self.partner_leaf
        return f"{self.module_path}.{self.partner_leaf}"


def _differs_between_shards(
    kind: str, geometry: AnyShardGeometry | None, owner
) -> bool:
    """Whether the loader of this entry reads the rank, so two ranks' shards differ.

    The two ways a loader here reads it: a shard geometry, which cuts by the rank; or
    a routed bank above expert-parallel degree 1, whose expert block is chosen by the
    rank's expert-parallel index even when no width is divided (world size equal to
    the degree). Every other loader in ``loader_for_mapped_keys`` is rank-blind.
    """
    if geometry is not None:
        return True
    return kind == MAPPED_KEY_STACKED_BANK and int(getattr(owner, "ep_degree", 1)) > 1


def _the_load_weight_map(model: nn.Module) -> dict[str, str | list[str]]:
    """The text weight map ``load_weights`` reads the own shard by, built the same way.

    The same call with the same three inputs, so the partner load reads the same keys
    for the same parameters. ``load_weights`` adds a vision tower's entries for its
    reader alone; a vision tower is replicated and has no partner.
    """
    return build_weight_mappings(
        model.text_config,
        quantised=model.config.is_block_quantized,
        modules_to_not_convert=tuple(model.config.modules_to_not_convert or ()),
    )


def partner_tensor_plan(model: nn.Module, world_size: int) -> tuple[PartnerTensor, ...]:
    """Every tensor of ``model`` that differs between two ranks' shards, in load order.

    Built from the module tree and the weight map alone, so it lists the same tensors
    before any is loaded, on the shape-only meta pass, and after the load-time preps
    have released the own checkpoint tensors.

    Two passes, mirroring the two places the own load puts checkpoint tensors:

    1. The declared parameters, as ``_materialise_declared_parameters`` walks them:
       each with its own map entry, its own shard geometry and its own placeholder
       dtype, so the partner is cut, read and cast as the own one is.
    2. The scale grids ``_load_out_of_band_scales`` reads into plain attributes,
       selected as it selects them: a routed bank's interleaved grids, and the one
       scale companion of a weight whose grid is not a declared parameter of its own.

    A tensor whose loader is rank-blind is left out (:func:`_differs_between_shards`).
    """
    # The shard table's single reader, imported here because the model module
    # imports the weight loaders this module builds on.
    from .model_fp8 import _shard_geometry_for

    mappings = _the_load_weight_map(model)
    plan: list[PartnerTensor] = []

    for module_path, module in model.named_modules():
        for leaf in getattr(module, "declared_param_names", ()):
            name = f"{module_path}.{leaf}" if module_path else leaf
            keys = mappings.get(name, name)
            kind = classify_mapped_keys(keys)
            geometry = _shard_geometry_for(module, leaf, world_size)
            if not _differs_between_shards(kind, geometry, module):
                continue
            plan.append(
                PartnerTensor(
                    module_path=module_path,
                    module=module,
                    leaf=leaf,
                    param_name=name,
                    checkpoint_keys=tuple(_as_key_list(keys)),
                    geometry=geometry,
                    form=PARTNER_PARAMETER,
                    dtype=model._placeholder_dtype(
                        keys, param_name=name, mappings=mappings
                    ),
                    is_bank=kind == MAPPED_KEY_STACKED_BANK,
                )
            )

    plan.extend(_out_of_band_grid_plan(model, mappings, world_size))
    return tuple(plan)


def _out_of_band_grid_plan(
    model: nn.Module,
    mappings: Mapping[str, str | list[str]],
    world_size: int,
) -> list[PartnerTensor]:
    """The partner entries of the grids ``_load_out_of_band_scales`` reads.

    Selected by the same three tests in the same order as that reader: a bank entry
    takes all its keys; otherwise only an entry holding a weight and exactly one scale
    key; and of those, only one whose scale key is not a declared grid parameter of its
    own (that one is a parameter, and pass 1 of :func:`partner_tensor_plan` lists it).
    The grid takes the name the reader stores it under, ``<weight>_weight_scale_inv``,
    and the geometry of its weight, as the reader shards it.
    """
    from .model_fp8 import _shard_geometry_for

    declared_grids = {
        _as_key_list(keys)[0]
        for keys in mappings.values()
        if classify_mapped_keys(keys) == MAPPED_KEY_SCALE_GRID
    }
    plan: list[PartnerTensor] = []
    for param_name, keys in mappings.items():
        key_list = tuple(_as_key_list(keys))
        kind = classify_mapped_keys(key_list)
        scales = scale_keys(key_list)
        if kind == MAPPED_KEY_STACKED_BANK:
            form, read_keys = PARTNER_BANK_SCALE_GRID, key_list
        elif (
            len(key_list) >= 2 and len(scales) == 1 and scales[0] not in declared_grids
        ):
            form, read_keys = PARTNER_SCALE_GRID, (scales[0],)
        else:
            continue
        module_path, _, leaf = param_name.rpartition(".")
        module = model.get_submodule(module_path) if module_path else model
        geometry = _shard_geometry_for(module, leaf, world_size)
        if not _differs_between_shards(kind, geometry, module):
            continue
        attribute = model._sibling_scale_grid_name(leaf)
        if not attribute:
            raise Glm5NextWeightMapError(
                f"{param_name} carries scale keys but its leaf name is not a weight "
                f"leaf, so its grid has no name to be stored under; the map and the "
                f"out-of-band reader disagree about what a weight is called"
            )
        plan.append(
            PartnerTensor(
                module_path=module_path,
                module=module,
                leaf=attribute,
                param_name=param_name,
                checkpoint_keys=read_keys,
                geometry=geometry,
                form=form,
                dtype=torch.float32,
                is_bank=kind == MAPPED_KEY_STACKED_BANK,
            )
        )
    return plan


class LoadedPartner(NamedTuple):
    """One partner tensor as the load attached it."""

    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass(frozen=True)
class PairShardLoad:
    """What one rank's partner load attached, for the caller's log and HBM account."""

    rank: int
    partner: int
    tensors: tuple[LoadedPartner, ...]

    @property
    def partner_bytes(self) -> int:
        """The bytes the partner shard adds on the device, as loaded."""
        return sum(tensor.nbytes for tensor in self.tensors)
