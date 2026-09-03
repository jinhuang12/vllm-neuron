# SPDX-License-Identifier: Apache-2.0
"""S0 — the ONE shared full-partition resolver, its registry, and S0-INV-1.

WHY THIS FILE EXISTS
    A collective's ``replica_groups`` used to be THIS RANK'S OWN TILE. Two
    resolvers produced it independently -- ``overrides/xla_collectives.py``
    (the HLO-emission path) and ``fx_passes/collective_replica_groups_pass.py``
    (the cache-key path) -- and both returned ``[dist.get_process_group_ranks(g)]``.

    The consequence is a compile-cache population fact, not a numerics one: the
    16 compile keys observed in production are **2 graph shapes x 8 rank-tiles**,
    not 16 buckets. The same graph hashes 8 ways because rank 0 renders its own
    tile and rank 8 renders a different one. Resolving the FULL PARTITION
    instead makes the rendering rank-independent, and 16 keys collapse to 2.

    ``all_to_all_single``'s own docstring in ``overrides/xla_collectives.py``
    already SPECIFIES the full partition ("replica_groups must be a complete
    partition covering every rank exactly once. We build the full partition
    from all registered sibling process groups") while the code it documents
    returned the local tile. This module is the "registered sibling process
    groups" that sentence presupposes.

THE THREE ARMS, in order
    1. REGISTRY HIT -> the registered sibling tiles of that group.
       **NEVER synthesized by size.** ``parallel/neuron_parallel_state.py``
       builds an 8x8 TRN2 mesh that is HARD-CODED AND STRIDED
       (``[0,1,2,3,12,13,14,15]``, ``[4,5,6,7,8,9,10,11]``, ...) whenever
       ``total == 64 and row_size == 8`` and ``VLLM_NEURON_SWITCH_CC`` is
       unset, and CONTIGUOUS in the ``else`` branch. Two different tilings
       live in one file, so a size-derived ``range(start, start + n)`` would be
       silently WRONG for any group built through the strided path. Only the
       registry knows which tiling a given group actually used.
    2. WORLD-WIDE GROUP -> ``[ranks]``. A group whose rank list IS the whole
       world already is a complete partition; returning it is not synthesis.
       The TP, EP and lm_head groups are created outside ``_build_subgroup``
       and land here.
    3. OTHERWISE -> RAISE ``ReplicaGroupResolutionError``.

    Arm 3 is deliberately loud, and arm 3 alone is NOT sufficient: on the
    hashing path a bare ``except Exception`` used to swallow the raise and
    return ``None``, which DROPPED the replica-groups component from the cache
    key rather than failing. Two graphs differing only in parallel topology
    would then hash identically. The caller in ``compile/cache.py`` now
    re-raises this dedicated type explicitly, and the canonicalization step
    that rewrites the graph runs OUTSIDE that ``try`` on purpose.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

logger = logging.getLogger(__name__)

__all__ = [
    "ReplicaGroupResolutionError",
    "assert_is_full_partition",
    "canonical_partition_token",
    "clear_registry",
    "lookup_full_partition",
    "register_full_partition",
    "registry_size",
    "resolve_full_partition",
]


class ReplicaGroupResolutionError(RuntimeError):
    """A collective's full rank partition could not be established.

    A DEDICATED type, not ``RuntimeError``, so the hashing path can re-raise
    exactly this and keep its pre-existing broad ``except`` for the unrelated
    "torch.distributed is not ready yet" cases. Raised both by the resolver's
    arm 3 and by the S0-INV-1 assertion.
    """


# group_name -> the full partition that group belongs to, and the world size it
# was built against. Written once per created group, read by the resolver.
_PARTITION_BY_GROUP_NAME: Dict[str, Tuple[Tuple[Tuple[int, ...], ...], int]] = {}
_REGISTRY_LOCK = threading.Lock()


def _canonical_tiles(
    partition: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Freeze a partition into a canonical, rank-independent tile order.

    Tiles are sorted by their FIRST rank. Order WITHIN a tile is preserved
    untouched: for ``reduce_scatter``/``all_gather`` the position of a rank
    inside its replica group decides which shard it owns, so sorting inside a
    tile would silently permute data. Order ACROSS tiles carries no such
    meaning, which is exactly why sorting there is safe and is what makes the
    rendering identical on every rank.
    """
    tiles = tuple(tuple(int(r) for r in tile) for tile in partition)
    return tuple(sorted(tiles, key=lambda t: (t[0] if t else -1, t)))


def assert_is_full_partition(
    partition: Sequence[Sequence[int]],
    world_size: int,
    context: str,
) -> None:
    """S0-INV-1. Every tile the same length, and the tiles tile the world once.

    THIS IS AN ASSERTION IN CODE, NOT A COMMENT, because S0 silently changes
    the meaning of two derivations that read ``replica_groups[0]``:

        ``overrides/xla_collectives.py`` ``shard_count = len(replica_groups[0])``
        (reduce_scatter) and ``split_count = ... else len(replica_groups[0])``
        (all_to_all_single, whose one call site passes EMPTY split lists).

    Before S0 ``replica_groups[0]`` was THIS RANK'S OWN tile. After S0 it is the
    FIRST tile of the full partition -- a different object. For the deployed
    topology both are length 8 so the value does not move, but they are equal
    ONLY while every tile has the same size. Without this assertion S0 would
    convert a would-be crash into a silently wrong ``shard_count``.

    Raises:
        ReplicaGroupResolutionError: on a non-uniform, gapped, duplicated or
            non-covering partition.
    """
    tiles = _canonical_tiles(partition)
    if not tiles:
        raise ReplicaGroupResolutionError(
            f"{context}: empty replica-group partition"
        )

    widths = {len(t) for t in tiles}
    if len(widths) != 1:
        raise ReplicaGroupResolutionError(
            f"{context}: replica-group tiles are NON-UNIFORM (widths "
            f"{sorted(widths)}); `len(replica_groups[0])` is used as "
            f"shard_count/split_count and would be wrong for some tile. "
            f"partition={[list(t) for t in tiles]}"
        )

    flat = [r for tile in tiles for r in tile]
    if len(flat) != len(set(flat)):
        duplicated = sorted({r for r in flat if flat.count(r) > 1})
        raise ReplicaGroupResolutionError(
            f"{context}: rank(s) {duplicated} appear in more than one tile; "
            f"replica_groups must cover every rank EXACTLY once. "
            f"partition={[list(t) for t in tiles]}"
        )
    if set(flat) != set(range(world_size)):
        missing = sorted(set(range(world_size)) - set(flat))
        extra = sorted(set(flat) - set(range(world_size)))
        raise ReplicaGroupResolutionError(
            f"{context}: replica-group tiles are not a partition of "
            f"range({world_size}) -- missing={missing} unexpected={extra}. "
            f"partition={[list(t) for t in tiles]}"
        )


def canonical_partition_token(partition: Sequence[Sequence[int]]) -> str:
    """A deterministic, rank-independent string for a full rank partition.

    This is what replaces the opaque ``group_name`` in the HASHED copy of the
    graph. ``torch.distributed`` names groups from a monotonically increasing
    counter (``_process_group_name`` -> ``str(_world.group_count)``), so rank 0
    prints ``"N"`` where rank 8 prints ``"N+1"`` for the SAME logical
    collective, and ``str(gm.graph)`` renders that name verbatim. Substituting
    the partition removes the rank from the rendering.

    Distinct partitions stay distinct. Identical partitions reached through
    different group objects COLLAPSE, which is correct: the compiler is only
    ever handed ``replica_groups``, so two such graphs really are the same
    compilation.
    """
    tiles = _canonical_tiles(partition)
    return "pg:" + ";".join(",".join(str(r) for r in tile) for tile in tiles)


def register_full_partition(
    created: Sequence[Tuple[object, Sequence[int]]],
    *,
    world_size: int,
) -> int:
    """Register every tile of ONE ``_build_subgroup`` call under EVERY name.

    ``created`` is the ``(process_group, ranks)`` pair for each tile the caller
    built, in construction order -- including the tiles this rank does NOT
    belong to. That is the whole point: ``dist.new_group`` is collective, so
    every rank really does construct every sibling tile, but the caller's loop
    variable is overwritten each iteration and only the rank's own handle
    survives. The sibling groups exist in ``torch.distributed``'s global state
    and are reachable by name; what did not exist before S0 was any map from a
    name to the partition it belongs to. ``_resolve_process_group(name)``
    returns ONE group, never its siblings.

    Consistency across ranks rests on a mechanism, not on luck: the loop
    bounds, the loop order and the ``new_group`` call sequence are identical on
    every rank, and ``torch.distributed`` assigns each group's name from a
    monotonically increasing counter, so the Nth ``new_group`` call gets the
    same name string everywhere. Rank 0's OWN group is tile 0 and rank 8's is
    tile 1 -- DIFFERENT names -- and that difference is exactly the 8-way key
    split. Under S0 both names map to the SAME partition and therefore to the
    same canonical token.

    Returns:
        the number of group names registered (0 when nothing was resolvable).
    """
    partition = [list(ranks) for _group, ranks in created]
    if not partition:
        return 0

    # Fail at REGISTRATION, where the offending construction is on the stack,
    # rather than later inside a cache-key computation.
    assert_is_full_partition(
        partition, world_size, "register_full_partition"
    )
    canonical = _canonical_tiles(partition)

    registered = 0
    with _REGISTRY_LOCK:
        for group, _ranks in created:
            name = getattr(group, "group_name", None)
            if not isinstance(name, str):
                # A backend that does not expose ``group_name`` cannot be
                # looked up by the resolver either, so skip it rather than
                # registering an unusable entry. Arm 2/3 still apply to it.
                logger.debug(
                    "replica-group registry: group %r exposes no str "
                    "group_name; not registered",
                    group,
                )
                continue
            _PARTITION_BY_GROUP_NAME[name] = (canonical, int(world_size))
            registered += 1

    logger.debug(
        "replica-group registry: registered %d name(s) for partition %s "
        "(world_size=%d)",
        registered,
        [list(t) for t in canonical],
        world_size,
    )
    return registered


def lookup_full_partition(group_name: str) -> Optional[List[List[int]]]:
    """Arm 1's raw lookup. ``None`` on a miss -- callers fall through."""
    with _REGISTRY_LOCK:
        entry = _PARTITION_BY_GROUP_NAME.get(group_name)
    if entry is None:
        return None
    canonical, _world_size = entry
    return [list(t) for t in canonical]


def registry_size() -> int:
    """Number of registered group names. For tests and debug lines only."""
    with _REGISTRY_LOCK:
        return len(_PARTITION_BY_GROUP_NAME)


def clear_registry() -> None:
    """Drop every registration. For tests only; never called in serving."""
    with _REGISTRY_LOCK:
        _PARTITION_BY_GROUP_NAME.clear()


def resolve_full_partition(group_name: str) -> List[List[int]]:
    """The ONE resolver. Return the COMPLETE rank partition for ``group_name``.

    Authored once and imported by both the HLO-emission path
    (``overrides/xla_collectives.py``) and the cache-key path
    (``fx_passes/collective_replica_groups_pass.py``) so the two can never
    drift again. Every arm's result passes S0-INV-1 before it is returned.

    Raises:
        ReplicaGroupResolutionError: arm 3, or an S0-INV-1 violation.
    """
    # ARM 1 -- registry hit. Checked FIRST and never second-guessed by size:
    # the deployed 8x8 TRN2 mesh is strided, so size tells you nothing.
    registered = lookup_full_partition(group_name)
    if registered is not None:
        with _REGISTRY_LOCK:
            _canonical, world_size = _PARTITION_BY_GROUP_NAME[group_name]
        assert_is_full_partition(
            registered, world_size, f"arm 1 registry hit for {group_name!r}"
        )
        return registered

    # Resolve the group itself for arms 2 and 3.
    try:
        group = _resolve_process_group(group_name)
        ranks = list(dist.get_process_group_ranks(group))
    except ReplicaGroupResolutionError:
        raise
    except Exception as exc:
        raise ReplicaGroupResolutionError(
            f"could not resolve process group {group_name!r}: {exc}"
        ) from exc

    # ARM 2 -- the group IS the world, so it already is a complete partition.
    world_size = dist.get_world_size()
    if len(ranks) == world_size:
        partition = [ranks]
        assert_is_full_partition(
            partition, world_size, f"arm 2 world group {group_name!r}"
        )
        return partition

    # ARM 3 -- loud. A subgroup nobody registered cannot be completed by
    # guesswork: synthesizing `range(start, start + len(ranks))` is exactly the
    # error the strided TRN2 mesh punishes, and returning the bare local tile is
    # the rank-dependent rendering S0 exists to remove.
    raise ReplicaGroupResolutionError(
        f"process group {group_name!r} has {len(ranks)} of {world_size} ranks "
        f"({ranks}) and is NOT in the full-partition registry. A partial group "
        f"must be registered by whoever created it (see "
        f"`register_full_partition`); its sibling tiles cannot be synthesized "
        f"from the size, because the deployed 8x8 TRN2 mesh in "
        f"`parallel/neuron_parallel_state.py` is hard-coded and STRIDED, not "
        f"contiguous."
    )
