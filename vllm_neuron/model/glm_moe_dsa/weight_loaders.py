# SPDX-License-Identifier: Apache-2.0
"""Header-only checkpoint planning and rank-local tensor slicing for GLM-5.2.

The full interface validates every indexed key and safetensors header. The
lightweight interface reads only ``model.safetensors.index.json``. Neither
interface loads checkpoint tensor payloads.
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from .quantization import PINNED_FP8, Fp8BlockQuantization, ScaleCoverage

PINNED_INDEX_KEY_COUNT = 118_629
PINNED_SHARD_COUNT = 141
PINNED_TOTAL_SIZE = 755_617_140_416
PINNED_TP_SIZE = 64
PINNED_EP_SIZE = 64


class UnexpectedCheckpointKey(ValueError):
    """A checkpoint key is outside the pinned main-model and MTP grammar."""


class HeaderMismatch(ValueError):
    """The index and safetensors headers disagree."""


class Disposition(str, Enum):
    LOAD_TARGET = "load_target"
    FP8_SCALE = "fp8_scale_metadata"
    INTENTIONAL_SKIP = "intentional_main_workload_skip"


class TensorCategory(str, Enum):
    OUTER = "outer"
    LAYER_NORM = "layer_norm"
    ATTENTION = "attention"
    DENSE_MLP = "dense_mlp"
    ROUTER = "router"
    ROUTED_EXPERT = "routed_expert"
    SHARED_EXPERT = "shared_expert"
    MTP = "mtp"


@dataclass(frozen=True)
class GlmMoeDsaCheckpointContract:
    """Pinned layer topology used to reject unknown or misplaced keys."""

    num_hidden_layers: int = 78
    num_nextn_predict_layers: int = 1
    first_k_dense_replace: int = 3
    num_routed_experts: int = 256
    indexer_layer_indices: tuple[int, ...] = (
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
        78,
    )

    @property
    def mtp_layer_indices(self) -> tuple[int, ...]:
        start = self.num_hidden_layers
        return tuple(range(start, start + self.num_nextn_predict_layers))

    @property
    def all_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + self.num_nextn_predict_layers))


PINNED_CONTRACT = GlmMoeDsaCheckpointContract()


@dataclass(frozen=True)
class TensorHeader:
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class KeyInfo:
    key: str
    disposition: Disposition
    category: TensorCategory
    layer_index: int | None = None
    expert_index: int | None = None
    is_scale: bool = False


@dataclass(frozen=True)
class ManifestEntry:
    info: KeyInfo
    header: TensorHeader

    @property
    def key(self) -> str:
        return self.info.key


@dataclass(frozen=True)
class CheckpointIndex:
    path: Path
    key_to_shard: Mapping[str, str]
    total_size: int

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_key_count: int | None = None,
    ) -> CheckpointIndex:
        path = Path(path)
        with path.open() as index_file:
            raw = json.load(index_file)
        if set(raw) != {"metadata", "weight_map"}:
            raise ValueError(f"Unexpected index fields: {sorted(raw)}")
        weight_map = raw["weight_map"]
        metadata = raw["metadata"]
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(shard, str)
            for key, shard in weight_map.items()
        ):
            raise ValueError("weight_map must map tensor names to shard names")
        if expected_key_count is not None and len(weight_map) != expected_key_count:
            raise ValueError(
                f"Checkpoint index has {len(weight_map)} keys; "
                f"expected {expected_key_count}"
            )
        total_size = metadata.get("total_size")
        if not isinstance(total_size, int) or total_size <= 0:
            raise ValueError("metadata.total_size must be a positive integer")
        return cls(path=path, key_to_shard=weight_map, total_size=total_size)

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.key_to_shard.values())))


@dataclass(frozen=True)
class CheckpointManifest:
    index: CheckpointIndex
    entries: tuple[ManifestEntry, ...]

    @property
    def by_key(self) -> dict[str, ManifestEntry]:
        return {entry.key: entry for entry in self.entries}

    @property
    def disposition_counts(self) -> Counter[Disposition]:
        return Counter(entry.info.disposition for entry in self.entries)

    @property
    def category_counts(self) -> Counter[TensorCategory]:
        return Counter(entry.info.category for entry in self.entries)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_ROUTED_EXPERT_RE = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale_inv)$"
)
_SHARED_EXPERT_RE = re.compile(
    r"^mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale_inv)$"
)
_DENSE_RE = re.compile(
    r"^mlp\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)

_OUTER_KEYS = {
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
}
_LAYER_NORM_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
}
_ATTENTION_SUFFIXES = {
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_layernorm.weight",
    *{
        f"self_attn.{projection}.{tail}"
        for projection in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        )
        for tail in ("weight", "weight_scale_inv")
    },
}
_INDEXER_SUFFIXES = {
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wk.weight_scale_inv",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wq_b.weight_scale_inv",
}
_ROUTER_SUFFIXES = {
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
}
_MTP_SPECIAL_SUFFIXES = {
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
}


def _is_scale_suffix(suffix: str) -> bool:
    return suffix.endswith(".weight_scale_inv")


def classify_checkpoint_key(
    key: str,
    contract: GlmMoeDsaCheckpointContract = PINNED_CONTRACT,
) -> KeyInfo:
    """Classify one exact key or fail closed for an unknown key."""
    if key in _OUTER_KEYS:
        return KeyInfo(key, Disposition.LOAD_TARGET, TensorCategory.OUTER)

    match = _LAYER_RE.match(key)
    if match is None:
        raise UnexpectedCheckpointKey(f"Unexpected checkpoint key: {key}")
    layer_index = int(match.group(1))
    suffix = match.group(2)
    if layer_index not in contract.all_layer_indices:
        raise UnexpectedCheckpointKey(
            f"Layer {layer_index} is outside the pinned main and MTP topology: {key}"
        )

    category: TensorCategory | None = None
    expert_index: int | None = None
    if suffix in _LAYER_NORM_SUFFIXES:
        category = TensorCategory.LAYER_NORM
    elif suffix in _ATTENTION_SUFFIXES:
        category = TensorCategory.ATTENTION
    elif suffix in _INDEXER_SUFFIXES:
        if layer_index not in contract.indexer_layer_indices:
            raise UnexpectedCheckpointKey(
                f"Indexer tensor is on an undeclared indexer layer: {key}"
            )
        category = TensorCategory.ATTENTION
    elif (dense_match := _DENSE_RE.match(suffix)) is not None:
        if layer_index >= contract.first_k_dense_replace:
            raise UnexpectedCheckpointKey(f"Dense MLP tensor is on a MoE layer: {key}")
        category = TensorCategory.DENSE_MLP
    elif suffix in _ROUTER_SUFFIXES:
        if layer_index < contract.first_k_dense_replace:
            raise UnexpectedCheckpointKey(f"Router tensor is on a dense layer: {key}")
        category = TensorCategory.ROUTER
    elif (expert_match := _ROUTED_EXPERT_RE.match(suffix)) is not None:
        if layer_index < contract.first_k_dense_replace:
            raise UnexpectedCheckpointKey(f"Expert tensor is on a dense layer: {key}")
        expert_index = int(expert_match.group(1))
        if not 0 <= expert_index < contract.num_routed_experts:
            raise UnexpectedCheckpointKey(
                f"Routed expert {expert_index} is outside "
                f"0..{contract.num_routed_experts - 1}: {key}"
            )
        category = TensorCategory.ROUTED_EXPERT
    elif _SHARED_EXPERT_RE.match(suffix) is not None:
        if layer_index < contract.first_k_dense_replace:
            raise UnexpectedCheckpointKey(
                f"Shared expert tensor is on a dense layer: {key}"
            )
        category = TensorCategory.SHARED_EXPERT
    elif suffix in _MTP_SPECIAL_SUFFIXES:
        if layer_index not in contract.mtp_layer_indices:
            raise UnexpectedCheckpointKey(f"MTP-only tensor is on a main layer: {key}")
        category = TensorCategory.MTP

    if category is None:
        raise UnexpectedCheckpointKey(f"Unexpected checkpoint key: {key}")

    is_scale = _is_scale_suffix(suffix)
    if layer_index in contract.mtp_layer_indices:
        disposition = Disposition.INTENTIONAL_SKIP
        category = TensorCategory.MTP
    elif is_scale:
        disposition = Disposition.FP8_SCALE
    else:
        disposition = Disposition.LOAD_TARGET
    return KeyInfo(
        key=key,
        disposition=disposition,
        category=category,
        layer_index=layer_index,
        expert_index=expert_index,
        is_scale=is_scale,
    )


def _read_safetensors_header(path: Path, shard_name: str) -> dict[str, TensorHeader]:
    """Read only the JSON header at the start of one safetensors shard."""
    with path.open("rb") as shard_file:
        prefix = shard_file.read(8)
        if len(prefix) != 8:
            raise HeaderMismatch(f"Safetensors file is too short: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise HeaderMismatch(f"Invalid safetensors header length in {path}")
        raw = json.loads(shard_file.read(header_length))

    result: dict[str, TensorHeader] = {}
    for key, metadata in raw.items():
        if key == "__metadata__":
            continue
        try:
            dtype = metadata["dtype"]
            shape = tuple(int(size) for size in metadata["shape"])
            offsets = tuple(int(value) for value in metadata["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise HeaderMismatch(f"Malformed header entry {key!r} in {path}") from error
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise HeaderMismatch(f"Invalid data offsets for {key!r} in {path}")
        result[key] = TensorHeader(dtype, shape, shard_name, offsets)
    return result


def load_checkpoint_index(
    index_path: str | Path,
    *,
    expected_key_count: int | None = PINNED_INDEX_KEY_COUNT,
) -> CheckpointIndex:
    """Lightweight interface: read and validate the index, with no shard opens."""
    return CheckpointIndex.from_file(index_path, expected_key_count=expected_key_count)


def load_checkpoint_manifest(
    index_path: str | Path,
    *,
    contract: GlmMoeDsaCheckpointContract = PINNED_CONTRACT,
    quantization: Fp8BlockQuantization = PINNED_FP8,
    expected_key_count: int | None = PINNED_INDEX_KEY_COUNT,
) -> CheckpointManifest:
    """Full metadata interface: validate all index keys and all shard headers."""
    index = load_checkpoint_index(index_path, expected_key_count=expected_key_count)
    base = index.path.parent
    expected_by_shard: dict[str, set[str]] = {
        shard: set() for shard in index.shard_names
    }
    for key, shard in index.key_to_shard.items():
        expected_by_shard[shard].add(key)

    headers: dict[str, TensorHeader] = {}
    for shard in index.shard_names:
        shard_headers = _read_safetensors_header(base / shard, shard)
        expected = expected_by_shard[shard]
        actual = set(shard_headers)
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise HeaderMismatch(
                f"Index/header mismatch for {shard}: missing={missing}, extra={extra}"
            )
        headers.update(shard_headers)

    entries = tuple(
        ManifestEntry(classify_checkpoint_key(key, contract), headers[key])
        for key in sorted(index.key_to_shard)
    )
    by_key = {entry.key: entry for entry in entries}
    for entry in entries:
        if not entry.info.is_scale:
            continue
        weight_key = entry.key.removesuffix("_scale_inv")
        weight = by_key.get(weight_key)
        if weight is None:
            raise HeaderMismatch(f"Inverse scale has no weight: {entry.key}")
        quantization.validate_header_pair(
            weight_dtype=weight.header.dtype,
            weight_shape=weight.header.shape,
            scale_dtype=entry.header.dtype,
            scale_shape=entry.header.shape,
        )
    for entry in entries:
        if entry.header.dtype != quantization.checkpoint_weight_dtype:
            continue
        scale_key = f"{entry.key}_scale_inv"
        if scale_key not in by_key:
            raise HeaderMismatch(f"FP8 weight has no inverse scale: {entry.key}")

    return CheckpointManifest(index=index, entries=entries)


@dataclass(frozen=True)
class TPShardSpec:
    """Raw checkpoint shard contract for one tensor."""

    global_shape: tuple[int, ...]
    shard_dim: int | None
    world_size: int = PINNED_TP_SIZE

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.shard_dim is not None:
            if not 0 <= self.shard_dim < len(self.global_shape):
                raise ValueError(f"Invalid shard_dim {self.shard_dim}")
            dimension = self.global_shape[self.shard_dim]
            if dimension % self.world_size:
                raise ValueError(
                    f"dimension {dimension} is not divisible by TP={self.world_size}"
                )

    @property
    def local_shape(self) -> tuple[int, ...]:
        if self.shard_dim is None:
            return self.global_shape
        shape = list(self.global_shape)
        shape[self.shard_dim] //= self.world_size
        return tuple(shape)

    def slices_for_rank(self, rank: int) -> tuple[slice, ...]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside TP={self.world_size}")
        slices = [slice(None)] * len(self.global_shape)
        if self.shard_dim is not None:
            size = self.local_shape[self.shard_dim]
            slices[self.shard_dim] = slice(rank * size, (rank + 1) * size)
        return tuple(slices)

    def load_slice(
        self,
        source: Any,
        rank: int,
        *,
        expected_local_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Read only this rank's slice and verify its exact output shape."""
        source_shape = (
            tuple(source.get_shape())
            if hasattr(source, "get_shape")
            else tuple(source.shape)
        )
        if source_shape != self.global_shape:
            raise ValueError(
                f"Source shape {source_shape} does not match {self.global_shape}"
            )
        result = source[self.slices_for_rank(rank)]
        expected = tuple(expected_local_shape or self.local_shape)
        if tuple(result.shape) != expected:
            raise ValueError(
                f"Local shard shape {tuple(result.shape)} does not match {expected}"
            )
        return result.contiguous()


_REPLICATED_PROJECTIONS = ("q_a_proj.weight", "kv_a_proj_with_mqa.weight")
_COLUMN_PROJECTIONS = ("q_b_proj.weight", "kv_b_proj.weight")


def tp_shard_spec_for_key(
    key: str,
    shape: Sequence[int],
    *,
    world_size: int = PINNED_TP_SIZE,
) -> TPShardSpec:
    """Return the raw TP64 rule for a supported representative weight."""
    if key.endswith("weight_scale_inv"):
        raise ValueError("Use fp8_scale_coverage_for_key for inverse-scale metadata")
    shard_dim: int | None
    if key.endswith(_REPLICATED_PROJECTIONS):
        shard_dim = None
    elif key.endswith(_COLUMN_PROJECTIONS):
        shard_dim = 0
    elif key.endswith("self_attn.o_proj.weight"):
        shard_dim = 1
    elif re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$", key):
        # The pinned topology is pure EP: EP=64 spans the TP=64 world, so the
        # native EP-TP subgroup has degree one. Each owned routed expert keeps
        # its full intermediate dimension on its owner rank.
        shard_dim = None
    elif re.search(r"mlp\.(?:shared_experts\.)?(gate|up)_proj\.weight$", key):
        shard_dim = 0
    elif re.search(r"mlp\.(?:shared_experts\.)?down_proj\.weight$", key):
        shard_dim = 1
    elif key.endswith("mlp.gate.weight"):
        shard_dim = None
    else:
        raise ValueError(f"No tensor-parallel shard rule for {key}")
    return TPShardSpec(tuple(int(size) for size in shape), shard_dim, world_size)


def fp8_scale_coverage_for_key(
    weight_key: str,
    weight_shape: Sequence[int],
    *,
    rank: int,
    world_size: int = PINNED_TP_SIZE,
    quantization: Fp8BlockQuantization = PINNED_FP8,
) -> ScaleCoverage:
    """Return the scale blocks needed by a supported rank-local weight."""
    spec = tp_shard_spec_for_key(weight_key, weight_shape, world_size=world_size)
    return quantization.scale_coverage_for_weight_shard(
        weight_shape,
        shard_dim=spec.shard_dim,
        rank=rank,
        world_size=world_size,
    )


@dataclass(frozen=True)
class ExpertParallelContract:
    num_routed_experts: int = 256
    world_size: int = PINNED_EP_SIZE
    tensor_parallel_world_size: int = PINNED_TP_SIZE

    def __post_init__(self) -> None:
        if self.num_routed_experts % self.world_size:
            raise ValueError(
                f"{self.num_routed_experts} experts are not divisible by EP={self.world_size}"
            )
        if self.tensor_parallel_world_size % self.world_size:
            raise ValueError(
                f"TP={self.tensor_parallel_world_size} is not divisible by "
                f"EP={self.world_size}"
            )

    @property
    def experts_per_rank(self) -> int:
        return self.num_routed_experts // self.world_size

    @property
    def routed_expert_tp_size(self) -> int:
        """Native EP-TP subgroup size for each owned routed expert."""
        return self.tensor_parallel_world_size // self.world_size

    def routed_experts_for_rank(self, rank: int) -> tuple[int, ...]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside EP={self.world_size}")
        start = rank * self.experts_per_rank
        return tuple(range(start, start + self.experts_per_rank))

    def owner_of_routed_expert(self, expert_index: int) -> int:
        if not 0 <= expert_index < self.num_routed_experts:
            raise ValueError(f"Invalid routed expert {expert_index}")
        return expert_index // self.experts_per_rank

    def rank_loads(self, info: KeyInfo, rank: int) -> bool:
        """Routed experts are owned; the shared expert is separate and replicated."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside EP={self.world_size}")
        if info.category is TensorCategory.ROUTED_EXPERT:
            assert info.expert_index is not None
            return self.owner_of_routed_expert(info.expert_index) == rank
        return info.category is not TensorCategory.MTP


PINNED_EP = ExpertParallelContract()


def local_load_plan(
    manifest: CheckpointManifest,
    *,
    ep_rank: int,
    ep: ExpertParallelContract = PINNED_EP,
) -> tuple[ManifestEntry, ...]:
    """Select main-workload metadata sources for one EP rank.

    Shared-expert tensors remain distinct from routed experts and are present
    on every rank. MTP tensors never enter a main-workload plan.
    """
    return tuple(
        entry
        for entry in manifest.entries
        if entry.info.disposition is not Disposition.INTENTIONAL_SKIP
        and ep.rank_loads(entry.info, ep_rank)
    )
