# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) weight loading -- shard index and checkpoint key map
============================================================================

**Landed in two halves; this is the SKELETON half** (``inc-glm53f-011``): the
shard index, the ``{param_name: checkpoint_key}`` mapping builder, and the
coverage reconciliation over the two. The NUMERICS half (``inc-glm53f-012``)
adds the blockwise-FP8 scale loaders below the seam marked at the foot of this
module. Nothing here reads a tensor value; this half is host-side key routing.

Why the shard index is a per-shard KEY LIST, not a ``{key: shard}`` dict
-----------------------------------------------------------------------
The pin's checkpoint reader flattens every shard into one
``{tensor_name: file_path}`` dict, assigning inside the per-file loop::

    for key in self._open_safetensor_files[file_path].keys():
        self._tensor_name_to_file[key] = file_path

That appears three times (``vllm_neuron/utils/checkpoints.py:226-227``,
``:396-397``, ``:642-643``), and in all three a tensor name present in **two**
shards is silently overwritten -- last file in iteration order wins. Nothing
downstream can see it happened: ``CheckpointLoadResult``
(``checkpoints.py:26-37``) carries ``missing_keys`` and ``unexpected_keys`` and
has no duplicate channel. A 62-shard checkpoint is where that matters, so
:class:`Glm5NextShardIndex` keeps **one key list per shard** and never collapses
them -- which makes duplicate detection a property of this class rather than of
its caller or of any test fixture.

<-- MODEL-SPECIFIC: the key vocabulary is GLM-5.3-Flash's, tagged by provenance
in :data:`KEY_FAMILY_PROVENANCE` because the families are not equally settled.
:data:`GROUNDED` families follow a leaf-name convention this repo or the HF
DeepSeek-MLA/MoE family already fixes; :data:`PROVISIONAL` families are required
by ``config.json`` but their leaf names are **not yet confirmed against the
checkpoint's own ``model.safetensors.index.json``**. Two further families are
deliberately absent rather than guessed -- see :data:`ABSENT_KEY_FAMILIES`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .config import DSA_LAYER_TYPE, Glm5NextTextConfig

#: The HF shard-index filename. Read for its ``weight_map`` object; see
#: :meth:`Glm5NextShardIndex.from_weight_map` for why loading it is lossy.
SHARD_INDEX_FILENAME = "model.safetensors.index.json"

#: Blockwise-FP8 scale companion suffix. A quantised ``<name>.weight`` in this
#: checkpoint is accompanied by ``<name>.weight_scale_inv`` holding the per-block
#: reciprocal scales (``weight_block_size = [128, 128]``,
#: ``activation_scheme = "dynamic"``). The suffix is mapped here; the numerics
#: that consume it are ``inc-glm53f-012``'s half.
FP8_SCALE_SUFFIX = "weight_scale_inv"

#: Provenance tags for a key family -- see the module docstring.
GROUNDED = "GROUNDED"
PROVISIONAL = "PROVISIONAL"

#: Every key family :func:`build_weight_mappings` emits, with its provenance.
#: Kept as data rather than prose so a caller -- or a test -- can assert that no
#: family slipped in untagged, which is the only thing that keeps the
#: distinction honest as this file grows.
KEY_FAMILY_PROVENANCE: dict[str, str] = {
    "embeddings_and_head": GROUNDED,
    "layer_norms": GROUNDED,
    "mla_dsa_attention": GROUNDED,
    "dsa_indexer": PROVISIONAL,
    "kda_linear_attention": PROVISIONAL,
    "dense_mlp": GROUNDED,
    "moe_router": GROUNDED,
    "moe_routed_experts": GROUNDED,
    "moe_shared_experts": GROUNDED,
}

#: Families ``config.json`` requires that this module deliberately does **not**
#: map, and why. Declared rather than omitted: an absent family that nobody
#: wrote down is indistinguishable from one that was forgotten.
ABSENT_KEY_FAMILIES: dict[str, str] = {
    "multi_hyper_connections": (
        "mhc/hc_mult have no in-repo precedent and no settled checkpoint leaf "
        "names; mapping them here would be invention, not porting"
    ),
    "vision_tower": (
        "glm5_next_vision is a separate module surface, following the "
        "qwen3_vl split between the decoder and its vision encoder"
    ),
}


class Glm5NextWeightMapError(ValueError):
    """Base for shard-index and key-mapping faults raised by this module."""


class DuplicateShardKeyError(Glm5NextWeightMapError):
    """A checkpoint key is present in more than one shard.

    Raised by :meth:`Glm5NextShardIndex.require_no_duplicates`. This is the
    condition the pin's ``{tensor_name: file_path}`` dict cannot report (see the
    module docstring), which is why it gets its own error type.
    """


# --------------------------------------------------------------------------- #
# The shard index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Glm5NextShardIndex:
    """Checkpoint keys grouped by the shard file that physically holds them.

    ``shard_keys`` maps a shard filename to that shard's own key list, which is
    what enumerating a safetensors header per file actually yields. Keeping the
    lists separate is the whole point: a key in two shards survives here and is
    reported by :meth:`duplicated_keys`.

    Shard order is the mapping's insertion order and is preserved, so
    :meth:`per_shard_counts` reads back in the order the shards were declared.
    """

    shard_keys: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_shard_key_lists(
        cls, shard_keys: Mapping[str, Sequence[str]]
    ) -> Glm5NextShardIndex:
        """Build from ``{shard_filename: [key, ...]}`` -- the faithful direction.

        A key repeated *within* one shard is impossible in a real safetensors
        header (its keys are a set), so a repeat inside one list is a caller
        bug and raises immediately rather than being counted twice.
        """
        frozen: dict[str, tuple[str, ...]] = {}
        for shard, keys in shard_keys.items():
            keys = tuple(keys)
            counts = Counter(keys)
            repeated = sorted(key for key, n in counts.items() if n > 1)
            if repeated:
                raise Glm5NextWeightMapError(
                    f"shard {shard!r} lists the same key more than once: "
                    f"{repeated}; a safetensors header cannot do this"
                )
            frozen[shard] = keys
        return cls(shard_keys=frozen)

    @classmethod
    def from_weight_map(cls, weight_map: Mapping[str, str]) -> Glm5NextShardIndex:
        """Build from an index ``weight_map`` (``{key: shard_filename}``).

        **Lossy by construction, and named so.** ``weight_map`` is a JSON
        object, so a key held by two shards cannot be represented in it at all
        -- one entry has already won before this method is called. An index
        built this way therefore always reports zero duplicates, which says
        nothing about the shards themselves. Use
        :meth:`from_shard_key_lists` off the shard headers when the duplicate
        question is the one being asked.
        """
        shard_keys: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            shard_keys.setdefault(shard, []).append(key)
        return cls.from_shard_key_lists(shard_keys)

    @classmethod
    def from_index_json(cls, text: str) -> Glm5NextShardIndex:
        """Parse an index file's text and build from its ``weight_map``.

        Carries :meth:`from_weight_map`'s lossiness; see that docstring.
        """
        document = json.loads(text)
        try:
            weight_map = document["weight_map"]
        except (TypeError, KeyError) as exc:
            raise Glm5NextWeightMapError(
                f"{SHARD_INDEX_FILENAME} has no 'weight_map' object"
            ) from exc
        return cls.from_weight_map(weight_map)

    # -- counts ------------------------------------------------------------- #

    def per_shard_counts(self) -> dict[str, int]:
        """``{shard_filename: how many keys that shard holds}``."""
        return {shard: len(keys) for shard, keys in self.shard_keys.items()}

    @property
    def num_shards(self) -> int:
        return len(self.shard_keys)

    @property
    def total_shard_key_count(self) -> int:
        """Sum of the per-shard counts -- **duplicates counted once each**.

        Equals :attr:`unique_key_count` exactly when no key is duplicated
        across shards, so the difference between the two is itself the
        duplicate measure.
        """
        return sum(len(keys) for keys in self.shard_keys.values())

    @property
    def unique_keys(self) -> tuple[str, ...]:
        """Every distinct key, in first-seen shard order."""
        seen: dict[str, None] = {}
        for keys in self.shard_keys.values():
            for key in keys:
                seen.setdefault(key, None)
        return tuple(seen)

    @property
    def unique_key_count(self) -> int:
        return len(self.unique_keys)

    # -- duplicates --------------------------------------------------------- #

    def duplicated_keys(self) -> dict[str, tuple[str, ...]]:
        """``{key: the shards holding it}``, for keys held by two or more.

        Empty when the index is clean. This is the report the pin's flattened
        ``{tensor_name: file_path}`` dict structurally cannot produce.
        """
        holders: dict[str, list[str]] = {}
        for shard, keys in self.shard_keys.items():
            for key in keys:
                holders.setdefault(key, []).append(shard)
        return {
            key: tuple(shards)
            for key, shards in holders.items()
            if len(shards) > 1
        }

    def require_no_duplicates(self) -> None:
        """Raise :class:`DuplicateShardKeyError` if any key is in two shards."""
        duplicated = self.duplicated_keys()
        if duplicated:
            detail = "; ".join(
                f"{key!r} in {list(shards)}"
                for key, shards in sorted(duplicated.items())
            )
            raise DuplicateShardKeyError(
                f"{len(duplicated)} checkpoint key(s) held by more than one "
                f"shard: {detail}"
            )


# --------------------------------------------------------------------------- #
# The checkpoint key map
# --------------------------------------------------------------------------- #


def _quantised(prefix: str, leaf: str, *, quantised: bool) -> list[str]:
    """Checkpoint key(s) for one projection: the weight, plus its FP8 scale.

    A blockwise-FP8 projection contributes **two** checkpoint keys, and both
    have to be referenced or the scale key shows up as unmatched. Norms, biases
    and embeddings are not quantised in this checkpoint and contribute one.
    """
    weight = f"{prefix}.{leaf}.weight"
    if not quantised:
        return [weight]
    return [weight, f"{prefix}.{leaf}.{FP8_SCALE_SUFFIX}"]


def _add(
    mappings: dict[str, str | list[str]],
    param: str,
    keys: Sequence[str],
) -> None:
    """Record one parameter's checkpoint key(s), scalar or list.

    Mirrors the pin's mapping shape exactly: a single key is stored as a bare
    string and several as a list (``llama3/model.py:1854-1865``), because
    ``load_sharded`` normalises with ``mappings.get(name, name)`` and only then
    wraps a scalar (``checkpoints.py:242-247``).
    """
    if param in mappings:
        raise Glm5NextWeightMapError(
            f"parameter {param!r} already has a mapping; refusing to overwrite"
        )
    mappings[param] = list(keys) if len(keys) > 1 else keys[0]


def build_weight_mappings(
    text_config: Glm5NextTextConfig,
    *,
    quantised: bool = True,
) -> dict[str, str | list[str]]:
    """Build ``{param_name: checkpoint_key | [checkpoint_key, ...]}``.

    Follows the standalone-builder convention the fork already uses
    (``llama3/eagle3_model.py:666-668``,
    ``qwen3_vl/vision_encoder_bf16.py:885-887``) and the parameter-naming
    convention of the fork's MoE precedent -- ``{prefix}.mlp.experts.<x>_weight``
    (``gpt_oss/model_mxfp4.py:2336-2358``).

    The attention family per layer is chosen off ``text_config.layer_types``
    by **equality**, never substring: ``"attention"`` is a substring of both
    family names (``config.py:33-37``).

    Args:
        text_config: drives every count -- layer schedule, ``first_k_dense_replace``,
            ``n_routed_experts``, ``n_shared_experts``, ``tie_word_embeddings``.
        quantised: when True (the checkpoint's own case) every projection also
            references its ``weight_scale_inv`` companion.

    Returns:
        The mapping. Parameter names are this half's declaration and settle
        when ``model_fp8.py`` lands (``inc-glm53f-013``); the checkpoint-key
        side is what this increment's coverage measures.
    """
    mappings: dict[str, str | list[str]] = {}
    layer_types = list(text_config.layer_types or ())

    # -- outside the layer stack (GROUNDED) --------------------------------- #
    _add(mappings, "model.embed_tokens_weight", ["model.embed_tokens.weight"])
    _add(mappings, "model.norm_weight", ["model.norm.weight"])
    if not text_config.tie_word_embeddings:
        _add(mappings, "lm_head_weight", ["lm_head.weight"])

    for layer_id, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{layer_id}"

        # -- per-layer norms (GROUNDED) ------------------------------------- #
        _add(
            mappings,
            f"{prefix}.input_layernorm_weight",
            [f"{prefix}.input_layernorm.weight"],
        )
        _add(
            mappings,
            f"{prefix}.post_attention_layernorm_weight",
            [f"{prefix}.post_attention_layernorm.weight"],
        )

        if layer_type == DSA_LAYER_TYPE:
            _add_dsa_attention(mappings, prefix, quantised=quantised)
        else:
            _add_kda_attention(mappings, prefix, quantised=quantised)

        if layer_id < text_config.first_k_dense_replace:
            _add_dense_mlp(mappings, prefix, quantised=quantised)
        else:
            _add_moe_mlp(mappings, prefix, text_config, quantised=quantised)

    return mappings


def _add_dsa_attention(
    mappings: dict[str, str | list[str]],
    prefix: str,
    *,
    quantised: bool,
) -> None:
    """MLA on the ``deepseek_sparse_attention`` half, plus the DSA indexer.

    <-- MODEL-SPECIFIC: ``mla_use_nope`` with ``qk_rope_head_dim == 0``
    (``config.py:119-127``) means there is **no** rotary head slice, so no
    ``*_rope_*`` projection is mapped. A reused DeepSeek-MLA mapping that
    assumes a RoPE split would ask for keys this checkpoint does not have.

    The lora projections are GROUNDED (the HF DeepSeek-MLA leaf names). The
    indexer block is PROVISIONAL -- ``index_n_heads``/``index_head_dim`` are in
    ``config.json`` but the indexer's leaf names are unconfirmed.
    """
    attn = f"{prefix}.self_attn"
    for leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        _add(
            mappings,
            f"{attn}.{leaf}_weight",
            _quantised(attn, leaf, quantised=quantised),
        )
    for leaf in ("q_a_layernorm", "kv_a_layernorm"):
        _add(mappings, f"{attn}.{leaf}_weight", _quantised(attn, leaf, quantised=False))

    # <-- PROVISIONAL: DSA indexer leaf names unconfirmed against the index.
    indexer = f"{attn}.indexer"
    for leaf in ("wq", "wk"):
        _add(
            mappings,
            f"{attn}.indexer.{leaf}_weight",
            _quantised(indexer, leaf, quantised=quantised),
        )
    for leaf in ("k_norm", "weights_proj"):
        _add(
            mappings,
            f"{attn}.indexer.{leaf}_weight",
            _quantised(indexer, leaf, quantised=False),
        )


def _add_kda_attention(
    mappings: dict[str, str | list[str]],
    prefix: str,
    *,
    quantised: bool,
) -> None:
    """The ``linear_attention`` (KDA, gated-delta) half.

    <-- PROVISIONAL: every leaf here is required by ``linear_attn_config``
    (``num_heads``, ``head_dim``, ``short_conv_kernel_size``,
    ``gate_lower_bound`` -- ``config.py:110-117``) and follows the gated-delta
    convention of the nearest relative named at intake (``qwen3_next``), but
    none is confirmed against this checkpoint's index.
    """
    attn = f"{prefix}.linear_attn"
    for leaf in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
        _add(
            mappings,
            f"{attn}.{leaf}_weight",
            _quantised(attn, leaf, quantised=quantised),
        )
    for leaf in ("conv1d", "norm"):
        _add(mappings, f"{attn}.{leaf}_weight", _quantised(attn, leaf, quantised=False))
    # Unprojected per-head state: no ``.weight`` leaf, so not via _quantised.
    _add(mappings, f"{attn}.conv1d_bias", [f"{attn}.conv1d.bias"])
    _add(mappings, f"{attn}.dt_bias", [f"{attn}.dt_bias"])
    _add(mappings, f"{attn}.A_log", [f"{attn}.A_log"])


def _add_dense_mlp(
    mappings: dict[str, str | list[str]],
    prefix: str,
    *,
    quantised: bool,
) -> None:
    """The dense MLP on the first ``first_k_dense_replace`` layers (GROUNDED).

    Gate and up stay **separate** parameters, matching the fork's own dense
    precedent (``llama3/model.py`` maps ``mlp.gate_proj_weight`` and
    ``mlp.up_proj_weight`` one-to-one) rather than fusing them here.
    """
    mlp = f"{prefix}.mlp"
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        _add(
            mappings,
            f"{mlp}.{leaf}_weight",
            _quantised(mlp, leaf, quantised=quantised),
        )


def _add_moe_mlp(
    mappings: dict[str, str | list[str]],
    prefix: str,
    text_config: Glm5NextTextConfig,
    *,
    quantised: bool,
) -> None:
    """Routed + shared experts on the sparse layers (GROUNDED).

    <-- MODEL-SPECIFIC: this checkpoint stores **one tensor per expert**
    (``model.layers.N.mlp.experts.E.gate_proj.weight``), the HF DeepSeek/GLM MoE
    convention, whereas the fork's only extant MoE precedent (``gpt_oss``) reads
    a single pre-stacked tensor for all experts. So the fork's per-projection
    expert parameter maps to a **list** of ``n_routed_experts`` checkpoint keys
    -- the list-valued branch of the mapping shape -- rather than to one key.

    ``topk_method = "noaux_tc"`` (``config.py:134``) is why the router carries
    ``e_score_correction_bias`` alongside its weight.
    """
    mlp = f"{prefix}.mlp"

    # Router. Not quantised: it runs in float32 (``moe_router_dtype``).
    _add(mappings, f"{mlp}.experts.router_weight", [f"{mlp}.gate.weight"])
    _add(
        mappings,
        f"{mlp}.experts.router_bias",
        [f"{mlp}.gate.e_score_correction_bias"],
    )

    for leaf in ("gate_proj", "up_proj", "down_proj"):
        expert_keys: list[str] = []
        for expert_id in range(text_config.n_routed_experts):
            expert_keys.extend(
                _quantised(
                    f"{mlp}.experts.{expert_id}", leaf, quantised=quantised
                )
            )
        _add(mappings, f"{mlp}.experts.{leaf}_weight", expert_keys)

    if text_config.n_shared_experts:
        shared = f"{mlp}.shared_experts"
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            _add(
                mappings,
                f"{mlp}.shared_experts.{leaf}_weight",
                _quantised(shared, leaf, quantised=quantised),
            )


# --------------------------------------------------------------------------- #
# Coverage: reconcile a shard index against a mapping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Glm5NextKeyCoverage:
    """How completely a mapping and a shard index account for one another.

    Both directions of non-correspondence are carried, because either one is a
    mapping defect: a parameter asking for a key the shards lack, and a shard
    key no parameter asks for. This mirrors ``CheckpointLoadResult``'s
    ``missing_keys`` / ``unexpected_keys`` split (``checkpoints.py:26-37``) and
    adds the duplicate channel that class has no room for.
    """

    per_shard_counts: dict[str, int]
    matched_parameters: dict[str, tuple[str, ...]]
    unmatched_parameters: dict[str, tuple[str, ...]]
    unmatched_checkpoint_keys: tuple[str, ...]
    duplicated_keys: dict[str, tuple[str, ...]]
    unique_checkpoint_key_count: int

    @property
    def total_shard_key_count(self) -> int:
        """Sum of the per-shard counts."""
        return sum(self.per_shard_counts.values())

    @property
    def mapped_key_count(self) -> int:
        """Distinct shard keys some parameter's mapping references."""
        return self.unique_checkpoint_key_count - len(self.unmatched_checkpoint_keys)

    @property
    def coverage_fraction(self) -> float:
        """Mapped share of the distinct shard keys. 1.0 is "100% mapped"."""
        if not self.unique_checkpoint_key_count:
            return 0.0
        return self.mapped_key_count / self.unique_checkpoint_key_count

    @property
    def unmatched_count(self) -> int:
        """Both directions in one number, as "0 unmatched" states it.

        Deliberately sums a count of *parameters* and a count of *keys*, the
        same join the fork's own fake-load result makes
        (``LoadFromSlicesResult.unmatched_keys``, which concatenates
        ``missing_keys`` -- parameter names -- with ``unexpected_keys``). Both
        are failures of the mapping under test, and at the value that matters
        here, zero, the two units cannot disagree.
        """
        return len(self.unmatched_parameters) + len(self.unmatched_checkpoint_keys)

    @property
    def duplicated_count(self) -> int:
        return len(self.duplicated_keys)

    @property
    def is_complete(self) -> bool:
        """Every key mapped, nothing unmatched either way, nothing duplicated."""
        return (
            self.coverage_fraction == 1.0
            and self.unmatched_count == 0
            and self.duplicated_count == 0
        )


def _referenced_keys(mappings: Mapping[str, str | list[str]]) -> dict[str, tuple[str, ...]]:
    """Normalise every mapping value to a tuple, as ``load_sharded`` does."""
    normalised: dict[str, tuple[str, ...]] = {}
    for param, keys in mappings.items():
        normalised[param] = tuple(keys) if isinstance(keys, list) else (keys,)
    return normalised


def check_key_coverage(
    index: Glm5NextShardIndex,
    mappings: Mapping[str, str | list[str]],
    *,
    strict: bool = False,
) -> Glm5NextKeyCoverage:
    """Reconcile ``index`` against ``mappings`` and report the coverage.

    Args:
        index: the shard index, ideally built from per-shard key lists so the
            duplicate channel is meaningful (see :class:`Glm5NextShardIndex`).
        mappings: as :func:`build_weight_mappings` returns.
        strict: also raise :class:`DuplicateShardKeyError` on a cross-shard
            duplicate. Off by default so a caller can *inspect* a dirty index;
            the load path should pass True.

    Returns:
        :class:`Glm5NextKeyCoverage`.
    """
    if strict:
        index.require_no_duplicates()

    available = set(index.unique_keys)
    referenced = _referenced_keys(mappings)

    matched: dict[str, tuple[str, ...]] = {}
    unmatched_parameters: dict[str, tuple[str, ...]] = {}
    for param, keys in referenced.items():
        missing = tuple(key for key in keys if key not in available)
        if missing:
            unmatched_parameters[param] = missing
        else:
            matched[param] = keys

    all_referenced = {key for keys in referenced.values() for key in keys}
    unmatched_checkpoint_keys = tuple(
        key for key in index.unique_keys if key not in all_referenced
    )

    return Glm5NextKeyCoverage(
        per_shard_counts=index.per_shard_counts(),
        matched_parameters=matched,
        unmatched_parameters=unmatched_parameters,
        unmatched_checkpoint_keys=unmatched_checkpoint_keys,
        duplicated_keys=index.duplicated_keys(),
        unique_checkpoint_key_count=index.unique_key_count,
    )


def scale_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """The blockwise-FP8 scale companions among ``keys``, in order.

    The seam onto ``inc-glm53f-012``: this half decides which keys are scales,
    that half decides what to do with the numbers inside them.
    """
    return tuple(key for key in keys if key.endswith(f".{FP8_SCALE_SUFFIX}"))


# --------------------------------------------------------------------------- #
# SEAM -- inc-glm53f-012 (numerics half) lands below this line:
# the blockwise-FP8 scale loaders and the 240-max downscale-and-compensate.
# Nothing above this line reads a tensor value.
# --------------------------------------------------------------------------- #
