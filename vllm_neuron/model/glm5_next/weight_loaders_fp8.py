# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) weight loading
=======================================

Two layers, in this order:

* Host-side key routing -- the shard index, the ``{param_name: checkpoint_key}``
  mapping builder, and the coverage reconciliation over the two. Nothing here
  reads a tensor value.
* Blockwise-FP8 numerics -- the scale loaders, the 240-max downscale, the
  expert-bank stackers -- below the seam marked further down this module.

Why the shard index is a per-shard key list, not a ``{key: shard}`` dict
-----------------------------------------------------------------------
The platform's checkpoint reader flattens every shard into one
``{tensor_name: file_path}`` dict, assigning inside the per-file loop::

    for key in self._open_safetensor_files[file_path].keys():
        self._tensor_name_to_file[key] = file_path

In ``vllm_neuron/utils/checkpoints.py`` that appears three times, and in all
three a tensor name present in two shards is silently overwritten -- last file
in iteration order wins. Nothing downstream can see it happened, because
``CheckpointLoadResult`` carries ``missing_keys`` and ``unexpected_keys`` and has
no duplicate channel. With 62 shards that matters, so
:class:`Glm5NextShardIndex` keeps one key list per shard and never collapses
them, which makes duplicate detection a property of the class rather than of its
caller.

<-- MODEL-SPECIFIC: the key vocabulary is GLM-5.3-Flash's, tagged in
:data:`KEY_FAMILY_PROVENANCE` by how each family's leaf names were established.
Families ``config.json`` requires but this module deliberately does not map are
named in :data:`ABSENT_KEY_FAMILIES` rather than left out silently.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .config import DSA_LAYER_TYPE, Glm5NextTextConfig
from .quantization import keeps_bf16

#: The HF shard-index filename. Read for its ``weight_map`` object; see
#: :meth:`Glm5NextShardIndex.from_weight_map` for why loading it is lossy.
SHARD_INDEX_FILENAME = "model.safetensors.index.json"

#: Blockwise-FP8 scale companion suffix. A quantised ``<name>.weight`` in this
#: checkpoint comes with ``<name>.weight_scale_inv`` holding the per-block
#: reciprocal scales (``weight_block_size = [128, 128]``,
#: ``activation_scheme = "dynamic"``).
FP8_SCALE_SUFFIX = "weight_scale_inv"

#: The checkpoint's own prefix for the whole text model. Two namespaces, not one:
#: every text-model tensor in this checkpoint is named
#: ``model.language_model.<...>`` while the module tree is named ``model.<...>``,
#: so a mapping needs both strings and they are not interchangeable.
#: ``lm_head.weight`` is outside both and spelled the same on each side.
CKPT_TEXT_PREFIX = "model.language_model"

#: The module-tree prefix -- what ``model_fp8.py`` calls the same tensors.
PARAM_TEXT_PREFIX = "model"

#: How a key family's leaf names were established: read off the checkpoint's own
#: index (``GROUNDED``) or taken from a naming convention (``PROVISIONAL``).
GROUNDED = "GROUNDED"
PROVISIONAL = "PROVISIONAL"

#: Every key family :func:`build_weight_mappings` emits, with its provenance.
#: Data rather than prose so a caller can check that no family slipped in
#: untagged.
KEY_FAMILY_PROVENANCE: dict[str, str] = {
    "embeddings_and_head": GROUNDED,
    "layer_norms": GROUNDED,
    "mla_dsa_attention": GROUNDED,
    "dsa_indexer": GROUNDED,
    "kda_linear_attention": GROUNDED,
    "multi_hyper_connections": GROUNDED,
    "dense_mlp": GROUNDED,
    "moe_router": GROUNDED,
    "moe_routed_experts": GROUNDED,
    "moe_shared_experts": GROUNDED,
}

#: Families ``config.json`` requires that this module deliberately does not map,
#: and why. Recorded rather than omitted: an absent family nobody wrote down is
#: indistinguishable from one that was forgotten.
ABSENT_KEY_FAMILIES: dict[str, str] = {
    "vision_tower": (
        "glm5_next_vision is a separate module surface, following the "
        "qwen3_vl split between the decoder and its vision encoder"
    ),
}


class Glm5NextWeightMapError(ValueError):
    """Base for shard-index and key-mapping faults raised by this module."""


class DuplicateShardKeyError(Glm5NextWeightMapError):
    """A checkpoint key is present in more than one shard.

    Raised by :meth:`Glm5NextShardIndex.require_no_duplicates`. It has its own
    type because a flat ``{tensor_name: file_path}`` dict cannot report the
    condition at all -- see the module docstring.
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
    :meth:`per_shard_counts` reads back in the order the shards were given.
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

        Lossy by construction: ``weight_map`` is a JSON object, so a key held by
        two shards cannot be represented in it -- one entry has already won
        before this method is called. An index built this way always reports
        zero duplicates, which says nothing about the shards. Use
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
        """Sum of the per-shard counts, counting a duplicated key once per shard.

        Equals :attr:`unique_key_count` exactly when no key is duplicated across
        shards, so the difference between the two measures the duplication.
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

        Empty when the index is clean. A flattened ``{tensor_name: file_path}``
        dict cannot produce this report at all.
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


def _quantised(
    prefix: str,
    leaf: str,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> list[str]:
    """Checkpoint key(s) for one projection: the weight, plus its FP8 scale.

    A blockwise-FP8 projection contributes two checkpoint keys, and both have to
    be referenced or the scale key shows up as unmatched. Norms, biases and
    embeddings are not quantised in this checkpoint and contribute one.

    ``skip`` is the checkpoint's own ``modules_to_not_convert``. A tensor it
    names gets no scale companion whatever ``quantised`` says, because the
    checkpoint keeps that tensor in BF16: no scale key exists to ask for, and
    asking leaves the parameter unmatched. The predicate is :func:`keeps_bf16`;
    this function adds no rule and holds no per-family table. An empty ``skip``
    suppresses nothing.
    """
    weight = f"{prefix}.{leaf}.weight"
    if not quantised or keeps_bf16(f"{prefix}.{leaf}", skip):
        return [weight]
    return [weight, f"{prefix}.{leaf}.{FP8_SCALE_SUFFIX}"]


def _add(
    mappings: dict[str, str | list[str]],
    param: str,
    keys: Sequence[str],
) -> None:
    """Record one parameter's checkpoint key(s), scalar or list.

    Mirrors the platform's mapping shape: a single key is stored as a bare string
    and several as a list, because ``load_sharded`` normalises with
    ``mappings.get(name, name)`` and only then wraps a scalar.
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
    modules_to_not_convert: Sequence[str] = (),
) -> dict[str, str | list[str]]:
    """Build ``{param_name: checkpoint_key | [checkpoint_key, ...]}``.

    Follows the standalone-builder convention the fork already uses in
    ``llama3/eagle3_model.py`` and ``qwen3_vl/vision_encoder_bf16.py``, and the
    MoE parameter-naming convention of ``gpt_oss/model_mxfp4.py``
    (``{prefix}.mlp.experts.<x>_weight``).

    Each layer's attention family is chosen off ``text_config.layer_types`` by
    equality, never substring: ``"attention"`` is a substring of both family
    names.

    Two prefixes, not one. Each layer builds a ``ckpt_prefix`` in the
    checkpoint's namespace and a ``param_prefix`` in the module tree's, and both
    are threaded into every family adder; the two namespaces are not
    interchangeable.

    The checkpoint's skip list decides which projections carry a scale. Pass
    ``modules_to_not_convert`` and no tensor the checkpoint keeps in BF16 has a
    ``weight_scale_inv`` companion asked for. The families that carry no scale at
    all are also structural in their own adders, so the two agree and each checks
    the other.

    Args:
        text_config: drives every count -- layer schedule, ``first_k_dense_replace``,
            ``n_routed_experts``, ``n_shared_experts``, ``tie_word_embeddings``.
        quantised: when True (the checkpoint's own case) a projection also
            references its ``weight_scale_inv`` companion, unless the skip list
            keeps it in BF16.
        modules_to_not_convert: the checkpoint's own BF16 skip list, as
            ``Glm5NextConfig`` lifts it. Empty suppresses nothing.

    Returns:
        The mapping, parameter name to checkpoint key or list of keys.
    """
    mappings: dict[str, str | list[str]] = {}
    layer_types = list(text_config.layer_types or ())
    skip = tuple(modules_to_not_convert or ())

    # -- outside the layer stack (GROUNDED) --------------------------------- #
    # The two namespaces part company here: parameter names are the module tree's,
    # checkpoint keys carry the text-model prefix. ``lm_head.weight`` is
    # unprefixed on both sides.
    _add(
        mappings,
        "model.embed_tokens_weight",
        [f"{CKPT_TEXT_PREFIX}.embed_tokens.weight"],
    )
    _add(mappings, "model.norm_weight", [f"{CKPT_TEXT_PREFIX}.norm.weight"])
    if not text_config.tie_word_embeddings:
        _add(mappings, "lm_head_weight", ["lm_head.weight"])

    for layer_id, layer_type in enumerate(layer_types):
        ckpt_prefix = f"{CKPT_TEXT_PREFIX}.layers.{layer_id}"
        param_prefix = f"{PARAM_TEXT_PREFIX}.layers.{layer_id}"

        # -- per-layer norms (GROUNDED) ------------------------------------- #
        _add(
            mappings,
            f"{param_prefix}.input_layernorm_weight",
            [f"{ckpt_prefix}.input_layernorm.weight"],
        )
        _add(
            mappings,
            f"{param_prefix}.post_attention_layernorm_weight",
            [f"{ckpt_prefix}.post_attention_layernorm.weight"],
        )

        _add_mhc(mappings, ckpt_prefix, param_prefix)

        if layer_type == DSA_LAYER_TYPE:
            _add_dsa_attention(
                mappings, ckpt_prefix, param_prefix, quantised=quantised, skip=skip
            )
        else:
            _add_kda_attention(
                mappings, ckpt_prefix, param_prefix, quantised=quantised, skip=skip
            )

        if layer_id < text_config.first_k_dense_replace:
            _add_dense_mlp(
                mappings, ckpt_prefix, param_prefix, quantised=quantised, skip=skip
            )
        else:
            _add_moe_mlp(
                mappings,
                ckpt_prefix,
                param_prefix,
                text_config,
                quantised=quantised,
                skip=skip,
            )

    return mappings


#: The six multi-hyper-connection leaves each layer carries, in index order.
#: Bare tensors: none has a ``.weight`` leaf and none has a scale companion.
MHC_LEAVES: tuple[str, ...] = (
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

#: The plain leaves the checkpoint holds in float32, besides the two in
#: :data:`KDA_BARE_LEAVES`: the four mHC mix leaves and the router correction
#: bias. Each arrives as a plain key, so a placeholder typed by kind alone would
#: give it the config dtype and narrow the checkpoint's float32 before any
#: consumer reads it. The reference is float32 on all five -- its fused mHC entry
#: point asserts the mix leaves and its router asserts an fp32 gate.
#:
#: The two ``fn`` leaves are absent on purpose: the checkpoint holds them in the
#: config dtype, so there is no cast to remove.
FLOAT32_PLAIN_LEAVES: tuple[str, ...] = (
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_scale",
    "router_bias",
)

#: The DSA half's four scaled projections -- the only ``self_attn`` leaves on a
#: sparse-attention layer that carry a ``weight_scale_inv`` companion.
#: ``kv_b_proj`` and every indexer leaf carry none, so asking for one there leaves
#: the parameter unmatched.
DSA_SCALED_PROJECTIONS: tuple[str, ...] = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "o_proj",
)

#: The KDA half's 15 leaves, as ``self_attn.*`` on each linear-attention layer.
#: None is quantised: no ``weight_scale_inv`` exists under any KDA ``self_attn``.
#: Split by whether the leaf has a ``.weight``.
KDA_PROJECTIONS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "q_conv1d",
    "k_conv1d",
    "v_conv1d",
    "o_norm",
    "o_proj",
)
#: The two unprojected per-head state tensors: no ``.weight`` leaf at all.
KDA_BARE_LEAVES: tuple[str, ...] = ("A_log", "dt_bias")


def _add_mhc(
    mappings: dict[str, str | list[str]],
    ckpt_prefix: str,
    param_prefix: str,
) -> None:
    """The multi-hyper-connection leaves (GROUNDED against the real index).

    Six bare tensors per layer, hanging off the layer rather than off the
    attention or the MLP -- they are the layer's own residual-mixing state. Every
    decoder layer carries all six; the MTP layer carries none, which this function
    never has to know because that layer is not in ``layer_types``.
    """
    for leaf in MHC_LEAVES:
        _add(mappings, f"{param_prefix}.{leaf}", [f"{ckpt_prefix}.{leaf}"])


def _add_dsa_attention(
    mappings: dict[str, str | list[str]],
    ckpt_prefix: str,
    param_prefix: str,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> None:
    """MLA on the ``deepseek_sparse_attention`` half, plus the DSA indexer.

    <-- MODEL-SPECIFIC: ``mla_use_nope`` with ``qk_rope_head_dim == 0`` means
    there is no rotary head slice, so no ``*_rope_*`` projection is mapped. A
    reused DeepSeek-MLA mapping that assumes a RoPE split would ask for keys this
    checkpoint does not have.

    18 checkpoint keys per layer: 14 tensors, 4 of which carry a scale. The
    indexer's query projection is spelled ``wq_b``, and neither ``kv_b_proj`` nor
    any indexer projection has a scale companion, so only the four leaves in
    :data:`DSA_SCALED_PROJECTIONS` ask for one.
    """
    ckpt_attn = f"{ckpt_prefix}.self_attn"
    param_attn = f"{param_prefix}.self_attn"

    # A scaled projection's weight parameter maps to both checkpoint keys, the
    # shape the dense and shared MLP projections use, so the loader chooser reads a
    # quantised weight and applies the trn2 downscale to the bytes. The scale key
    # also maps to a parameter of its own, which is what the attention dequantises
    # from; the compensation on that grid pairs with the downscale on the bytes.
    for leaf in DSA_SCALED_PROJECTIONS:
        keys = _quantised(ckpt_attn, leaf, quantised=quantised, skip=skip)
        _add(mappings, f"{param_attn}.{leaf}_weight", keys)
        # Zero or one scale key: ``_quantised`` returns the companion only when
        # this leaf carries one, so a leaf the checkpoint keeps in BF16 maps no
        # scale parameter.
        for scale_key in keys[1:]:
            _add(
                mappings,
                f"{param_attn}.{leaf}_{FP8_SCALE_SUFFIX}",
                [scale_key],
            )
    # kv_b_proj is a real projection with no scale in this checkpoint, so it sits
    # here rather than in the loop above.
    for leaf in ("kv_b_proj", "q_a_layernorm", "kv_a_layernorm"):
        _add(
            mappings,
            f"{param_attn}.{leaf}_weight",
            _quantised(ckpt_attn, leaf, quantised=False),
        )

    ckpt_indexer = f"{ckpt_attn}.indexer"
    param_indexer = f"{param_attn}.indexer"
    for leaf in ("wq_b", "wk", "k_norm", "weights_proj"):
        _add(
            mappings,
            f"{param_indexer}.{leaf}_weight",
            _quantised(ckpt_indexer, leaf, quantised=False),
        )
    # k_norm carries a bias as well as a weight; the compress pair are bare
    # tensors with no ``.weight`` leaf.
    _add(
        mappings,
        f"{param_indexer}.k_norm_bias",
        [f"{ckpt_indexer}.k_norm.bias"],
    )
    for leaf in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
        _add(mappings, f"{param_indexer}.{leaf}", [f"{ckpt_indexer}.{leaf}"])


def _add_kda_attention(
    mappings: dict[str, str | list[str]],
    ckpt_prefix: str,
    param_prefix: str,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> None:
    """The ``linear_attention`` (KDA, gated-delta) half.

    15 checkpoint keys per layer and not one scale companion. The leaf names are
    not the ``qwen3_next`` gated-delta convention: instead of
    ``linear_attn.{in_proj_qkvz, in_proj_ba, out_proj, conv1d, norm}`` the
    checkpoint carries 15 distinct ``self_attn.*`` leaves -- eight projections,
    three per-projection convolutions, an output norm, an output projection and
    two bare state tensors. There is no ``conv1d.bias``.

    ``quantised`` and ``skip`` are accepted and unused: this family asks for no
    scale companion at any setting, so the skip list has nothing to suppress. The
    arguments stay in the signature so the four adders share one call shape. The
    checkpoint's own skip list names all 15 leaves, which says the same thing
    independently.
    """
    del quantised  # this family is unquantised in the checkpoint, at any setting
    del skip  # nothing to suppress: no leaf here asks for a scale companion

    ckpt_attn = f"{ckpt_prefix}.self_attn"
    param_attn = f"{param_prefix}.self_attn"

    for leaf in KDA_PROJECTIONS:
        _add(
            mappings,
            f"{param_attn}.{leaf}_weight",
            _quantised(ckpt_attn, leaf, quantised=False),
        )
    for leaf in KDA_BARE_LEAVES:
        _add(mappings, f"{param_attn}.{leaf}", [f"{ckpt_attn}.{leaf}"])


def _add_dense_mlp(
    mappings: dict[str, str | list[str]],
    ckpt_prefix: str,
    param_prefix: str,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> None:
    """The dense MLP on the first ``first_k_dense_replace`` layers (GROUNDED).

    Gate and up stay separate parameters, matching the fork's dense precedent in
    ``llama3/model.py``, rather than being fused here.
    """
    ckpt_mlp = f"{ckpt_prefix}.mlp"
    param_mlp = f"{param_prefix}.mlp"
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        _add(
            mappings,
            f"{param_mlp}.{leaf}_weight",
            _quantised(ckpt_mlp, leaf, quantised=quantised, skip=skip),
        )


def _add_moe_mlp(
    mappings: dict[str, str | list[str]],
    ckpt_prefix: str,
    param_prefix: str,
    text_config: Glm5NextTextConfig,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> None:
    """Routed + shared experts on the sparse layers (GROUNDED).

    <-- MODEL-SPECIFIC: this checkpoint stores one tensor per expert
    (``model.layers.N.mlp.experts.E.gate_proj.weight``), the HF DeepSeek/GLM MoE
    convention, where the fork's only other MoE precedent (``gpt_oss``) reads a
    single pre-stacked tensor for all experts. So each per-projection expert
    parameter maps to a list of ``n_routed_experts`` checkpoint keys rather than to
    one key.

    ``topk_method = "noaux_tc"`` is why the router carries
    ``e_score_correction_bias`` alongside its weight.
    """
    ckpt_mlp = f"{ckpt_prefix}.mlp"
    param_mlp = f"{param_prefix}.mlp"

    # Router. Not quantised: it runs in float32 (``moe_router_dtype``).
    _add(mappings, f"{param_mlp}.experts.router_weight", [f"{ckpt_mlp}.gate.weight"])
    _add(
        mappings,
        f"{param_mlp}.experts.router_bias",
        [f"{ckpt_mlp}.gate.e_score_correction_bias"],
    )

    for leaf in ("gate_proj", "up_proj", "down_proj"):
        expert_keys: list[str] = []
        for expert_id in range(text_config.n_routed_experts):
            expert_keys.extend(
                _quantised(
                    f"{ckpt_mlp}.experts.{expert_id}",
                    leaf,
                    quantised=quantised,
                    skip=skip,
                )
            )
        _add(mappings, f"{param_mlp}.experts.{leaf}_weight", expert_keys)

    if text_config.n_shared_experts:
        ckpt_shared = f"{ckpt_mlp}.shared_experts"
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            _add(
                mappings,
                f"{param_mlp}.shared_experts.{leaf}_weight",
                _quantised(ckpt_shared, leaf, quantised=quantised, skip=skip),
            )


# --------------------------------------------------------------------------- #
# Coverage: reconcile a shard index against a mapping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Glm5NextKeyCoverage:
    """How completely a mapping and a shard index account for one another.

    Both directions of non-correspondence are carried, because either one is a
    mapping defect: a parameter asking for a key the shards lack, and a shard key
    no parameter asks for. Mirrors ``CheckpointLoadResult``'s ``missing_keys`` /
    ``unexpected_keys`` split, and adds the duplicate channel that class has no
    room for.
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
        """Both directions of non-correspondence in one number.

        Sums a count of parameters and a count of keys -- the same join
        ``LoadFromSlicesResult.unmatched_keys`` makes. Both are mapping failures,
        and at the value that matters, zero, the two units cannot disagree.
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

    The seam onto the numerics section below: key routing decides which keys are
    scales, the numerics decide what to do with the numbers inside them.
    """
    return tuple(key for key in keys if key.endswith(f".{FP8_SCALE_SUFFIX}"))


# --------------------------------------------------------------------------- #
# Blockwise-FP8 numerics: the scale loaders, the 240-max downscale-and-compensate,
# and the expert-bank stackers. Nothing above this line reads a tensor value.
#
# The imports below sit in this section rather than in the module header so the
# host-side key routing above stays importable on a host with no vendor runtime;
# see :func:`resolved_fp8_clamp_max`.
# --------------------------------------------------------------------------- #

import logging

import torch

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    sharding_weight_loader,
    tensor_width_sharding_loader,
)

from .quantization import DEFAULT_WEIGHT_BLOCK_SIZE

logger = logging.getLogger(__name__)

#: How many floored tile coordinates the warning names before it stops listing and
#: gives the count alone. A pathological grid can floor thousands of tiles, and a
#: warning that dumps every one of them is its own defect.
_FLOORED_BLOCKS_NAMED_IN_WARNING = 8

# --------------------------------------------------------------------------- #
# Blockwise-FP8 range constants
# --------------------------------------------------------------------------- #
#
# Trn2's kernels read fp8 SBUF as legacy ``nl.float8_e4m3`` (max finite 240) while
# this checkpoint's bytes are OCP ``float8_e4m3fn`` (max finite 448). The two
# encodings share a bit layout for every finite magnitude at or below 240, so
# squeezing the bytes into the 240 range and compensating the per-block dequant
# scale by the inverse factor preserves the dequantised tensor without
# reinterpreting a byte pattern. Same downscale-plus-compensation as
# ``model/llama3/weight_loaders_static_fp8.py``, at per-block rather than
# per-parameter granularity.
#
# The factor is an exact power of two, not the range ratio 240/448, and both
# halves of that matter. A power of two shifts an fp8 exponent and leaves the
# mantissa alone, so the squeeze loses nothing: 118 of the 126 positive magnitudes
# survive the round trip bit-exactly at 1/2, against 14 at 240/448 = 15/28, which
# is not a binary fraction and re-rounds. And 1/2 is the largest power of two that
# fits, since 448 * 1/2 = 224 is inside 240 and the next power up is not; the 16
# counts between 224 and 240 stay unused.
#
# The cost is bounded and named: the smallest subnormal does not survive, because
# 2**-9 halves to exactly half a step and round-to-nearest-even sends it to zero.
# Eight of the 126 magnitudes are inexact and all eight are below 2**-5.
#
# llama3's static path keeps the 240/448 ratio and is not entered from here. The
# exactness argument applies there in kind, but the cost differs: this path holds
# one scale per ``[128, 128]`` block, floors it at :data:`MINVAL` and names every
# floored tile through :func:`report_floored_blocks`, so a block pushed towards
# zero is visible per tile in the load log. The static path holds one scalar per
# parameter with no per-block census, so the same loss is not observable at load
# time.

#: The dtype the squeezed bytes are stored in. OCP ``float8_e4m3fn``, as the
#: llama3 static path stores them: the representable magnitudes at or below 240 are
#: shared with legacy ``e4m3``, so the trn2 kernel's reinterpretation of these
#: bytes is value-preserving.
_FP8_DTYPE = torch.float8_e4m3fn

#: Legacy ``nl.float8_e4m3`` max finite magnitude (trn2).
_FP8_E4M3_MAX = 240.0

#: OCP ``float8_e4m3fn`` max finite magnitude (the checkpoint's scale space).
_FP8_E4M3FN_MAX = 448.0

#: Applied to the weight bytes: the largest power of two that fits
#: :data:`_FP8_E4M3FN_MAX` inside :data:`_FP8_E4M3_MAX` (``448 * 0.5 = 224 <= 240``,
#: while ``448 * 1.0 = 448 > 240``). Written as the literal rather than derived from
#: the two maxima, because a division would hide the power-of-two property and
#: re-round the mantissa of 112 of the 126 magnitudes. See the section header.
_FP8_WEIGHT_DOWNSCALE = 0.5

#: Applied to the per-block dequant scales -- the exact inverse, so the product
#: ``byte * scale`` is preserved up to the bytes' own re-quantisation. Exact in
#: fp32 because both factors are powers of two (``0.5 * 2.0 == 1.0``, no rounding),
#: which is what makes the round trip bit-exact for those 118 magnitudes rather
#: than merely close.
_FP8_SCALE_COMPENSATION = 2.0

#: Floor for a stored per-block dequant scale.
#:
#: What it protects: ``activation_scheme`` is ``"dynamic"``, so the consuming path
#: derives activation scales at runtime from these weight scales. A block whose
#: scale has collapsed towards zero -- an all-zero weight tile, or a quantiser that
#: emitted a denormal -- turns that reciprocal into ``inf`` or ``NaN`` and poisons
#: the whole matmul. Flooring the stored scale keeps the reciprocal finite.
#:
#: The floor is reported per block rather than applied silently: the census is on
#: :class:`BlockScaleCompensation.floored_blocks`, and on the load path
#: :func:`report_floored_blocks` warns when it engaged, naming the parameter and the
#: tiles.
MINVAL = 1e-5


# --------------------------------------------------------------------------- #
# The platform gate
# --------------------------------------------------------------------------- #


def resolved_fp8_clamp_max() -> float:
    """The FP8 clamp the vendor resolved for this process, read at call time.

    Reads :data:`vllm_neuron.utils.dtype_utils.FP8_CLAMP_MAX` through the module
    rather than binding it with a ``from`` import, so this module never takes an
    import-time snapshot of a value the vendor resolved once at its own import
    time.

    The import is lazy on purpose: ``dtype_utils`` imports
    ``libtorch_neuronx_lite`` at module scope, and the key-routing half of this
    file must stay importable on a host where that vendor package is unavailable.
    """
    from vllm_neuron.utils import dtype_utils

    return dtype_utils.FP8_CLAMP_MAX


def needs_240_downscale() -> bool:
    """True when the resolved platform clamp is 240.0, the case the squeeze is for.

    The downscale is conditional, never unconditional: it exists because trn2's
    fp8 maximum is 240.0 while the checkpoint's OCP scale space runs to 448.0. On a
    448.0-max platform an unconditional rescale would corrupt correct weights. The
    condition reads the vendor's own resolved clamp rather than querying the
    platform a second time.

    The clamp is the right quantity rather than ``get_platform_target()``, which
    the llama3 static path uses: that helper has no CPU-mode fallback and raises
    ``RuntimeError`` on a bare CPU host with no NRT, while
    ``dtype_utils._resolve_fp8_clamp_max()`` handles that case and returns exactly
    the quantity the squeeze is about.
    """
    return resolved_fp8_clamp_max() == _FP8_E4M3_MAX


# --------------------------------------------------------------------------- #
# The block grid
# --------------------------------------------------------------------------- #


def block_grid_shape(
    weight_shape: tuple[int, ...],
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> tuple[int, int]:
    """The ``(rows, cols)`` shape of the scale grid a weight of this shape needs.

    Ceiling division on both axes: a real projection dimension is not always a
    multiple of the block edge, and the checkpoint ships a partial tile's scale
    rather than dropping it.
    """
    if len(weight_shape) != 2:
        raise Glm5NextWeightMapError(
            f"blockwise FP8 expects a 2-D weight, got shape {tuple(weight_shape)}"
        )
    rows, cols = weight_shape
    block_rows, block_cols = block_size
    return (
        (rows + block_rows - 1) // block_rows,
        (cols + block_cols - 1) // block_cols,
    )


def _require_grid(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[int, int]:
    """Check the scale grid against the weight, and return the expected shape."""
    expected = block_grid_shape(tuple(weight.shape), block_size)
    if tuple(scale_inv.shape) != expected:
        raise Glm5NextWeightMapError(
            f"scale grid shape {tuple(scale_inv.shape)} does not match a "
            f"{tuple(weight.shape)} weight at block size {block_size}: "
            f"expected {expected}"
        )
    return expected


def expand_block_scales(
    scale_inv: torch.Tensor,
    weight_shape: tuple[int, ...],
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> torch.Tensor:
    """Broadcast a ``[grid_rows, grid_cols]`` scale grid to the weight's shape.

    Repeat-interleave then slice, so a partial edge tile is truncated to the
    weight's real extent instead of padding the weight up to a whole tile.
    """
    rows, cols = weight_shape
    block_rows, block_cols = block_size
    expanded = scale_inv.repeat_interleave(block_rows, dim=0).repeat_interleave(
        block_cols, dim=1
    )
    return expanded[:rows, :cols]


def dequantise_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantise blockwise-FP8 bytes: ``byte * scale_inv[block]``, in fp32.

    ``weight_scale_inv`` is the checkpoint's **dequant multiplier** per tile (the
    reciprocal of the quantiser's scale -- hence the key's ``_inv`` suffix; see
    :data:`FP8_SCALE_SUFFIX`), so dequantisation is a multiply, never a divide.
    """
    _require_grid(weight, scale_inv, block_size)
    dense = weight.to(torch.float32)
    return dense * expand_block_scales(
        scale_inv.to(torch.float32), tuple(weight.shape), block_size
    )


# --------------------------------------------------------------------------- #
# The squeeze: bytes down, scales up
# --------------------------------------------------------------------------- #


#: Value censuses skipped because the tensor they would count carried no values.
#: A shape-only load appends one name per skip; a load with values never appends.
#: A record, not a control -- nothing in this module reads it.
SKIPPED_VALUE_CENSUSES: list[str] = []


@dataclass(frozen=True)
class BlockScaleCompensation:
    """A compensated per-block scale grid, with the census of what it changed.

    The counts are the function's own report rather than something a caller
    recomputes, so a count cannot disagree with the transform that produced it.
    """

    #: The grid to store: compensated and floored, fp32.
    scale_inv: torch.Tensor
    #: False when the resolved clamp is not 240.0 -- the grid is then the input.
    applied: bool
    #: Scales strictly below :data:`MINVAL` in the input grid.
    below_minval_before: int
    #: Scales strictly below :data:`MINVAL` in the stored grid.
    below_minval_after: int
    #: ``(row, col)`` of every block whose scale the floor raised.
    floored_blocks: tuple[tuple[int, int], ...]
    #: False when the grid carried no values and the census was skipped. The three
    #: counted fields are then zero because they are unanswerable, not because the
    #: grid was clean, and a caller that reports them must say which it has.
    values_read: bool = True


def compensate_block_scales(scale_inv: torch.Tensor) -> BlockScaleCompensation:
    """Multiply a block scale grid by :data:`_FP8_SCALE_COMPENSATION` and floor it at
    :data:`MINVAL`.

    Conditional on :func:`needs_240_downscale`, and the floor travels with the
    compensation rather than standing alone: the floor exists to keep the grid this
    transform produces safe to take a reciprocal of, so where the transform is a
    no-op the floor is too and the checkpoint's own scales are stored as shipped.
    The census is measured either way.
    """
    grid = scale_inv.to(torch.float32)
    if grid.device.type == "meta":
        # A shape-only grid carries no numbers, and the census below reads three of
        # them. The transform is arithmetic and runs on shapes, so the grid returned
        # here is the grid a real load would store; what cannot be answered is how
        # many of its scales fall below the floor. Those counts are zero, no block is
        # named, and the skip is recorded on the result.
        applied = needs_240_downscale()
        stored = (grid * _FP8_SCALE_COMPENSATION).clamp(min=MINVAL) if applied else grid
        SKIPPED_VALUE_CENSUSES.append("compensate_block_scales")
        return BlockScaleCompensation(
            scale_inv=stored,
            applied=applied,
            below_minval_before=0,
            below_minval_after=0,
            floored_blocks=(),
            values_read=False,
        )
    below_before = int((grid < MINVAL).sum().item())

    if not needs_240_downscale():
        return BlockScaleCompensation(
            scale_inv=grid,
            applied=False,
            below_minval_before=below_before,
            below_minval_after=below_before,
            floored_blocks=(),
        )

    compensated = grid * _FP8_SCALE_COMPENSATION
    needs_floor = compensated < MINVAL
    floored = compensated.clamp(min=MINVAL)
    floored_blocks = tuple(
        (int(row), int(col)) for row, col in needs_floor.nonzero().tolist()
    )
    return BlockScaleCompensation(
        scale_inv=floored,
        applied=True,
        below_minval_before=below_before,
        below_minval_after=int((floored < MINVAL).sum().item()),
        floored_blocks=floored_blocks,
    )


def report_floored_blocks(
    compensation: BlockScaleCompensation, param_name: str | None = None
) -> bool:
    """Warn when the ``MINVAL`` floor engaged on a load. Returns whether it did.

    A floored tile is not cosmetic: a stored scale of ``1e-6`` raised to ``1e-5``
    makes every weight in that tile dequantise about 5.4x too large. The loader
    keeps the compensated grid but drops the census, so without this call nothing
    outside a test can see that a tile was floored.

    It warns rather than raises, because raising would make a checkpoint the floor
    exists to rescue fail to load instead.

    The parameter name is optional because ``SafetensorsWeightLoader``'s transform
    signature is ``(slices, rank)`` and carries no name. A caller that knows the
    name passes it; otherwise the message says so rather than printing ``None``.
    """
    if not compensation.floored_blocks:
        return False

    named = compensation.floored_blocks[:_FLOORED_BLOCKS_NAMED_IN_WARNING]
    tail = len(compensation.floored_blocks) - len(named)
    where = ", ".join(f"({row},{col})" for row, col in named)
    if tail > 0:
        where = f"{where} and {tail} more"
    logger.warning(
        "fp8 block-scale floor ENGAGED while loading %s: %d of the grid's tiles had "
        "a compensated scale below MINVAL=%g and were raised to it, at %s. Weights in "
        "those tiles dequantise larger than the checkpoint intended, by MINVAL "
        "divided by the compensated scale. Tiles below the floor before "
        "compensation: %d; after: %d.",
        param_name if param_name else "an unnamed parameter",
        len(compensation.floored_blocks),
        MINVAL,
        where,
        compensation.below_minval_before,
        compensation.below_minval_after,
    )
    return True


def downscale_fp8_weight_bytes(weight: torch.Tensor) -> torch.Tensor:
    """Squeeze fp8 weight bytes into the 240 range, or return them unchanged.

    Conditional on :func:`needs_240_downscale`. The ``clamp`` is defensive and has
    16 counts of slack: the largest OCP magnitude, 448, maps to 224, while the bound
    stays at 240 because 240 is what the trn2 kernel reads rather than what this
    squeeze produces. No in-range input can reach it. The clamp is kept because a
    checkpoint carrying a non-finite or out-of-spec byte would otherwise store one,
    and because the llama3 static path clamps at the same point.
    """
    if not needs_240_downscale():
        return weight
    return (
        (weight.to(torch.float32) * _FP8_WEIGHT_DOWNSCALE)
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(_FP8_DTYPE)
    )


@dataclass(frozen=True)
class BlockwiseFp8Squeeze:
    """The full downscale-and-compensate result over one weight and its grid."""

    #: Stored bytes, ``_FP8_DTYPE``.
    weight: torch.Tensor
    #: Stored per-block dequant scales, fp32.
    scale_inv: torch.Tensor
    block_size: tuple[int, int]
    #: False when the resolved clamp is not 240.0.
    applied: bool
    #: ``max(abs(stored bytes))``, as a Python float.
    max_abs_stored: float
    #: Share of stored bytes with ``abs(x) <= 240.0``. 1.0 is "100%".
    fraction_within_240: float
    below_minval_before: int
    below_minval_after: int
    floored_blocks: tuple[tuple[int, int], ...]


def squeeze_blockwise_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> BlockwiseFp8Squeeze:
    """Downscale the bytes and compensate the block scales, as one transform.

    The tensor-level entry point. The loader factories below wrap the same two
    halves for the checkpoint path, where the weight and its ``weight_scale_inv``
    companion arrive as separate keys and therefore separate loaders.
    """
    _require_grid(weight, scale_inv, block_size)

    squeezed = downscale_fp8_weight_bytes(weight)
    compensation = compensate_block_scales(scale_inv)

    stored = squeezed.to(torch.float32).abs()
    return BlockwiseFp8Squeeze(
        weight=squeezed,
        scale_inv=compensation.scale_inv,
        block_size=block_size,
        applied=compensation.applied,
        max_abs_stored=float(stored.max().item()),
        fraction_within_240=float(
            (stored <= _FP8_E4M3_MAX).to(torch.float32).mean().item()
        ),
        below_minval_before=compensation.below_minval_before,
        below_minval_after=compensation.below_minval_after,
        floored_blocks=compensation.floored_blocks,
    )


# --------------------------------------------------------------------------- #
# Per-block agreement, for the dequantisation claim
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlockAgreement:
    """How well one block's dequantisation survived the squeeze.

    The reference magnitude is the block's own absolute maximum, not the tensor's:
    a global abs-max normalisation would let the largest-scaled block set the
    tolerance for every other one and hide a small-scaled block's disagreement.
    """

    index: tuple[int, int]
    #: ``max(abs(after - before))`` over the block.
    max_abs_diff: float
    #: ``max(abs(before))`` over the block -- the normalising reference.
    max_abs_before: float
    #: ``max_abs_diff / max_abs_before``, the block-normalised difference.
    normalised_diff: float
    #: ``atol + rtol * max_abs_before``.
    tolerance: float
    #: ``max_abs_diff <= tolerance``.
    within: bool
    #: The single worst per-element relative difference in the block. Reported,
    #: never gated, so the choice of normalisation above is visible to a caller
    #: rather than implicit.
    worst_element_relative: float


def block_agreement(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
    rtol: float,
    atol: float,
) -> tuple[BlockAgreement, ...]:
    """Compare two dequantised tensors block by block.

    ``rtol`` and ``atol`` are required keyword arguments with no defaults, because
    the fork's tolerance map has no fp8 entry and silently falls back to the bf16
    pair. An fp8 comparison has to state both.
    """
    if before.shape != after.shape:
        raise Glm5NextWeightMapError(
            f"dequantisation shapes disagree: {tuple(before.shape)} vs "
            f"{tuple(after.shape)}"
        )
    rows, cols = before.shape
    block_rows, block_cols = block_size
    grid_rows, grid_cols = block_grid_shape((rows, cols), block_size)

    lhs = before.to(torch.float32)
    rhs = after.to(torch.float32)

    reports: list[BlockAgreement] = []
    for grid_row in range(grid_rows):
        row_slice = slice(grid_row * block_rows, min((grid_row + 1) * block_rows, rows))
        for grid_col in range(grid_cols):
            col_slice = slice(
                grid_col * block_cols, min((grid_col + 1) * block_cols, cols)
            )
            block_before = lhs[row_slice, col_slice]
            block_after = rhs[row_slice, col_slice]
            diff = (block_after - block_before).abs()

            max_abs_diff = float(diff.max().item())
            max_abs_before = float(block_before.abs().max().item())
            tolerance = atol + rtol * max_abs_before
            nonzero = block_before.abs() > 0
            worst_relative = (
                float((diff[nonzero] / block_before.abs()[nonzero]).max().item())
                if bool(nonzero.any())
                else 0.0
            )
            reports.append(
                BlockAgreement(
                    index=(grid_row, grid_col),
                    max_abs_diff=max_abs_diff,
                    max_abs_before=max_abs_before,
                    normalised_diff=(
                        max_abs_diff / max_abs_before if max_abs_before else 0.0
                    ),
                    tolerance=tolerance,
                    within=max_abs_diff <= tolerance,
                    worst_element_relative=worst_relative,
                )
            )
    return tuple(reports)


# --------------------------------------------------------------------------- #
# Checkpoint-path loaders (mirror the llama3 static-fp8 entry points)
# --------------------------------------------------------------------------- #


def wrap_with_blockwise_fp8_downscale(
    loader: SafetensorsWeightLoader,
) -> SafetensorsWeightLoader:
    """Wrap a weight loader so its result is squeezed into the 240 range.

    Same shape as ``weight_loaders_static_fp8.py``'s ``_wrap_with_fp8_downscale``,
    with one difference: on a platform that needs no squeeze the result is still the
    weight slice, so an entry pairing a weight with its scale companion loads the
    same either way.
    """
    base_transform = loader.transform or (lambda slices, rank: slices[0][:])
    if not needs_240_downscale():
        return SafetensorsWeightLoader(transform=base_transform)

    def transform(slices, rank):
        return downscale_fp8_weight_bytes(base_transform(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def blockwise_scale_loader(param_name: str | None = None) -> SafetensorsWeightLoader:
    """Load a ``weight_scale_inv`` grid, compensated and floored.

    The grid is loaded whole: one fp32 value per weight tile, four orders of
    magnitude smaller than the weight, so it needs no sharding of its own here. A
    sharded scale grid follows the weight's own shard geometry, which
    ``model_fp8.py`` supplies.

    An engaged floor is reported rather than discarded, through
    :func:`report_floored_blocks`, because a load that inflates a whole tile of
    weights should leave a trace. Pass ``param_name`` when the caller knows which
    parameter this grid belongs to so the warning can name it; the returned tensor
    is the same either way.
    """

    def transform(slices, rank):
        if len(slices) != 1:
            raise Glm5NextWeightMapError(
                f"blockwise_scale_loader expects 1 slice, got {len(slices)}"
            )
        compensation = compensate_block_scales(slices[0][:])
        report_floored_blocks(compensation, param_name)
        return compensation.scale_inv

    return SafetensorsWeightLoader(transform=transform)


# --------------------------------------------------------------------------- #
# The loader choice: given one map entry's checkpoint key or keys, which loader
# above serves it. The parameter materialisation that consumes the answer lives in
# ``model_fp8.py``.
# --------------------------------------------------------------------------- #

#: A map entry naming one scale grid on its own.
MAPPED_KEY_SCALE_GRID = "scale_grid"

#: A map entry naming a quantised weight together with its scale companion.
MAPPED_KEY_QUANTISED_WEIGHT = "quantised_weight"

#: A map entry naming a whole bank of quantised experts -- one weight and one scale
#: grid per expert, interleaved weight-then-scale in checkpoint order. Separate from
#: :data:`MAPPED_KEY_QUANTISED_WEIGHT` because the two need different loaders and
#: different placeholder dtypes; see :func:`classify_mapped_keys`.
MAPPED_KEY_STACKED_BANK = "stacked_bank"

#: A map entry naming ordinary unquantised tensors.
MAPPED_KEY_PLAIN = "plain"


class Glm5NextExpertBankNotLoadableError(Glm5NextWeightMapError):
    """One map entry names a whole expert bank, which no loader here can stack.

    Subclasses :class:`Glm5NextWeightMapError`, this module's form for a named map
    refusal, so a caller that already handles a map error handles this one too and it
    stays a ``ValueError`` like ``model_fp8.py``'s ``Glm5NextWeightLoadError``.
    """


def _as_key_list(checkpoint_keys: str | Sequence[str]) -> list[str]:
    """One map entry's checkpoint keys as a list, whether it named one or many.

    :func:`build_weight_mappings` stores a lone key as a bare string and a fused
    family as a list, so every consumer has to normalise before counting. Extracted
    so both consumers normalise the same way.
    """
    if isinstance(checkpoint_keys, str):
        return [checkpoint_keys]
    return list(checkpoint_keys)


def classify_mapped_keys(checkpoint_keys: str | Sequence[str]) -> str:
    """Which of the four kinds one map entry's checkpoint key(s) describe.

    One classifier with two consumers: the loader :func:`loader_for_mapped_keys`
    picks below, and the placeholder dtype ``model_fp8.py`` gives the parameter
    before the load. One function rather than two, because the pipelined loader
    reads its target dtype off the placeholder and warns whenever the tensor it
    built differs -- so a placeholder typed against a different case than its own
    loader would make every load emit that warning. Two classifiers of the same
    cases can drift apart; one cannot. Adding a kind here therefore means teaching
    ``model_fp8.py``'s ``_placeholder_dtype`` about it in the same change.

    "Is this key a scale?" is not re-decided here: :func:`scale_keys` answers it and
    is called rather than copied, so the suffix convention lives in one place.

    The four cases are exhaustive by construction on the scale-key count. A key list
    holds no scale key (plain), exactly one alongside something else (quantised
    weight), is a lone scale key (scale grid), or holds more than one -- which can
    only be a bank of quantised experts, and needs :func:`stacked_expert_bank_loader`
    rather than the single-weight loader.
    """
    keys = _as_key_list(checkpoint_keys)
    scales = scale_keys(keys)
    if len(keys) == 1 and len(scales) == 1:
        return MAPPED_KEY_SCALE_GRID
    if len(scales) > 1:
        return MAPPED_KEY_STACKED_BANK
    if scales:
        return MAPPED_KEY_QUANTISED_WEIGHT
    return MAPPED_KEY_PLAIN


# --------------------------------------------------------------------------- #
# The tensor-parallel shard geometry, carried in as an input.
#
# ``model_fp8.py`` owns the per-rank geometry. A loader here shards only when a
# geometry is handed to it; without one, every rank reads the whole tensor.
#
# The geometry is an input, not a second classifier: :func:`classify_mapped_keys`
# decides which kind a map entry is, and the geometry only decides whether that
# kind's loader shards. A geometry that classified anything would be a second
# opinion about the same entry.
#
# Every number arrives already divided. The per-rank arithmetic happens in
# ``model_fp8.py``, at the one site that knows the world size, so this file never
# resolves a world size and never holds a partition that could disagree with the
# module's -- the same rule :func:`_bank_expert_indices` states for the expert
# count.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShardGeometry:
    """One parameter family's tensor-parallel shard, fully resolved.

    Three plain integers, because the caller has already done every division:
    ``shard_dim`` is the dimension in the final parameter shape, ``shard_size`` is
    this rank's extent along it, and ``num_shards`` is the world size the extent was
    divided by.

    Resolved integers rather than a module reference, even though the expert bank
    already passes its owner here. The world size lives on the root while the widths
    live on ten different classes, so resolving them here would split the arithmetic
    across two files; passing values keeps :func:`shard_geometry_for_grid` a pure
    function of numbers.
    """

    shard_dim: int
    shard_size: int
    num_shards: int

    def __post_init__(self) -> None:
        if self.shard_dim not in (0, 1):
            raise ValueError(
                f"shard_dim must be 0 or 1, got {self.shard_dim}; this package's "
                f"parameter layout is [out_features, in_features] and a shard "
                f"outside those two dimensions has no declared meaning"
            )
        if self.shard_size < 1:
            raise ValueError(
                f"shard_size must be >= 1, got {self.shard_size}; a rank with no "
                f"rows of a sharded family would load an empty tensor"
            )
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")


@dataclass(frozen=True)
class DeferredShardGeometry:
    """One family's shard when its FULL WIDTH is only known at load time.

    Same dimension and rank count as :class:`ShardGeometry`, with ``shard_size``
    absent on purpose: the tensor that answers it has not arrived yet.

    The families that need this are the ones whose width is not on the module.
    ``Glm5NextSharedExperts`` carries ``num_shared_experts`` and ``swiglu_limit``
    and no width at all, so a resolved geometry for its three projections cannot be
    built where it is attached. The dense MLP's three take the same route so that
    one rule pads both, and so the width comes from the checkpoint tensor, which
    cannot disagree with the weights, rather than from a config product.

    ``pad_to_multiple_of`` is the consumer's block extent, not a convenience. When
    set, the full width is first rounded up to the smallest multiple of
    ``num_shards * pad_to_multiple_of``, so every rank's shard is a whole number of
    consumer blocks and the padded tail is zeros (weight) or ones (grid). That is
    what makes a 12288-wide dense intermediate loadable at 64 ranks, where
    ``12288 // 64 == 192`` is not a whole number of 128-row blocks.
    """

    shard_dim: int
    num_shards: int
    pad_to_multiple_of: int | None = None
    #: What the padded elements hold. 0.0 for a weight, 1.0 for a reciprocal
    #: block scale -- see :func:`~vllm_neuron.utils.weight_loader.pad_to_shape`.
    pad_value: float = 0.0
    #: The owning module's expert-parallel degree, carried for one reason: the EP-TP
    #: group only exists above 1. ``get_neuron_ep_tp_group`` asserts on
    #: ``ep_degree > 1``, so at degree 1 there is no group to read a column from and
    #: the group is the whole world. Without the degree here, the column reader
    #: mistakes a degree-1 load for a missing group and refuses it. Only the bank
    #: sets this above 1.
    expert_parallel_degree: int = 1
    #: The extent one rank's shard must already be a whole multiple of, for a
    #: geometry that pads nothing. The only no-pad check the load path makes on its
    #: own is even division, which a shard can pass while still being a fraction of a
    #: consumer block -- and then the failure surfaces much later, inside the kernel's
    #: ``scale_grid_shape``. This field moves that refusal onto the load.
    #:
    #: Set per family rather than inferred from "this family pads nothing", because
    #: that would be a proxy for "is consumed by the block-FP8 kernel" and would
    #: wrongly refuse a future unquantised deferred family. A family that leaves it
    #: ``None`` is unaffected.
    require_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.shard_dim not in (0, 1):
            raise ValueError(
                f"shard_dim must be 0 or 1, got {self.shard_dim}; this package's "
                f"parameter layout is [out_features, in_features] and a shard "
                f"outside those two dimensions has no declared meaning"
            )
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of < 1:
            raise ValueError(
                f"pad_to_multiple_of must be >= 1 or None, got "
                f"{self.pad_to_multiple_of}"
            )
        if self.expert_parallel_degree < 1:
            raise ValueError(
                f"expert_parallel_degree must be >= 1, got "
                f"{self.expert_parallel_degree}"
            )
        if self.require_multiple_of is not None and self.require_multiple_of < 1:
            raise ValueError(
                f"require_multiple_of must be >= 1 or None, got "
                f"{self.require_multiple_of}"
            )
        if self.require_multiple_of is not None and self.pad_to_multiple_of is not None:
            # Both together is a contradiction rather than belt and braces: padding
            # makes the extent a whole multiple, so a requirement beside it either
            # restates the pad or disagrees with it, and a reader cannot tell which
            # was meant.
            raise ValueError(
                f"a geometry declares both pad_to_multiple_of "
                f"{self.pad_to_multiple_of} and require_multiple_of "
                f"{self.require_multiple_of}; padding already makes every rank's "
                f"extent a whole multiple, so declare one or the other"
            )


#: Either kind of shard declaration. Named once so the four functions that accept
#: both say so identically.
AnyShardGeometry = ShardGeometry | DeferredShardGeometry


def dense_consumer_block_quant_size() -> int:
    """The block extent the DENSE block-FP8 kernel will accept, from that kernel.

    Imported rather than re-typed, which is the point of the function:
    ``blockwise_fp8_mm.scale_grid_shape`` refuses any weight extent that is not a
    whole number of ``SCALE_BLOCK_SIZE`` blocks, and a literal here would be a second
    place for the consumer's granularity to live.

    This is the number for the dense MLP, the shared expert and every MLA
    projection -- every family ``blockwise_fp8_mm`` dequantises. That kernel indexes
    its scales by the 128-row blocks the checkpoint itself stores.

    The import is function-local so a load-path module does not depend on a kernel
    module at import time.
    """
    from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE

    return int(SCALE_BLOCK_SIZE)


def routed_bank_consumer_block_quant_size() -> int:
    """The block extent the ROUTED EXPERT BANK will accept, from its own producer.

    One block constant per consumer, each read from the producer that enforces it.
    The bank does not go through ``blockwise_fp8_mm``: its scale operands are built
    by the MoE retile, which refuses any extent that is not a whole
    ``BLOCK_QUANT_SIZE`` block.

    The two consumers cannot share one function, because their extents differ -- the
    dense kernel indexes 128-row blocks and the retile 256-row ones. A bank shard
    checked against the dense number can be 384 rows, which the retile then refuses
    inside the prep: exactly the late failure the load-time check exists to prevent.
    Naming the consumer at every call site is what keeps the two apart.
    """
    from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE

    return int(BLOCK_QUANT_SIZE)


def consumer_block_quant_size() -> int:
    """Deprecated alias for :func:`routed_bank_consumer_block_quant_size`.

    Kept because existing callers were written when the package had one consumer
    granularity. It forwards rather than holding a second copy of the number. New
    code should call :func:`dense_consumer_block_quant_size` or
    :func:`routed_bank_consumer_block_quant_size`, which say which consumer they
    mean.
    """
    return routed_bank_consumer_block_quant_size()


def refuse_inadmissible_shard_extent(
    extent: int,
    geometry: DeferredShardGeometry,
    param_name: str | None = None,
    *,
    shard_dim: int,
) -> None:
    """Refuse a per-rank extent the geometry's own requirement does not admit.

    A deferred geometry that pads gets its whole-multiple property by construction;
    one that declares :attr:`DeferredShardGeometry.require_multiple_of` instead has
    to be checked, and the only place the number exists is here, after the tensor
    arrived and the extent was divided.

    One rule, two arrival sites. A deferred family reaches its extent either through
    :func:`_sharding_loader` (the non-bank families) or through
    :func:`_column_of_each_expert` (the routed bank, which takes
    ``stacked_expert_bank_loader`` and never passes through ``_sharding_loader``).
    Both call this function, so there is one predicate and one message rather than
    two that can drift apart.

    The refusal names the nearest admissible full widths, not just the failure: its
    reader chose a degree and a world size, so the useful answer is which widths
    would have worked.

    Args:
        extent: this rank's extent along the sharded dimension, as loaded.
        geometry: the deferred geometry whose requirement is being enforced.
        param_name: named first in any refusal, through :func:`_refuse`.
        shard_dim: the dimension ``extent`` was measured on, named in the message so
            a refusal cannot be read against the wrong axis.
    """
    required = geometry.require_multiple_of
    if required is None or extent % required == 0:
        return
    ranks = geometry.num_shards
    # The downward neighbour is named only when it exists. Flooring an extent already
    # narrower than one block gives 0, and offering a rank no rows at all is not
    # advice.
    below = (extent // required) * required
    above = below + required
    if below >= required:
        options = (
            f"The nearest per-rank extents that are admissible are {below} and "
            f"{above}, which at {ranks} ranks means a full width of {below * ranks} "
            f"or {above * ranks}."
        )
    else:
        options = (
            f"This shard is narrower than one whole block, so the smallest "
            f"admissible per-rank extent is {above}, which at {ranks} ranks means a "
            f"full width of {above * ranks}. Fewer ranks dividing this width would "
            f"also do it."
        )
    _refuse(
        param_name,
        f"loads {extent} rows per rank along dim {shard_dim}, which is not a whole "
        f"multiple of the {required}-row block its consumer requires "
        f"({extent} % {required} = {extent % required}). Its {ranks} ranks divide the "
        f"checkpoint width evenly, so nothing above refused it, but the block-FP8 "
        f"consumer reads whole {required}-row blocks and would refuse this shard "
        f"later, on the kernel rather than on the load. {options} This geometry asks "
        f"for a whole-multiple extent rather than a pad, so the shard is refused "
        f"instead of padded.",
    )


def shard_geometry_for_grid(
    geometry: AnyShardGeometry,
    param_name: str | None = None,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> AnyShardGeometry:
    """The scale grid's geometry for a weight sharded by ``geometry``.

    A grid holds one fp32 value per weight tile, so a weight sharded into
    ``shard_size`` rows takes ``shard_size // block_rows`` grid rows, on the same
    dimension, because the grid's axes correspond to the weight's.

    The boundary is checked, not rounded. If the weight's shard boundary does not
    fall on a block boundary, two ranks share one tile and there is no way to split
    that tile's single scale between them; rounding would hand one rank a tile scaled
    for another's rows, silently corrupting a whole 128-row band. So it is refused by
    name, with the parameter and the boundary in the message. The refusal happens
    when the scale grid is read, which is after the module tree is materialised --
    unlike the bank refusals further down this file, which happen at construction.

    Two boundaries are checked and they currently coincide: the checkpoint's tile is
    128 rows and the dense consumer's block
    (:func:`dense_consumer_block_quant_size`) is also 128, so a shard that clears the
    tile rule clears the block rule too. The second check is kept because it becomes
    load-bearing again the moment either granularity moves; each has its own message.
    ``block_size`` is the checkpoint's own ``weight_block_size``, threaded in by the
    caller -- the default is only the parser's fallback.

    A deferred geometry converts without either check, because there is nothing yet
    to check: the width arrives with the tensor, and the pad multiple then makes the
    shard a whole consumer block by construction. The pad multiple is converted from
    weight rows to grid rows -- one grid row per ``extent`` weight rows -- and the two
    ceilings agree exactly, since ``ceil(full / (N * pad)) == ceil((full / extent) /
    (N * pad / extent))`` when ``extent`` divides both ``full`` and ``pad``. The
    padded grid entries hold 1.0 rather than 0.0, because the grid is a reciprocal
    scale that multiplies its weight block: 1.0 leaves the padded zero rows exactly
    zero and 0.0 would not.
    """
    if isinstance(geometry, DeferredShardGeometry):
        extent = block_size[0] if geometry.shard_dim == 0 else block_size[1]
        pad = geometry.pad_to_multiple_of
        # ``expert_parallel_degree`` is carried through both returns below. The field
        # decides whether ``_expert_parallel_shard_column`` consults the EP-TP group or
        # short-circuits to ``rank % tp_per_ep``, and the group only exists above degree
        # 1. Letting it fall back to 1 here would tell the reader "no group" for every
        # bank grid while the weight path kept the true degree, so the two halves of one
        # bank would read their columns differently -- and on a non-contiguous mesh those
        # are different columns. This conversion changes the grid's row arithmetic and
        # must not change which rank a column comes from.
        #
        # ``require_multiple_of`` is deliberately not carried. It is the block-FP8
        # consumer's rule about a weight extent, enforced once on the weight by
        # :func:`refuse_inadmissible_shard_extent`. A grid is sharded exactly when its
        # weight is, on the same dimension, so a grid from an admissible weight is
        # admissible by construction -- and a grid row count is the weight's divided by
        # the tile extent, so carrying the number unconverted would enforce a bound the
        # rule never meant.
        if pad is None:
            return DeferredShardGeometry(
                shard_dim=geometry.shard_dim,
                num_shards=geometry.num_shards,
                pad_to_multiple_of=None,
                pad_value=1.0,
                expert_parallel_degree=geometry.expert_parallel_degree,
            )
        if pad % extent:
            _refuse(
                param_name,
                f"pads its weight to a multiple of {pad} rows along dim "
                f"{geometry.shard_dim}, which is not a multiple of the {extent}-row "
                f"block extent at block size {block_size}. Its scale grid holds one "
                f"value per {block_size[0]}x{block_size[1]} tile, so a pad that ends "
                f"inside a tile has no whole number of grid rows to pad to.",
            )
        return DeferredShardGeometry(
            shard_dim=geometry.shard_dim,
            num_shards=geometry.num_shards,
            pad_to_multiple_of=pad // extent,
            pad_value=1.0,
            expert_parallel_degree=geometry.expert_parallel_degree,
        )
    extent = block_size[0] if geometry.shard_dim == 0 else block_size[1]
    if geometry.shard_size % extent:
        _refuse(
            param_name,
            f"is sharded into {geometry.shard_size} rows along dim "
            f"{geometry.shard_dim}, which is not a multiple of the {extent}-row "
            f"block extent at block size {block_size}. Its scale grid holds one "
            f"value per {block_size[0]}x{block_size[1]} tile, so this boundary "
            f"puts two ranks inside one tile and there is no way to split that "
            f"tile's single scale between them. The nearest block-aligned shard "
            f"sizes are {(geometry.shard_size // extent) * extent} and "
            f"{((geometry.shard_size // extent) + 1) * extent}.",
        )
    # The dense consumer's block, named. Every family reaching this branch carries a
    # resolved width -- the MLA projections and KDA -- and all of them are dequantised
    # by ``blockwise_fp8_mm``. The routed bank is deferred and never arrives here, so
    # reading the bank's larger block here would enforce a bound no consumer of these
    # weights has.
    consumer_block = dense_consumer_block_quant_size()
    if geometry.shard_size % consumer_block:
        _refuse(
            param_name,
            f"is sharded into {geometry.shard_size} rows along dim "
            f"{geometry.shard_dim}, which clears the {extent}-row checkpoint tile "
            f"but is not a whole number of the CONSUMER's {consumer_block}-row "
            f"blocks. The block-FP8 kernel this weight is dequantised for indexes "
            f"its scales by whole {consumer_block}-row blocks "
            f"(blockwise_fp8_mm.SCALE_BLOCK_SIZE fixes that number and "
            f"scale_grid_shape refuses anything else; the routed expert bank has its "
            f"own, larger block and is not this rule's subject), so this shard would "
            f"load and then be refused at the first forward pass instead of here. "
            f"The nearest consumer-aligned shard sizes are "
            f"{(geometry.shard_size // consumer_block) * consumer_block} and "
            f"{((geometry.shard_size // consumer_block) + 1) * consumer_block}; at "
            f"{geometry.num_shards} ranks that is a full width of "
            f"{(geometry.shard_size // consumer_block) * consumer_block * geometry.num_shards}"
            f" or {((geometry.shard_size // consumer_block) + 1) * consumer_block * geometry.num_shards}"
            f", which is what a padded shard supplies.",
        )
    return ShardGeometry(
        shard_dim=geometry.shard_dim,
        shard_size=geometry.shard_size // extent,
        num_shards=geometry.num_shards,
    )


def _sharding_loader(
    geometry: AnyShardGeometry, param_name: str | None = None
) -> SafetensorsWeightLoader:
    """The platform's shard loader, built from either kind of geometry.

    ``is_storage_transposed`` is False because in this package the parameter layout
    is the checkpoint layout: the wrapped loader's base transform is ``slices[0][:]``
    with no transpose, and every consumer transposes at compute time instead. The
    qwen3 precedent passes True because its checkpoint stores those weights
    transposed; copying that flag here would slice the wrong dimension of every
    family.

    The dispatch is on which geometry arrived, not on a flag. A resolved geometry
    already carries its per-rank extent and takes ``sharding_weight_loader``; a
    deferred one takes ``tensor_width_sharding_loader``, which reads the extent off
    the checkpoint slice. Neither can serve the other's case, so this is the only
    place that has to know the difference.
    """
    if isinstance(geometry, DeferredShardGeometry):
        inner = tensor_width_sharding_loader(
            shard_dim=geometry.shard_dim,
            num_shards=geometry.num_shards,
            pad_to_multiple_of=geometry.pad_to_multiple_of,
            pad_value=geometry.pad_value,
            param_name=param_name,
        )
        if geometry.require_multiple_of is None:
            # A deferred family with no requirement is untouched: the inner loader is
            # returned as it is, rather than wrapped in a check that always passes.
            return inner
        # The requirement is checked on the loaded extent, the first moment it exists.
        # Wrapped rather than pushed into ``tensor_width_sharding_loader``, because that
        # function is a shared utility while the block extent is a block-FP8 fact this
        # file owns, so the rule belongs here beside the family declarations.
        def checked(slices: list, rank: int) -> torch.Tensor:
            result = inner.transform(slices, rank)
            refuse_inadmissible_shard_extent(
                int(result.shape[geometry.shard_dim]),
                geometry,
                param_name,
                shard_dim=geometry.shard_dim,
            )
            return result

        return SafetensorsWeightLoader(transform=checked)
    return sharding_weight_loader(
        shard_dim=geometry.shard_dim,
        shard_size=geometry.shard_size,
        num_shards=geometry.num_shards,
        is_storage_transposed=False,
    )


def _weight_slice_only(
    loader: SafetensorsWeightLoader, weight_index: int
) -> SafetensorsWeightLoader:
    """Give a one-slice transform the weight slice of a multi-key map entry.

    The platform's sharding transform asserts ``len(slices) == 1``, while a
    blockwise-FP8 projection is two checkpoint keys -- the weight and its scale
    companion, see :func:`_quantised` -- which the reader hands over as two slices.
    Composing the two without this adapter makes every sharded quantised family fail
    on a bare assertion that names no parameter.

    ``weight_index`` is read off the key list by the caller rather than assumed to be
    0, because the order is the map's and the map is built elsewhere.
    """
    inner = loader.transform
    if inner is None:  # pragma: no cover -- _sharding_loader always sets one
        raise Glm5NextWeightMapError(
            "_weight_slice_only wraps a loader that has a transform; the "
            "default loader needs no adapter because it already reads slice 0"
        )

    def transform(slices, rank):
        return inner([slices[weight_index]], rank)

    return SafetensorsWeightLoader(transform=transform)


#: The one parameter this package stores in the transpose of its checkpoint
#: orientation. Matched on the leaf, because the prefix carries a layer number.
TRANSPOSED_AT_LOAD_LEAF = "experts.router_weight"


def transposed_weight_loader() -> SafetensorsWeightLoader:
    """Load one plain weight key with its two dimensions swapped.

    The checkpoint stores a projection as ``[out_features, in_features]`` and this
    package keeps that orientation, because every consumer transposes at compute
    time. The router's consumer cannot: ``noaux_tc_rmsnorm_router_topk`` hands the
    tensor to an NKI kernel whose HBM operand layout is ``[H, E]`` and validates it
    before either route, so the orientation is that kernel's contract rather than a
    matmul convention it could absorb.

    Swapping once here rather than at the call site keeps the parameter in the
    orientation the kernel consumes and puts no transpose in the traced graph.
    ``SafetensorsWeightLoader.load`` makes the result contiguous, so the view does
    not reach the parameter.
    """

    def transform(slices, rank):
        return slices[0][:].t()

    return SafetensorsWeightLoader(transform=transform)


def _grid_spans_more_blocks_whole_than_per_rank(
    geometry: ShardGeometry,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> bool:
    """Whether this weight's grid actually has rows to divide between ranks.

    A grid holds one value per weight tile, so a weight can be sharded while its
    grid is not: when the whole weight fits inside one block along the shard
    dimension, both ranks' rows live in the same tile and the whole grid already
    describes each rank's shard exactly. ``_require_grid`` agrees, because
    ``ceil(shard / block) == ceil(full / block) == 1`` there.

    Ask this before :func:`shard_geometry_for_grid`, in that order. That function
    refuses a shard whose boundary splits a tile, which is right when the grid has
    tiles to divide and wrong here -- a 64-row weight at a 128-row block would be
    refused for a load that needs no sharding at all. A geometry that both splits
    tiles and spans several of them still reaches that refusal.
    """
    extent = block_size[0] if geometry.shard_dim == 0 else block_size[1]
    full = geometry.shard_size * geometry.num_shards
    return -(-full // extent) != -(-geometry.shard_size // extent)


def compensating_sharded_scale_grid_loader(
    geometry: ShardGeometry, param_name: str | None = None
) -> SafetensorsWeightLoader:
    """This rank's rows of a scale grid that is a registered parameter, compensated.

    Compensates where :func:`sharded_scale_grid_loader` does not, because the two
    serve different readers. The out-of-band reader stores a grid raw and leaves
    compensation to the prep that consumes it, so its loader must not compensate. A
    grid that is a registered parameter is consumed by
    ``_dequantised_projection_weight``, which expects the same compensated, floored
    value :func:`blockwise_scale_loader` produces for an unsharded one. Sharding must
    not change what a grid means, so the compensation and its floor report are the
    same, applied to this rank's rows.
    """
    if not _grid_spans_more_blocks_whole_than_per_rank(geometry):
        return blockwise_scale_loader(param_name)
    # ``param_name`` has to be passed explicitly, even though it is optional.
    # ``shard_geometry_for_grid`` can return a ``DeferredShardGeometry``, which is the
    # one branch of ``_sharding_loader`` that forwards the name, and
    # ``tensor_width_sharding_loader`` uses it to label every width refusal it raises.
    # Omitting it costs no number and every failure message.
    inner = _sharding_loader(
        shard_geometry_for_grid(geometry, param_name), param_name
    ).transform
    if inner is None:  # pragma: no cover -- _sharding_loader always sets one
        raise Glm5NextWeightMapError(
            "compensating_sharded_scale_grid_loader wraps a loader that has a "
            "transform; a grid with no shard takes the whole-grid loader above"
        )

    def transform(slices, rank):
        compensation = compensate_block_scales(inner(slices, rank))
        report_floored_blocks(compensation, param_name)
        return compensation.scale_inv

    return SafetensorsWeightLoader(transform=transform)


def sharded_scale_grid_loader(
    geometry: AnyShardGeometry,
    param_name: str | None = None,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> SafetensorsWeightLoader:
    """This rank's rows of a sharded weight's scale grid, and nothing else.

    Takes the weight's geometry and converts it to the grid's with
    :func:`shard_geometry_for_grid`, so the caller never divides by the block extent
    itself.

    It does not compensate, which is why it exists separately from
    :func:`compensating_sharded_scale_grid_loader`. Its only consumer is the
    out-of-band reader, which stores the grid as it arrives and leaves
    :func:`compensate_block_scales` to the load-time prep. A loader that compensated
    here would make a grid mean one thing at world size 1 and another at world size
    2, so this shards and stops.

    ``block_size`` is the checkpoint's own ``quantization_config.weight_block_size``,
    threaded in by the caller. The default is the parser's fallback, which is right
    for this checkpoint and would silently be wrong for another.
    """
    return _sharding_loader(
        shard_geometry_for_grid(geometry, param_name, block_size), param_name
    )


def loader_for_mapped_keys(
    checkpoint_keys: str | Sequence[str],
    *,
    param_name: str | None = None,
    owner: object | None = None,
    geometry: AnyShardGeometry | None = None,
) -> SafetensorsWeightLoader | None:
    """The loader that serves one mapped parameter, or ``None`` for the default.

    ``geometry`` is this parameter's tensor-parallel shard, already resolved, or
    ``None`` for a replicated family. It is an input to each kind's loader and never
    a kind of its own: it adds no branch to :func:`classify_mapped_keys` and no case
    below, it only decides whether the case's loader shards.

    One leaf is answered by name, before the kinds: :data:`TRANSPOSED_AT_LOAD_LEAF`,
    whose consumer's operand layout is the transpose of the checkpoint's. Answering
    it first makes the answer total, so a combination this loader cannot serve
    honestly is refused rather than silently loaded in the checkpoint's orientation.

    Then one case per :func:`classify_mapped_keys` kind:

    * A lone scale grid gets :func:`blockwise_scale_loader`, which compensates the
      grid for the trn2 range and reports a floored block by name -- passing
      ``param_name`` is what lets that report name the parameter. With a geometry it
      gets :func:`compensating_sharded_scale_grid_loader` instead, because a
      registered grid belonging to a sharded weight must be cut to the same rows.
    * A quantised weight with one scale companion gets
      :func:`wrap_with_blockwise_fp8_downscale` over the default loader, or, with a
      geometry, over the shard loader behind :func:`_weight_slice_only`. Without a
      geometry the wrapper's base transform is ``slices[0][:]``, which keeps the
      weight slice and drops the scale companion -- which is why the scales are read
      out of band rather than through this map. With a geometry the sharding
      transform asserts ``len(slices) == 1``, so the weight slice has to be selected
      by index first, and the index is read off the keys rather than assumed to be 0.
      The squeeze is elementwise, so squeezing this rank's rows and taking this
      rank's rows of the squeezed tensor give the same bytes.
    * An expert bank -- more than one scale key -- gets
      :func:`stacked_expert_bank_loader`. ``owner`` is the module that declares the
      parameter, which is where the expert geometry lives.
    * A plain entry naming one weight key gets the shard loader when a geometry is
      given: the unquantised sharded family, an ordinary case.
    * A plain entry naming more than one weight key is an unquantised bank and is
      refused by name. There is no scale to de-interleave and no shape to stack
      into, and the default loader's ``len(slices) == 1`` assertion would otherwise
      fail without naming the parameter and after it was registered. No such entry
      exists on the real index; a quantisation profile that keeps an expert leaf in
      bf16 would produce one.
    * Anything else gets ``None``, meaning attach nothing:
      :func:`~vllm_neuron.utils.weight_loader.get_weight_loader` already falls back
      to the identity loader when a parameter carries none. Every replicated family
      lands here.

    The bank loader is the only one that consults ``rank``, and it does so at load
    time rather than here. Everything else in this function is rank-blind, which is
    why ``owner`` arrives for the bank and the geometry is read off the module rather
    than recomputed.
    """
    keys = _as_key_list(checkpoint_keys)
    kind = classify_mapped_keys(keys)
    if (param_name or "").endswith(TRANSPOSED_AT_LOAD_LEAF):
        if kind != MAPPED_KEY_PLAIN or len(keys) != 1 or geometry is not None:
            raise Glm5NextWeightMapError(
                f"{param_name} is transposed at load, which is defined for one "
                f"replicated plain key, but it arrived as {kind} over "
                f"{len(keys)} keys with geometry={geometry!r}. A quantised or "
                f"sharded entry would need the transpose composed with the "
                f"other transform, and its dim would name the other axis, so "
                f"this is refused by name rather than loaded in the "
                f"checkpoint's orientation in silence."
            )
        return transposed_weight_loader()
    if kind == MAPPED_KEY_SCALE_GRID:
        # Two of the MLA scale grids -- ``q_b_proj``'s and ``o_proj``'s -- belong to
        # a weight sharded on the head-width dimension, so they arrive here with its
        # geometry. Without the sharded branch the weight would be this rank's half
        # while its grid still described the whole tensor, and ``_require_grid`` would
        # refuse the load at real widths. ``None`` means a replicated grid or world
        # size 1 and takes the whole-grid loader.
        if geometry is None:
            return blockwise_scale_loader(param_name)
        return compensating_sharded_scale_grid_loader(geometry, param_name)
    if kind == MAPPED_KEY_STACKED_BANK:
        # The bank's geometry is always a ``DeferredShardGeometry``, because
        # ``Glm5NextRoutedExperts`` holds counts and degrees but no width -- the width
        # comes off the checkpoint tensor. Its rank count is ``tp_per_ep``, not the
        # world size, since the experts are already divided across the
        # expert-parallel groups. At degree 1 ``tp_per_ep`` equals the world, so the
        # bank is still sharded and this branch divides its intermediate width across
        # every rank. ``geometry`` is ``None`` only when one rank holds each group
        # whole: world size equal to the degree, or world size 1.
        return stacked_expert_bank_loader(
            keys, param_name=param_name, owner=owner, geometry=geometry
        )
    if kind == MAPPED_KEY_QUANTISED_WEIGHT:
        if geometry is None:
            base = SafetensorsWeightLoader()
        else:
            scales = scale_keys(keys)
            weight_index = next(
                index for index, key in enumerate(keys) if key not in scales
            )
            base = _weight_slice_only(
                _sharding_loader(geometry, param_name), weight_index
            )
        return wrap_with_blockwise_fp8_downscale(base)
    if len(keys) == 1 and geometry is not None:
        return _sharding_loader(geometry, param_name)
    if len(keys) > 1:
        raise Glm5NextExpertBankNotLoadableError(
            f"{param_name or '<unnamed parameter>'} maps to {len(keys)} "
            f"checkpoint keys carrying 0 scale keys, so it is a bank of "
            f"{len(keys)} unquantised experts rather than one plain tensor. "
            f"This module stacks a bank of quantised experts (weight and scale "
            f"per expert); an unquantised bank has no scale to de-interleave "
            f"and no stacked shape to read, so it is refused by name here "
            f"rather than reaching the default loader, whose len(slices) == 1 "
            f"assertion would fail without naming this parameter and after it "
            f"was registered."
        )
    return None


# --------------------------------------------------------------------------- #
# The expert-stacked loader.
#
# One map entry holds a routed expert bank: E weights and E scale grids,
# interleaved weight-then-scale in checkpoint order. This section turns that entry
# into one tensor holding this rank's experts stacked on a new leading axis, and
# de-interleaves the scales onto the same axis.
#
# The rank's expert subset is selected by calling the platform's
# ``expert_parallel_interleaved_loader``, which already restricts a K-interleaved
# slice list to a rank's contiguous expert block; this map's weight-then-scale
# layout is that layout at K = 2. The stacking is what this section adds, because
# those wrappers require an inner loader to do it.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BankLayout:
    """Where one bank entry's weights and scales sit in its own key list.

    Produced by :func:`bank_layout`, which is the only thing that decides it, so
    a consumer never re-derives "every other key is a scale" for itself.
    """

    #: Positions of the expert weight keys, ascending.
    weight_at: tuple[int, ...]
    #: Positions of the expert scale keys, ascending.
    scale_at: tuple[int, ...]
    #: How many experts the entry enumerates.
    experts: int


def _refuse(param_name: str | None, tail: str) -> None:
    """Raise the bank refusal, always naming the parameter first.

    Every refusal in this section goes through here so the parameter name is never
    left to a call site to remember. A bank failure that names no parameter is very
    hard to attribute.
    """
    raise Glm5NextExpertBankNotLoadableError(
        f"{param_name or '<unnamed parameter>'} {tail}"
    )


def bank_layout(
    checkpoint_keys: str | Sequence[str], *, param_name: str | None = None
) -> BankLayout:
    """Read one bank entry's expert layout, or refuse it by name.

    The entry must alternate strictly weight-then-scale, one
    pair per expert, which is the layout ``_add_moe_mlp`` builds (``:679``,
    through ``_quantised``) and the layout
    ``expert_parallel_interleaved_loader`` documents at K = 2.

    Two malformed cases are refused by name, each naming the parameter and the
    defect:

    * an odd key count cannot be pairs at all;
    * a scale key where a weight belongs, or the reverse, is an entry whose pairing
      broke. Reported at the first offending position, which is the one a reader can
      act on.

    "Is this key a scale?" stays :func:`scale_keys`'s question and is asked of it
    rather than re-decided here.
    """
    keys = _as_key_list(checkpoint_keys)
    scales = set(scale_keys(keys))
    if len(keys) % 2 != 0:
        _refuse(
            param_name,
            f"maps to {len(keys)} checkpoint keys, which is an odd count, so it "
            f"cannot be one weight and one scale grid per expert. A stacked "
            f"expert bank alternates weight then scale, {len(scales)} of these "
            f"keys are scales.",
        )
    for position, key in enumerate(keys):
        wants_scale = position % 2 == 1
        is_scale = key in scales
        if wants_scale != is_scale:
            expected = "a scale key" if wants_scale else "a weight key"
            _refuse(
                param_name,
                f"maps to {len(keys)} checkpoint keys that do not alternate "
                f"weight then scale: position {position} holds "
                f"{key!r}, where {expected} belongs. A stacked expert bank is "
                f"read as one weight-and-scale pair per expert, so an entry "
                f"that breaks the pairing is refused rather than stacked into "
                f"the wrong order.",
            )
    return BankLayout(
        weight_at=tuple(range(0, len(keys), 2)),
        scale_at=tuple(range(1, len(keys), 2)),
        experts=len(keys) // 2,
    )


def _expert_parallel_rank_map(owner: object | None, param_name: str | None):
    """The map from the load's global rank to the bank partition's rank.

    The bank's expert partition is built over ``ep_degree``, the expert-parallel
    degree, while the load hands this file the global tensor-parallel rank. The two
    are not the same index, and this function converts between them. It is about the
    expert axis only -- the bank's intermediate width is a separate question, and at
    degree 1 that width is sharded across the whole world.

    At degree 1 the map is the constant 0, which the package says twice:
    ``get_neuron_ep_rank`` returns 0 when no expert-parallel group was initialised,
    and ``require_uniform_expert_partition``'s own refusal text notes that with
    expert parallelism off every expert is local on every rank.

    Above degree 1 the index is read from the group and never divided out of the
    global rank. ``_build_ep_group_ranks`` lays the ranks out through
    ``_build_2d_mesh``, which substitutes a non-contiguous mesh whenever the world is
    64 and the row is 8. That mesh's first row is ``[0, 1, 2, 3, 12, 13, 14, 15]``, so
    rank 12 belongs to group 0 while ``12 // 8`` is 1 -- dividing the global rank
    would place a bank on the wrong ranks on exactly the host this package targets.

    An owner that declares no degrees is passed through unchanged, so a caller that
    supplies only ``local_expert_indices`` and ``num_routed_experts`` keeps working;
    the partition's own bound check is still the guard behind this map.
    """
    ep_degree = getattr(owner, "ep_degree", None)
    tp_degree = getattr(owner, "tp_degree", None)
    if ep_degree is None:
        return lambda rank: rank
    ep_degree = int(ep_degree)
    if ep_degree == 1:
        return lambda rank: 0
    if tp_degree is None or int(tp_degree) <= 1:
        # One tensor-parallel rank means the only global rank is 0, which is also
        # its group index, so the identity is the read and not an assumption.
        return lambda rank: rank

    def read_expert_parallel_rank(rank: int) -> int:
        """This process's expert-parallel partition index, from the group.

        ``rank`` is deliberately unused: it is the global tensor-parallel rank, and
        on a non-contiguous mesh the partition index is not that rank divided by
        anything. The global rank is used once, by the one thing it answers -- the
        shard column check in :func:`_expert_parallel_shard_column`.
        """
        del rank
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_rank

        return int(get_neuron_ep_rank())

    return read_expert_parallel_rank


def _expert_parallel_shard_column(
    rank: int, tp_per_ep: int, ep_degree: int, param_name: str | None
) -> int:
    """This process's column inside its expert-parallel TP group, checked twice.

    Two independent answers to "which columns of each expert are mine" exist in this
    repository, and they do not always agree:

    * the group's own, ``get_neuron_ep_tp_group().rank_in_group``, which is what the
      mesh actually built; and
    * the consumer's, ``rank % tp_degree`` where ``tp_degree = world_size /
      ep_degree``, which ``functional/moe/moe_blockwise.py`` derives.

    The group's answer is taken and the consumer's is compared against it; a
    disagreement is refused by name, because it means the loader would place a column
    where the kernel will not look for it -- the weights load, every shape checks
    out, and the model computes the wrong function. The two disagree only when the
    mesh is non-contiguous: at world 64 they differ at expert-parallel degree 8 and
    agree at 16.

    At degree 1 there is no group to read, so the degree is checked first.
    ``get_neuron_ep_tp_group`` asserts on ``ep_degree > 1``, and a degree-1 load needs
    no group: the expert-parallel group is the whole world and the column is the
    rank's own place in it.

    Above degree 1 an uninitialised group is refused by name rather than left to that
    assert, which would land as a bare ``AssertionError`` naming no parameter in the
    middle of a load.
    """
    from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_tp_group

    if ep_degree <= 1:
        return rank % tp_per_ep

    try:
        group = get_neuron_ep_tp_group()
    except AssertionError:
        _refuse(
            param_name,
            f"is a routed expert bank whose owning module declares an "
            f"expert-parallel degree above 1, so its columns are divided among "
            f"{tp_per_ep} ranks inside one expert-parallel group -- but no EP-TP "
            f"group is initialised, so there is no group to read the column from "
            f"(parallel/neuron_parallel_state.py). Refusing rather than "
            f"falling back to the global rank, which on this platform's mesh is a "
            f"different column.",
        )
    column = int(group.rank_in_group)
    derived = rank % tp_per_ep
    if column != derived:
        _refuse(
            param_name,
            f"would be placed on two different columns by the two answers this "
            f"repository holds. The initialised EP-TP group puts global rank "
            f"{rank} at column {column}; the consumer derives the column as rank % "
            f"tp_per_ep = {rank} % {tp_per_ep} = {derived} "
            f"(functional/moe/moe_blockwise.py). The loader takes the "
            f"group's answer, so at this degree it would write column {column} "
            f"where the kernel reads column {derived} -- every shape would check "
            f"out and the model would compute the wrong function. This happens "
            f"when the mesh is non-contiguous "
            f"(parallel/neuron_parallel_state.py): at world 64 the two disagree at "
            f"expert-parallel degree 8 "
            f"and agree at the registered degree 16. Refused by name rather than "
            f"loaded wrong.",
        )
    return column


def _bank_expert_indices(
    owner: object | None, layout: BankLayout, param_name: str | None
):
    """The owning module's local-expert resolver, or refuse by name.

    The expert geometry lives in exactly one place, ``Glm5NextRoutedExperts`` in
    ``model_fp8.py``, and is read from there rather than derived a second time from
    the key count, so there is one partition and not two that can disagree.

    A bank whose owner carries no geometry cannot be placed: there is no answer to
    "which experts are mine", so stacking would be a guess.
    """
    resolve = getattr(owner, "local_expert_indices", None)
    if not callable(resolve):
        _refuse(
            param_name,
            f"maps to {layout.experts * 2} checkpoint keys, a bank of "
            f"{layout.experts} quantised experts, but its owning module "
            f"({type(owner).__name__}) DECLARES NO EXPERT GEOMETRY: no callable "
            f"local_expert_indices, so there is no answer to which experts this "
            f"rank owns. A bank cannot be stacked into a shape nobody declared, "
            f"so it is refused by name rather than placed by guess.",
        )
    declared = getattr(owner, "num_routed_experts", None)
    if declared is None:
        _refuse(
            param_name,
            f"maps to a bank of {layout.experts} quantised experts, and its "
            f"owning module ({type(owner).__name__}) declares "
            f"local_expert_indices but no num_routed_experts, so the entry's "
            f"expert count cannot be checked against the module's. Half a "
            f"geometry is refused like none at all.",
        )
    if int(declared) != layout.experts:
        _refuse(
            param_name,
            f"maps to a bank of {layout.experts} quantised experts while its "
            f"owning module declares {int(declared)}. The checkpoint and the "
            f"module disagree about how many experts exist, so stacking would "
            f"silently drop or invent one; refused instead.",
        )
    # What is returned takes the global rank the load supplies, maps it to the rank
    # the bank's partition was built over, and only then asks the owner which experts
    # are local, so the owner's two degrees are not confused for each other. The map
    # is built here, at construction, because it depends only on the owner's degrees --
    # which keeps this file's rule that every bank refusal is raised before any
    # parameter is registered.
    to_partition_rank = _expert_parallel_rank_map(owner, param_name)
    return lambda rank: resolve(to_partition_rank(rank))


def _stack_local_expert_weights(local_slices: list, rank: int) -> torch.Tensor:
    """Stack one rank's expert weights on a new leading axis, in expert order.

    :func:`~vllm_neuron.utils.weight_loader.expert_parallel_interleaved_loader` hands
    this the local experts' pairs, still interleaved, so the weights are the even
    positions. Slicing with ``[:]`` is what materialises a ``PySafeSlice``; the scales
    in the odd positions are never read, which is why the input is restricted before
    stacking rather than after.
    """
    return torch.stack([local_slices[i][:] for i in range(0, len(local_slices), 2)])


def _stack_local_expert_scales(param_name: str | None):
    """Build the transform that stacks one rank's expert scale grids.

    Each grid goes through :func:`compensate_block_scales` and
    :func:`report_floored_blocks`, exactly as :func:`blockwise_scale_loader` sends a
    lone grid, so a bank's scales are compensated by the same code as every other
    scale here and an engaged floor is still reported.
    """

    def transform(local_slices: list, rank: int) -> torch.Tensor:
        rows = []
        for position in range(1, len(local_slices), 2):
            compensation = compensate_block_scales(local_slices[position][:])
            report_floored_blocks(compensation, param_name)
            rows.append(compensation.scale_inv)
        return torch.stack(rows)

    return transform


def _column_of_each_expert(
    geometry: DeferredShardGeometry, param_name: str | None, stack_whole
):
    """Wrap a bank stacker so each expert contributes only this rank's column.

    The bank's experts are already divided across the expert-parallel groups; what one
    group still divides is each expert's intermediate width, among its
    ``geometry.num_shards`` (``tp_per_ep``) members. So this takes the stacker that
    reads whole experts and gives it a column instead.

    The column is read once per load, not once per expert: it is a property of this
    process's place in its group rather than of the expert. Reading it once also makes
    :func:`_expert_parallel_shard_column`'s disagreement check refuse before the first
    expert is touched rather than part way through a stack.

    The column is taken after stacking, for both weights and grids, which costs a full
    read: each rank reads each of its experts at full intermediate width and keeps its
    share. The grid has no alternative, because ``compensate_block_scales`` operates on
    the whole grid and a pre-sliced one would change what it computes. Matching the
    weight to it gives one implementation of the extent and guarantees that a weight's
    columns and its grid's rows end in the same place, which two independent ceiling
    divisions would not.
    """
    from vllm_neuron.utils.weight_loader import shard_tensor_at_load_time

    def transform(local_slices: list, rank: int) -> torch.Tensor:
        column = _expert_parallel_shard_column(
            rank,
            geometry.num_shards,
            geometry.expert_parallel_degree,
            param_name,
        )
        whole = stack_whole(local_slices, rank)
        # The stacked result carries a leading expert axis, so the geometry's
        # per-expert dim sits one place to the right. This is the one place the two
        # numberings meet.
        result = shard_tensor_at_load_time(
            whole,
            geometry.shard_dim + 1,
            geometry.num_shards,
            column,
            geometry.pad_to_multiple_of,
            geometry.pad_value,
            param_name,
        )
        # The bank's own arrival site for its extent requirement. Checked here and
        # not in ``_sharding_loader``, because the bank never reaches that function:
        # ``loader_for_mapped_keys`` sends a stacked bank to
        # ``stacked_expert_bank_loader``, which comes straight here.
        refuse_inadmissible_shard_extent(
            int(result.shape[geometry.shard_dim + 1]),
            geometry,
            param_name,
            shard_dim=geometry.shard_dim + 1,
        )
        return result

    return transform


def _bank_slice_count(slices: list, layout: BankLayout, param_name: str | None) -> None:
    """Refuse a load whose slice count is not the key count this entry declared.

    Defensive and cheap. The reader builds ``slices`` from the same key list this
    loader was constructed for, so a mismatch means the two drifted apart between
    materialisation and load. Checked here because the expert-parallel wrapper's own
    divisibility error names ``total_num_experts`` rather than this parameter.
    """
    if len(slices) != layout.experts * 2:
        _refuse(
            param_name,
            f"was built for a bank of {layout.experts} experts "
            f"({layout.experts * 2} checkpoint keys) but the load handed it "
            f"{len(slices)} slices, so the key list it was constructed for is "
            f"not the key list it is being asked to read.",
        )


def _stacked_bank_transform(
    layout: BankLayout,
    resolve,
    param_name: str | None,
    inner_transform,
):
    """One bank transform: this rank's experts, selected then stacked.

    The wrapper is built per load because
    :func:`~vllm_neuron.utils.weight_loader.expert_parallel_interleaved_loader`
    resolves its expert indices to a contiguous ``(lo, hi)`` at construction, while
    ``rank`` does not arrive until the load calls this transform. Building it here
    costs one object per load and keeps the partition arithmetic where it is owned;
    the alternative would thread a rank through every loader in this file for the sake
    of one case.

    The import is function-local so a load-path module does not depend on the
    expert-parallel wrapper at import time.
    """
    from vllm_neuron.utils.weight_loader import expert_parallel_interleaved_loader

    def transform(slices: list, rank: int) -> torch.Tensor:
        _bank_slice_count(slices, layout, param_name)
        local = list(resolve(rank))
        if not local:
            _refuse(
                param_name,
                f"is a bank of {layout.experts} experts, and its owning module "
                f"reports that rank {rank} owns none of them. An empty expert "
                f"subset has no tensor to stack; the shipped expert-parallel "
                f"loader refuses it too (utils/weight_loader.py), and "
                f"this refusal names the parameter as well.",
            )
        wrapper = expert_parallel_interleaved_loader(
            local,
            SafetensorsWeightLoader(transform=inner_transform),
            layout.experts,
        )
        return wrapper.transform(slices, rank)

    return transform


def stacked_expert_bank_loader(
    checkpoint_keys: str | Sequence[str],
    *,
    param_name: str | None = None,
    owner: object | None = None,
    geometry: DeferredShardGeometry | None = None,
) -> SafetensorsWeightLoader:
    """Load a routed expert bank's weights as one stacked tensor.

    Returns a loader whose transform, for the rank it is called with, selects that
    rank's experts and stacks their weight slices on a new leading axis in checkpoint
    order: element ``[e]`` of the result is expert ``local[e]``'s weight, and the
    result's element count is the sum of its own slices' counts rather than one
    expert's.

    It is composed under :func:`wrap_with_blockwise_fp8_downscale`, and that order is
    safe because :func:`downscale_fp8_weight_bytes` is elementwise -- multiply, clamp,
    cast -- so squeezing the stack and stacking the squeezed experts give the same
    bytes. Wrapping the stack costs one pass instead of one per expert, and on a
    platform needing no squeeze the wrapper returns this loader untouched.

    Every refusal happens at construction rather than at load, which the two-pass
    materialiser depends on: ``_materialise_declared_parameters`` chooses every loader
    before registering any parameter, so a refusal raised here leaves the module tree
    exactly as it arrived. Deferring one to transform time would land mid-load with
    half a tree registered.
    """
    layout = bank_layout(checkpoint_keys, param_name=param_name)
    resolve = _bank_expert_indices(owner, layout, param_name)
    # ``geometry`` is ``None`` for a load that shards nothing, which is not the same
    # as expert-parallel degree 1: at degree 1 ``tp_per_ep`` is the whole world, so a
    # geometry does come back and the column path below runs. ``None`` arrives only at
    # world size 1, or when the world size equals the degree so each group holds its
    # experts whole.
    stack = _stack_local_expert_weights
    if geometry is not None:
        stack = _column_of_each_expert(geometry, param_name, stack)
    return wrap_with_blockwise_fp8_downscale(
        SafetensorsWeightLoader(
            transform=_stacked_bank_transform(layout, resolve, param_name, stack)
        )
    )


def stacked_expert_scale_loader(
    checkpoint_keys: str | Sequence[str],
    *,
    param_name: str | None = None,
    owner: object | None = None,
    geometry: DeferredShardGeometry | None = None,
    block_size: tuple[int, int] = DEFAULT_WEIGHT_BLOCK_SIZE,
) -> SafetensorsWeightLoader:
    """Load a routed expert bank's scale grids as one stacked tensor.

    The mirror of :func:`stacked_expert_bank_loader` over the odd positions: this
    rank's experts, their grids compensated one expert at a time by
    :func:`compensate_block_scales`, stacked on the same leading axis in the same
    order. Row ``[e]`` of this result belongs to the weight at ``[e]`` of that one.

    Not wrapped in the weight downscale, and the asymmetry is deliberate: the 240
    squeeze applies to weight bytes, while a grid's own adjustment is
    :func:`compensate_block_scales`. That is the same split
    :func:`blockwise_scale_loader` and :func:`wrap_with_blockwise_fp8_downscale` make
    for a non-bank weight and its grid.

    No parameter is attached to this loader yet: ``Glm5NextRoutedExperts`` declares no
    bank scale parameter and ``_add_moe_mlp`` writes map entries only for
    ``experts.router_weight`` and ``experts.<leaf>_weight``. It is called directly,
    against the checkpoint's own grids.
    """
    layout = bank_layout(checkpoint_keys, param_name=param_name)
    resolve = _bank_expert_indices(owner, layout, param_name)
    # A sharded bank's grids are sharded with its weights, for the same reason as the
    # dense MLP's: a whole grid beside a column of weights describes the wrong blocks,
    # and the dequantisation would scale real rows by another column's scale. The
    # weight geometry is converted to the grid's by the one function that does that
    # conversion, so the block boundary is checked in one place and the padded entries
    # hold 1.0 rather than 0.0.
    stack = _stack_local_expert_scales(param_name)
    if geometry is not None:
        stack = _column_of_each_expert(
            shard_geometry_for_grid(geometry, param_name, block_size),
            param_name,
            stack,
        )
    return SafetensorsWeightLoader(
        transform=_stacked_bank_transform(layout, resolve, param_name, stack)
    )
