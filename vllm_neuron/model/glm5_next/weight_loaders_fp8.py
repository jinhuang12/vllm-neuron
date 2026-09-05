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
from .quantization import keeps_bf16

#: The HF shard-index filename. Read for its ``weight_map`` object; see
#: :meth:`Glm5NextShardIndex.from_weight_map` for why loading it is lossy.
SHARD_INDEX_FILENAME = "model.safetensors.index.json"

#: Blockwise-FP8 scale companion suffix. A quantised ``<name>.weight`` in this
#: checkpoint is accompanied by ``<name>.weight_scale_inv`` holding the per-block
#: reciprocal scales (``weight_block_size = [128, 128]``,
#: ``activation_scheme = "dynamic"``). The suffix is mapped here; the numerics
#: that consume it are ``inc-glm53f-012``'s half.
FP8_SCALE_SUFFIX = "weight_scale_inv"

#: The checkpoint's own prefix for the whole text model. **Two namespaces, not
#: one:** every text-model tensor in this checkpoint is named
#: ``model.language_model.<...>`` while the module tree ``inc-glm53f-013`` landed
#: is named ``model.<...>``, so a mapping needs both strings and they are not
#: interchangeable. ``lm_head.weight`` is outside both and is spelled the same on
#: each side. Measured off the real index at ``inc-glm53f-078``; before that this
#: module had never seen a checkpoint.
CKPT_TEXT_PREFIX = "model.language_model"

#: The module-tree prefix -- what ``model_fp8.py`` calls the same tensors.
PARAM_TEXT_PREFIX = "model"

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
    # Both were PROVISIONAL until inc-glm53f-078 read the real index: each leaf
    # below is now a name the checkpoint itself carries, not a convention guess.
    "dsa_indexer": GROUNDED,
    "kda_linear_attention": GROUNDED,
    # Declared ABSENT until inc-glm53f-078 found all 270 keys in the index.
    "multi_hyper_connections": GROUNDED,
    "dense_mlp": GROUNDED,
    "moe_router": GROUNDED,
    "moe_routed_experts": GROUNDED,
    "moe_shared_experts": GROUNDED,
}

#: Families ``config.json`` requires that this module deliberately does **not**
#: map, and why. Declared rather than omitted: an absent family that nobody
#: wrote down is indistinguishable from one that was forgotten.
#:
#: ``multi_hyper_connections`` was here until ``inc-glm53f-078``. The declaration
#: was honest and it was wrong: the real index carries ``hc_attn_{base,fn,scale}``
#: and ``hc_ffn_{base,fn,scale}`` on every one of layers 0-44, so the leaf names
#: were settled all along and the family is mapped off the file rather than
#: invented.
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


def _quantised(
    prefix: str,
    leaf: str,
    *,
    quantised: bool,
    skip: Sequence[str] = (),
) -> list[str]:
    """Checkpoint key(s) for one projection: the weight, plus its FP8 scale.

    A blockwise-FP8 projection contributes **two** checkpoint keys, and both
    have to be referenced or the scale key shows up as unmatched. Norms, biases
    and embeddings are not quantised in this checkpoint and contribute one.

    **THE SCALE-SUPPRESSION PREDICATE** (``inc-glm53f-079``). ``skip`` is the
    checkpoint's own ``modules_to_not_convert``, and a tensor it names gets NO
    scale companion however ``quantised`` is set: the checkpoint keeps that
    tensor in BF16, so no scale key exists to ask for and asking makes the
    parameter unmatched. The predicate is :func:`keeps_bf16`, the fork's own,
    applied to the tensor's checkpoint name -- this function invents no rule of
    its own and holds no per-family table.

    An empty ``skip`` suppresses nothing, so every caller that passed no skip
    list keeps the behaviour it had before this increment.
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
    modules_to_not_convert: Sequence[str] = (),
) -> dict[str, str | list[str]]:
    """Build ``{param_name: checkpoint_key | [checkpoint_key, ...]}``.

    Follows the standalone-builder convention the fork already uses
    (``llama3/eagle3_model.py:666-668``,
    ``qwen3_vl/vision_encoder_bf16.py:885-887``) and the parameter-naming
    convention of the fork's MoE precedent -- ``{prefix}.mlp.experts.<x>_weight``
    (``gpt_oss/model_mxfp4.py:2336-2358``).

    The attention family per layer is chosen off ``text_config.layer_types``
    by **equality**, never substring: ``"attention"`` is a substring of both
    family names (``config.py:36-40``).

    **TWO PREFIXES, NOT ONE** (``inc-glm53f-078``). Each layer builds a
    ``ckpt_prefix`` in the checkpoint's namespace and a ``param_prefix`` in the
    module tree's, and both are threaded into every family adder. The parameter
    side is unchanged from ``inc-glm53f-011``; the checkpoint side gained
    ``language_model.`` because that is what the published index says. Keeping
    one string for both is the defect this increment repairs.

    **THE CHECKPOINT'S SKIP LIST DECIDES WHICH PROJECTIONS CARRY A SCALE**
    (``inc-glm53f-079``). Pass ``modules_to_not_convert`` and no tensor the
    checkpoint keeps in BF16 gets a ``weight_scale_inv`` companion asked for.
    The families that carry no scale at all are still declared structurally by
    their own adders, which ``inc-glm53f-078`` measured off the index; the two
    agree on this checkpoint, and each is a check on the other.

    Args:
        text_config: drives every count -- layer schedule, ``first_k_dense_replace``,
            ``n_routed_experts``, ``n_shared_experts``, ``tie_word_embeddings``.
        quantised: when True (the checkpoint's own case) a projection also
            references its ``weight_scale_inv`` companion -- unless the skip
            list keeps it in BF16.
        modules_to_not_convert: the checkpoint's own BF16 skip list, as
            ``Glm5NextConfig`` lifts it. Empty suppresses nothing, which is the
            behaviour before this increment.

    Returns:
        The mapping. Parameter names are this half's declaration and settle
        when ``model_fp8.py`` lands (``inc-glm53f-013``); the checkpoint-key
        side is what this increment's coverage measures.
    """
    mappings: dict[str, str | list[str]] = {}
    layer_types = list(text_config.layer_types or ())
    skip = tuple(modules_to_not_convert or ())

    # -- outside the layer stack (GROUNDED) --------------------------------- #
    # The two namespaces part company here: the parameter names are the module
    # tree's and stay byte-identical, the checkpoint keys gain the text-model
    # prefix the real index actually uses. ``lm_head.weight`` is the one key the
    # module already spelled right and it is unprefixed on both sides.
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

#: The DSA half's four scaled projections -- the ONLY ``self_attn`` leaves on a
#: sparse-attention layer that carry a ``weight_scale_inv`` companion. Measured:
#: ``kv_b_proj`` and every indexer leaf carry none, so asking for one makes the
#: parameter unmatched.
DSA_SCALED_PROJECTIONS: tuple[str, ...] = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "o_proj",
)

#: The KDA half's 15 leaves, as ``self_attn.*`` on each linear-attention layer.
#: **None is quantised** -- there is not one ``weight_scale_inv`` under any KDA
#: ``self_attn`` in the index. Split by whether the leaf has a ``.weight``.
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
    """The multi-hyper-connection leaves (GROUNDED at ``inc-glm53f-078``).

    Six bare tensors per layer, hanging off the layer and not off the attention
    or the MLP -- they are the layer's own residual-mixing state. Every one of
    layers 0-44 carries all six; the MTP layer carries none, which this function
    never has to know because that layer is not in ``layer_types``.

    Declared ABSENT before this increment because the leaf names were unknown.
    They were in the published index the whole time.
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

    <-- MODEL-SPECIFIC: ``mla_use_nope`` with ``qk_rope_head_dim == 0``
    (``config.py:119-127``) means there is **no** rotary head slice, so no
    ``*_rope_*`` projection is mapped. A reused DeepSeek-MLA mapping that
    assumes a RoPE split would ask for keys this checkpoint does not have.

    **18 checkpoint keys per layer: 14 tensors, 4 of which carry a scale.**
    Three corrections ``inc-glm53f-078`` measured off the real index:
    ``indexer.wq`` is really ``indexer.wq_b``; ``kv_b_proj`` and both indexer
    projections carry NO scale companion, so only the four in
    ``DSA_SCALED_PROJECTIONS`` ask for one; and ``indexer.k_norm.bias`` plus
    ``indexer.index_kpool_compress_{ape,gate}`` were mapped by nothing.
    """
    ckpt_attn = f"{ckpt_prefix}.self_attn"
    param_attn = f"{param_prefix}.self_attn"

    # inc-glm53f-085 (WP5 repair) owns ONLY which parameter each of these four
    # scale keys lands on. As landed, BOTH of a scaled projection's checkpoint
    # keys were bound to the single parameter ``{leaf}_weight``, and the default
    # loader -- no ``transform``, more than one slice -- refuses that with
    # "should only take in a single slice but got N"
    # (``vllm_neuron/utils/weight_loader.py:71-73``, an ``assert``, so under
    # ``python -O`` it instead keeps slice 0 and drops the scale silently). Either
    # way the scale slice reached no arithmetic. Splitting the keys gives the
    # scale a parameter of its own, which is what ``model_fp8.py`` dequantises
    # from. The checkpoint keys themselves are unchanged -- redistributed, never
    # added to -- which is what ``c078g_dsa_keys`` 198 and
    # ``c078g_dsa_scale_keys`` 44 hold fixed.
    for leaf in DSA_SCALED_PROJECTIONS:
        keys = _quantised(ckpt_attn, leaf, quantised=quantised, skip=skip)
        _add(mappings, f"{param_attn}.{leaf}_weight", keys[:1])
        # Zero or one scale key: ``_quantised`` returns the companion only when
        # this leaf really carries one, so a leaf the checkpoint keeps in BF16
        # maps no scale parameter and asks the checkpoint for nothing new.
        for scale_key in keys[1:]:
            _add(
                mappings,
                f"{param_attn}.{leaf}_{FP8_SCALE_SUFFIX}",
                [scale_key],
            )
    # kv_b_proj is a real projection with no scale in this checkpoint, so it sits
    # here rather than in the loop above. Not an oversight -- a reading.
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

    **15 checkpoint keys per layer and not one scale companion.** The family was
    wholly misnamed before ``inc-glm53f-078``: it was mapped as
    ``linear_attn.{in_proj_qkvz, in_proj_ba, out_proj, conv1d, norm}`` on the
    ``qwen3_next`` gated-delta convention, and the checkpoint carries 15 distinct
    ``self_attn.*`` leaves instead -- eight separate projections, three separate
    per-projection convolutions, an output norm, an output projection and two
    bare state tensors. There is no ``conv1d.bias`` anywhere in the index.

    ``quantised`` and ``skip`` are both accepted and deliberately unused: this
    family asks for no scale companion at any setting, so there is nothing for
    the skip list to suppress here. The parameters keep the arguments so the four
    adders share one call shape. **The two declarations agree and neither is
    redundant:** this one is structural and was measured off the index by
    ``inc-glm53f-078``, and the skip list says the same thing independently --
    every one of these 15 leaves is named in ``modules_to_not_convert``, which
    ``inc-glm53f-079``'s conjunct (b) counts through ``get_scheme``.
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

    Gate and up stay **separate** parameters, matching the fork's own dense
    precedent (``llama3/model.py`` maps ``mlp.gate_proj_weight`` and
    ``mlp.up_proj_weight`` one-to-one) rather than fusing them here.
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

    <-- MODEL-SPECIFIC: this checkpoint stores **one tensor per expert**
    (``model.layers.N.mlp.experts.E.gate_proj.weight``), the HF DeepSeek/GLM MoE
    convention, whereas the fork's only extant MoE precedent (``gpt_oss``) reads
    a single pre-stacked tensor for all experts. So the fork's per-projection
    expert parameter maps to a **list** of ``n_routed_experts`` checkpoint keys
    -- the list-valued branch of the mapping shape -- rather than to one key.

    ``topk_method = "noaux_tc"`` (``glm5_next/config.py:189``) is why the router carries
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

# ===========================================================================
# inc-glm53f-012 -- WP1: block-fp8 scale loading with the 240-max downscale
# ===========================================================================
#
# WHY THE IMPORTS FOR THIS HALF ARE HERE AND NOT AT THE TOP OF THE MODULE
# ----------------------------------------------------------------------
# The increment plan declares this increment's change to this file a **pure
# addition** -- "no line ``inc-glm53f-011`` landed moves" -- and inserting an
# import into the module's header block moves every one of the 628 lines below
# it. Python resolves module-level imports wherever they appear, so the
# constraint and the language agree here: this half's imports live in this
# half's section. The diff for this file is therefore an append, and that is
# checkable with ``git diff`` rather than asserted.

import logging

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

from .quantization import DEFAULT_WEIGHT_BLOCK_SIZE

#: Same convention as ``config.py:34`` in this package.
logger = logging.getLogger(__name__)

#: How many floored tile coordinates the warning names before it stops listing and
#: gives the count alone. A pathological grid can floor thousands of tiles, and a
#: warning that dumps every one of them is its own defect.
_FLOORED_BLOCKS_NAMED_IN_WARNING = 8

# --------------------------------------------------------------------------- #
# Blockwise-FP8 range constants
# --------------------------------------------------------------------------- #
#
# Trn2's kernels read fp8 SBUF as legacy ``nl.float8_e4m3`` (max finite 240),
# while this checkpoint's bytes are OCP ``float8_e4m3fn`` (max finite 448). The
# two encodings share a bit layout for every finite magnitude at or below 240,
# so squeezing the bytes into the 240 range and compensating the per-block
# dequant scale by the inverse factor preserves the dequantised tensor without
# reinterpreting a single byte pattern. This is the same downscale-plus-
# compensation as ``model/llama3/weight_loaders_static_fp8.py:51-110``, moved
# from per-parameter to per-block granularity.

#: The dtype the squeezed bytes are stored in. OCP ``float8_e4m3fn``, exactly as
#: the llama3 static path stores them: the grid of representable magnitudes at or
#: below 240 is shared with legacy ``e4m3``, so the trn2 kernel's
#: reinterpretation of these bytes is value-preserving.
_FP8_DTYPE = torch.float8_e4m3fn

#: Legacy ``nl.float8_e4m3`` max finite magnitude (trn2).
_FP8_E4M3_MAX = 240.0

#: OCP ``float8_e4m3fn`` max finite magnitude (the checkpoint's scale space).
_FP8_E4M3FN_MAX = 448.0

#: Applied to the weight bytes.
_FP8_WEIGHT_DOWNSCALE = _FP8_E4M3_MAX / _FP8_E4M3FN_MAX

#: Applied to the per-block dequant scales -- the exact inverse, so the product
#: ``byte * scale`` is preserved up to the bytes' own re-quantisation.
_FP8_SCALE_COMPENSATION = _FP8_E4M3FN_MAX / _FP8_E4M3_MAX

#: Floor for a stored per-block dequant scale.
#:
#: **Defined here, not imported.** The value is the increment plan's
#: (``design/increment-plan.md`` L3582-L3583, ``MINVAL = 1e-5``); it appears
#: nowhere else in this tree and it is **not** a registered comparator, so
#: nothing about it is frozen by the acceptance pre-registration.
#:
#: What it protects: ``activation_scheme`` is ``"dynamic"``, so the consuming
#: path derives activation scales at runtime from these weight scales, and a
#: block whose scale has collapsed towards zero (an all-zero weight tile, or a
#: quantiser that emitted a denormal) turns that reciprocal into ``inf``/``NaN``
#: and poisons the whole matmul. Flooring the stored scale keeps the reciprocal
#: finite. The floor is reported per block rather than applied silently: the census is
#: on :class:`BlockScaleCompensation.floored_blocks`, and on the checkpoint load path
#: :func:`report_floored_blocks` warns when it engaged, naming the parameter and the
#: tiles. Until ``inc-glm53f-012``'s r1 round (batch R3) that second half was
#: missing and this sentence was false of the loader that ships -- ``B08-F1``.
MINVAL = 1e-5


# --------------------------------------------------------------------------- #
# The platform gate -- CONDITIONAL, per lead ruling
# --------------------------------------------------------------------------- #


def resolved_fp8_clamp_max() -> float:
    """The FP8 clamp the vendor resolved for this process, read at call time.

    Reads :data:`vllm_neuron.utils.dtype_utils.FP8_CLAMP_MAX` through the module
    rather than binding it with a ``from`` import, so this half never takes an
    import-time snapshot of a value the vendor already resolved once at *its*
    import time (``dtype_utils.py:41``).

    The import is **lazy** on purpose. ``dtype_utils`` imports
    ``libtorch_neuronx_lite`` at module scope, and ``inc-glm53f-011``'s half of
    this file -- with its own passing tests -- must keep importing on a host
    where that vendor package is unavailable. A module-scope import here would
    make the skeleton half's importability depend on the numerics half's
    platform dependency.
    """
    from vllm_neuron.utils import dtype_utils

    return dtype_utils.FP8_CLAMP_MAX


def needs_240_downscale() -> bool:
    """True iff the resolved platform clamp is 240.0 -- the trn2 squeeze.

    **The downscale is CONDITIONAL, never unconditional.** Per
    ``approvals/lead-ruling-012-downscale-gate.md``: the squeeze exists because
    trn2's fp8 maximum is 240.0 while the checkpoint's OCP scale space is
    448.0-max, so *"the downscale applies IFF the resolved platform clamp is
    240.0 … On a 448.0-max platform an unconditional 240/448 rescale would
    corrupt correct weights"*. The condition is expressed against the vendor's
    own resolution rather than a second platform query of this module's own.

    Why the clamp and not ``get_platform_target()`` directly, which is what the
    llama3 static path uses (``weight_loaders_static_fp8.py:62-66``): that
    helper has **no CPU-mode fallback**, so a copy of it raises ``RuntimeError``
    on a bare CPU host with no NRT. ``_resolve_fp8_clamp_max()`` handles exactly
    that case (``dtype_utils.py:25-32``), and the clamp it returns *is* the
    quantity this squeeze is about.

    Not exercised in both directions by this increment's acceptance, and that is
    a recorded limitation rather than an oversight: the pinned Tier T instrument
    fixes ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` before collection
    (``test/conftest.py:23-25``), so it cannot discriminate the two readings.
    The ruling declines a trn3/448 discriminating test as out of this
    increment's scope (ruling item 3).
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


@dataclass(frozen=True)
class BlockScaleCompensation:
    """A compensated per-block scale grid, with the census of what it changed.

    The counts are the function's own report rather than something a caller
    recomputes: the increment's expected result is stated in counts (*"0 scales
    fall below MINVAL"*), and a count produced next to the transform cannot
    disagree with the transform.
    """

    #: The grid to store: compensated and floored, fp32.
    scale_inv: torch.Tensor
    #: False when the resolved clamp is not 240.0 -- the grid is then the input.
    applied: bool
    #: Scales strictly below :data:`MINVAL` in the **input** grid.
    below_minval_before: int
    #: Scales strictly below :data:`MINVAL` in the **stored** grid.
    below_minval_after: int
    #: ``(row, col)`` of every block whose scale the floor raised.
    floored_blocks: tuple[tuple[int, int], ...]


def compensate_block_scales(scale_inv: torch.Tensor) -> BlockScaleCompensation:
    """Multiply a block scale grid by 448/240 and floor it at :data:`MINVAL`.

    Conditional on :func:`needs_240_downscale`, and the floor travels with the
    compensation rather than standing alone: the floor exists to keep the grid
    *this* transform produces safe to take a reciprocal of, so on a platform
    where this transform is a no-op it stays a no-op and the checkpoint's own
    scales are stored as shipped. The census is measured either way, so a caller
    on either platform gets the same reported quantities.
    """
    grid = scale_inv.to(torch.float32)
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

    WHY THIS EXISTS. :data:`MINVAL`'s own note says the floor is "reported per block
    rather than applied silently", and :class:`BlockScaleCompensation` does carry the
    report -- but until ``inc-glm53f-012``'s r1 round (batch R3) the loader that
    ships, :func:`blockwise_scale_loader`, returned ``.scale_inv`` and dropped it.
    Nothing outside the test suite could see that a tile had been floored. A floored
    tile is not a cosmetic event: a stored scale of ``1e-6`` is raised to ``1e-5``, so
    every weight in that tile dequantises about 5.4x too large. Finding
    ``B08-F1-minval-floor-silent-on-loader-path`` recorded that as silent corruption on
    the load path, and this is the trace it asked for.

    IT WARNS AND DOES NOT RAISE. Raising would make a checkpoint that the floor exists
    to rescue fail to load instead, which is a change to declared behaviour and the
    lead's call rather than a seat's. The floor still does exactly what it did; the only
    difference is that a load which engages it now says so.

    The parameter name is optional because ``SafetensorsWeightLoader``'s transform
    signature is ``(slices, rank)`` and carries no name
    (``vllm_neuron/utils/weight_loader.py:55``). When a caller knows the name it passes
    it; when it does not, the message says so rather than printing ``None``.
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

    Conditional on :func:`needs_240_downscale`. The ``clamp`` is defensive: the
    largest OCP magnitude, 448, maps to exactly 240, which is itself
    representable, so no in-range input can exceed the bound. It is kept because
    a checkpoint carrying a non-finite or out-of-spec byte would otherwise store
    one, and because the llama3 precedent clamps at the same point
    (``weight_loaders_static_fp8.py:69-75``).
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

    The tensor-level entry point: the loader factories below wrap the same two
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

    The reference magnitude is the **block's** own absolute maximum, not the
    tensor's: a global abs-max normalisation would let the largest-scaled block
    set the tolerance for every other one and mask a small-scaled block's
    disagreement entirely.
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
    #: The single worst **per-element** relative difference in the block.
    #: Reported, never gated: it is disclosed so the choice of normalisation is
    #: visible in the evidence rather than implicit in a pass.
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

    ``rtol`` and ``atol`` are **required keyword arguments with no defaults**, on
    purpose: the fork's tolerance map has no fp8 entry and silently falls back to
    the bf16 pair, so every fp8 comparison in this campaign passes both
    explicitly (acceptance pre-registration, PIT-13).
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

    Same shape as ``weight_loaders_static_fp8.py``'s
    ``_wrap_with_fp8_downscale``: on a platform that needs no squeeze the
    original loader is returned unwrapped, so the identity path costs nothing.
    """
    if not needs_240_downscale():
        return loader
    base_transform = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return downscale_fp8_weight_bytes(base_transform(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def blockwise_scale_loader(param_name: str | None = None) -> SafetensorsWeightLoader:
    """Load a ``weight_scale_inv`` grid, compensated and floored.

    The grid is loaded whole -- one fp32 value per weight tile, so it is four
    orders of magnitude smaller than the weight and needs no sharding of its own
    at this increment. A sharded scale grid follows the weight's own shard
    geometry and lands with the module that declares that geometry
    (``model_fp8.py``, ``inc-glm53f-013``).

    An engaged floor is REPORTED, not discarded (:func:`report_floored_blocks`), which
    is finding ``B08-F1``'s repair. This transform used to return
    ``compensate_block_scales(...).scale_inv`` and drop ``applied``,
    ``below_minval_before``, ``below_minval_after`` and ``floored_blocks`` on the floor,
    so a load that silently inflated a whole tile of weights left no trace outside the
    test suite. Pass ``param_name`` when the caller knows which parameter this grid
    belongs to, so the warning can name it; the returned tensor is unchanged either way.
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
# SEAM -- inc-glm53f-091 (which loader serves which mapped key) lands below
# this line.
#
# This increment's WHOLE hand on this file: given one map entry's checkpoint
# key or keys, say which of the two loaders above serves it. It writes no map
# entry, no numeric and no family adder, and it changes nothing above this
# line. The parameter materialisation that consumes the answer lives in
# ``model_fp8.py``, which is this increment's other surface.
# --------------------------------------------------------------------------- #

#: A map entry naming one scale grid on its own.
MAPPED_KEY_SCALE_GRID = "scale_grid"

#: A map entry naming a quantised weight together with its scale companion.
MAPPED_KEY_QUANTISED_WEIGHT = "quantised_weight"

#: A map entry naming a whole bank of quantised experts -- one weight and one
#: scale grid per expert, interleaved weight-then-scale in checkpoint order.
#: ``inc-glm53f-095``'s fourth kind. It is a SEPARATE kind from
#: :data:`MAPPED_KEY_QUANTISED_WEIGHT` because the two need different loaders,
#: and both of this classifier's consumers have to know which one they hold --
#: see :func:`classify_mapped_keys` on why the second consumer is the reason
#: this is a kind rather than a branch hidden inside the loader chooser.
MAPPED_KEY_STACKED_BANK = "stacked_bank"

#: A map entry naming ordinary unquantised tensors.
MAPPED_KEY_PLAIN = "plain"


class Glm5NextExpertBankNotLoadableError(Glm5NextWeightMapError):
    """One map entry names a whole expert bank, which no loader here can stack.

    ``inc-glm53f-091a``, raised on the increment's own measured finding rather
    than on a review comment. Subclassing :class:`Glm5NextWeightMapError` is
    this file's shipped form for a named map refusal --
    :class:`DuplicateShardKeyError` (``:117``) is the other one -- so a caller
    that already handles a map error handles this one too, and it stays a
    ``ValueError`` like ``model_fp8.py``'s ``Glm5NextWeightLoadError``.

    Declared BELOW the seam rather than beside its siblings at the top of the
    file, for one reason worth stating: every byte above the seam is frozen at
    this increment and proved unchanged as a digest, so moving a class up there
    would cost that proof and buy nothing.
    """


def _as_key_list(checkpoint_keys: str | Sequence[str]) -> list[str]:
    """One map entry's checkpoint keys as a list, whether it named one or many.

    :func:`build_weight_mappings` stores a lone key as a bare string and a fused
    family as a list, so every consumer has to normalise before counting.
    Extracted so the two consumers below normalise the same way -- the same
    reason there is one classifier here and not two.
    """
    if isinstance(checkpoint_keys, str):
        return [checkpoint_keys]
    return list(checkpoint_keys)


def classify_mapped_keys(checkpoint_keys: str | Sequence[str]) -> str:
    """Which of the four kinds one map entry's checkpoint key(s) describe.

    ``inc-glm53f-091``. **ONE classifier with two consumers** -- the loader
    :func:`loader_for_mapped_keys` picks below, and the placeholder dtype
    ``model_fp8.py`` gives the parameter before the load. Deliberately one
    function rather than two: the pipelined loader reads its target dtype off
    the placeholder and warns whenever the tensor it built differs
    (``utils/checkpoints.py:437`` and ``:570-575``), so a placeholder typed
    against a different case than its own loader would make every load emit
    that warning. Two classifiers of the same three cases can drift apart; one
    cannot.

    **"Is this key a scale?" is not re-decided here.** :func:`scale_keys` above
    is the landed authority for that question -- its own docstring says so --
    and it is called rather than copied, so the dot-qualified suffix
    convention lives in exactly one place.

    The four cases are exhaustive by construction on the scale-key count: a key
    list holds no scale key, or exactly one alongside something else, or is a
    lone scale key, or holds more than one.

    AN EXPERT BANK IS THE FOURTH KIND, AND ``inc-glm53f-095`` IS WHERE IT
    BECAME ONE. An entry carrying MORE THAN ONE scale key is a whole bank of
    quantised experts, and it now says so -- :data:`MAPPED_KEY_STACKED_BANK` --
    because it has its own loader (:func:`stacked_expert_bank_loader`) and
    because a caller cannot pick between two loaders from one answer. Until
    this increment there was no second loader to pick, so the honest answer was
    ``MAPPED_KEY_QUANTISED_WEIGHT`` plus a refusal in the chooser.

    THE SECOND CONSUMER MOVED WITH IT, AND HAD TO. ``model_fp8.py``'s
    ``_placeholder_dtype`` branches on this answer, so a fourth kind it did not
    know would have fallen past its ``MAPPED_KEY_QUANTISED_WEIGHT`` arm to the
    default and typed every bank placeholder ``text_config.torch_dtype``
    instead of fp8 -- and the pipelined loader would then warn on every bank
    load, because it reads its target dtype off the placeholder and warns when
    the tensor differs (``utils/checkpoints.py:437``, ``:570-575``). That is the
    failure the paragraph above used to predict for a THIRD consumer, and it
    arrives just as easily through the second. So the two moved together:
    ``_placeholder_dtype`` returns ``_FP8_DTYPE`` for this kind, and "one
    classifier, two consumers" stays true rather than becoming "one classifier
    and one consumer that guesses". Measured from source before the change, not
    after it: ``increments/contradiction-095-refusal-collision.md``.
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


def loader_for_mapped_keys(
    checkpoint_keys: str | Sequence[str],
    *,
    param_name: str | None = None,
    owner: object | None = None,
) -> SafetensorsWeightLoader | None:
    """The loader that serves one mapped parameter, or ``None`` for the default.

    ``inc-glm53f-091``, extended by ``inc-glm53f-095``. Four cases, one per
    :func:`classify_mapped_keys` kind, and one REFUSAL that cuts across the
    last of them:

    * a lone scale grid gets :func:`blockwise_scale_loader`, which compensates
      the grid for the trn2 range and reports a floored block by name. Passing
      ``param_name`` is what lets that report say which parameter floored.
    * a quantised weight with ONE scale companion gets
      :func:`wrap_with_blockwise_fp8_downscale` over the default loader. That
      wrapper's base transform is ``slices[0][:]`` (``:1339``), so it keeps the
      weight slice and DROPS the scale companion -- the measured reason the
      scales are read out of band rather than through this map.
    * an EXPERT BANK -- more than one scale key -- gets
      :func:`stacked_expert_bank_loader`, which is what ``inc-glm53f-095``
      adds. ``owner`` is the module that declares the parameter, and the bank
      loader needs it: the expert geometry lives there and nowhere else.
    * a PLAIN entry naming MORE THAN ONE weight key is refused by name. It is
      an unquantised bank, and the reason it is refused rather than served is
      below.
    * anything else gets ``None``, meaning "attach nothing".
      :func:`~vllm_neuron.utils.weight_loader.get_weight_loader` already falls
      back to the identity loader when a parameter carries none
      (``utils/weight_loader.py:102``), so attaching one here would be a second
      way of saying the same thing, and two ways is one too many.

    THE BANK LOADER IS THE ONLY ONE HERE THAT CONSULTS ``rank``, and it consults
    it at load time rather than now. Everything else in this function is
    rank-blind, which is the landed position of this file:
    :func:`blockwise_scale_loader`'s own docstring records that a sharded grid
    "follows the weight's own shard geometry and lands with the module that
    declares that geometry" -- and a bank is exactly that case, which is why
    ``owner`` arrives here and why the geometry is read off the module instead
    of being recomputed.

    WHY AN UNQUANTISED BANK IS REFUSED (``B65-N1``). The refusal that used to
    live here was keyed on the SCALE-key count, which is the wrong count: an
    entry with several WEIGHT keys and no scale at all classifies
    ``MAPPED_KEY_PLAIN`` and would reach ``None``, meaning the default loader,
    which asserts ``len(slices) == 1`` (``utils/weight_loader.py:71-73``). Real
    failure, but a bare ``AssertionError`` naming no parameter and AFTER
    registration. No such entry exists on the real FP8 index today; the next
    quantisation profile that keeps a leaf in bf16 (``keeps_bf16``) produces
    one, and a named refusal costs one branch.

    WHY A BANK WAS EVER REFUSED, KEPT AS THE MEASUREMENT THAT PAID FOR THIS
    LOADER. On the real configuration 126 of the map's 1,416 entries carry 576
    keys each. Before this increment the wrapped loader's base transform was
    ``slices[0][:]`` (``:1339``), so a bank loaded expert 0 and dropped the
    rest, and nothing downstream noticed -- ``utils/checkpoints.py`` validates
    no shape and the lazy placeholder makes torch skip its own check.
    ``increments/evidence-091.md``: a four-expert miniature bank loaded
    ``LOADED_NUMEL=16384`` where its slices imply ``65536``, while the
    acceptance read 125 of 125 loaded and exit ``0``. The ``B65-N2``
    correction to that story: the BARE loader asserts rather than taking slice 0
    silently, so the loud path is the unwrapped one and the silent path the
    wrapped one. :func:`stacked_expert_bank_loader` now prevents it by
    construction rather than by refusal.
    """
    keys = _as_key_list(checkpoint_keys)
    kind = classify_mapped_keys(keys)
    if kind == MAPPED_KEY_SCALE_GRID:
        return blockwise_scale_loader(param_name)
    if kind == MAPPED_KEY_STACKED_BANK:
        return stacked_expert_bank_loader(keys, param_name=param_name, owner=owner)
    if kind == MAPPED_KEY_QUANTISED_WEIGHT:
        return wrap_with_blockwise_fp8_downscale(SafetensorsWeightLoader())
    if len(keys) > 1:
        raise Glm5NextExpertBankNotLoadableError(
            f"{param_name or '<unnamed parameter>'} maps to {len(keys)} "
            f"checkpoint keys carrying 0 scale keys, so it is a bank of "
            f"{len(keys)} UNQUANTISED experts rather than one plain tensor. "
            f"This module stacks a bank of QUANTISED experts (weight and scale "
            f"per expert); an unquantised bank has no scale to de-interleave "
            f"and no landed shape to stack into, so it is refused by name here "
            f"rather than reaching the default loader, whose len(slices) == 1 "
            f"assertion would fail without naming this parameter and after it "
            f"was registered."
        )
    return None


# --------------------------------------------------------------------------- #
# inc-glm53f-095 -- the expert-stacked loader.
#
# One map entry holds a routed expert bank: E weights and E scale grids,
# interleaved weight-then-scale in checkpoint order. This section turns that
# entry into ONE tensor holding this rank's experts stacked on a new leading
# axis, and de-interleaves the scales onto the same axis. It writes no map
# entry and no numeric.
#
# THE SUBSET SELECTION IS A CALL, NOT A REWRITE.
# ``expert_parallel_interleaved_loader`` (``utils/weight_loader.py:719``)
# already restricts a K-interleaved slice list to a rank's contiguous expert
# block, and its documented layout at ``K = len(slices) // total_num_experts``
# is exactly this map's weight-then-scale layout at K = 2. The STACKING is the
# new part: ``torch.stack`` occurs zero times in ``utils/weight_loader.py``, and
# those wrappers require an inner loader to do it (``:637-639``). This is that
# shipped loader's FIRST caller in the tree, so conjunct 3 exercises the rank
# subset directly rather than trusting its docstring examples.
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

    Every refusal in this section goes through here for one reason: ``B65-N1``
    and the ``-091a`` measurement both turned on a failure that did not name its
    parameter, so the name is not left to each call site to remember.
    """
    raise Glm5NextExpertBankNotLoadableError(
        f"{param_name or '<unnamed parameter>'} {tail}"
    )


def bank_layout(
    checkpoint_keys: str | Sequence[str], *, param_name: str | None = None
) -> BankLayout:
    """Read one bank entry's expert layout, or REFUSE it by name.

    ``inc-glm53f-095``. The entry must alternate strictly weight-then-scale, one
    pair per expert, which is the layout ``_add_moe_mlp`` builds (``:679``,
    through ``_quantised``) and the layout
    ``expert_parallel_interleaved_loader`` documents at K = 2.

    THE REFUSALS HERE ARE CONJUNCT (4)'s MALFORMED CASES, and each one names the
    parameter and the defect rather than a position alone:

    * an ODD key count cannot be pairs at all;
    * a scale key where a weight belongs, or a weight where a scale belongs,
      is an entry whose pairing broke -- reported at the FIRST offending
      position, because the first one is the one a reader can act on.

    "Is this key a scale?" is still :func:`scale_keys`'s question and is asked of
    it, not re-decided here -- the same discipline
    :func:`classify_mapped_keys` records.
    """
    keys = _as_key_list(checkpoint_keys)
    scales = set(scale_keys(keys))
    if len(keys) % 2 != 0:
        _refuse(
            param_name,
            f"maps to {len(keys)} checkpoint keys, which is an ODD count, so it "
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


def _bank_expert_indices(
    owner: object | None, layout: BankLayout, param_name: str | None
):
    """The owning module's local-expert resolver, or REFUSE by name.

    ``inc-glm53f-095``. The expert geometry is declared in exactly one place --
    ``Glm5NextRoutedExperts`` (``model_fp8.py:816``, geometry set at
    ``:886-891``), authored by
    ``inc-glm53f-031`` -- and it is read there rather than derived a second time
    from the key count, so there is one partition and not two that can disagree.

    A bank whose owner declares no geometry cannot be placed: there is no answer
    to "which experts are mine", so stacking would be a guess. This refusal is
    also conjunct (1)'s CONTROL THAT MOVES -- the same bank, in one run, loads
    E/E experts with the geometry declared and refuses by name without it --
    which is why reading (ii) of ``-091``'s first item was re-anchored onto it
    (design entry ``design-20260905-r``).
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
            f"local_expert_indices but NO num_routed_experts, so the entry's "
            f"expert count cannot be checked against the module's. Half a "
            f"geometry declaration is refused like none at all.",
        )
    if int(declared) != layout.experts:
        _refuse(
            param_name,
            f"maps to a bank of {layout.experts} quantised experts while its "
            f"owning module declares {int(declared)}. The checkpoint and the "
            f"module disagree about how many experts exist, so stacking would "
            f"silently drop or invent one; refused instead.",
        )
    return resolve


def _stack_local_expert_weights(local_slices: list, rank: int) -> torch.Tensor:
    """Stack one rank's expert WEIGHTS on a new leading axis, in expert order.

    The inner loader
    :func:`~vllm_neuron.utils.weight_loader.expert_parallel_interleaved_loader`
    hands this the local experts' pairs, still interleaved, so the weights are
    the even positions. Slicing with ``[:]`` is what materialises a
    ``PySafeSlice``; the scales in the odd positions are not read here at all,
    which is the point of restricting the input before stacking rather than
    after (``utils/weight_loader.py:631-634``).
    """
    return torch.stack([local_slices[i][:] for i in range(0, len(local_slices), 2)])


def _stack_local_expert_scales(param_name: str | None):
    """Build the transform that stacks one rank's expert SCALE grids.

    Each grid goes through :func:`compensate_block_scales` and
    :func:`report_floored_blocks` exactly as :func:`blockwise_scale_loader`
    sends a lone grid, so a bank's scales are compensated by the same code as
    every other scale in this file and an engaged floor is still REPORTED rather
    than discarded (``B08-F1``'s repair, which a second copy of the arithmetic
    here would have quietly undone).
    """

    def transform(local_slices: list, rank: int) -> torch.Tensor:
        rows = []
        for position in range(1, len(local_slices), 2):
            compensation = compensate_block_scales(local_slices[position][:])
            report_floored_blocks(compensation, param_name)
            rows.append(compensation.scale_inv)
        return torch.stack(rows)

    return transform


def _bank_slice_count(slices: list, layout: BankLayout, param_name: str | None) -> None:
    """Refuse a load whose slice count is not the key count this entry declared.

    Defensive and cheap. The reader builds ``slices`` from the same key list
    this loader was constructed for, so a mismatch means the two drifted apart
    between materialisation and load -- and the EP wrapper's own divisibility
    error (``_items_per_expert``, ``utils/weight_loader.py:670``, whose
    divisibility raise is at ``:684-688``) would name
    ``total_num_experts`` rather than this parameter.
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

    WHY THE SHIPPED WRAPPER IS BUILT PER LOAD.
    :func:`~vllm_neuron.utils.weight_loader.expert_parallel_interleaved_loader`
    resolves its expert INDICES to a contiguous ``(lo, hi)`` when constructed,
    while ``rank`` does not arrive until the load calls this transform. So it is
    built here, which costs one object per load and keeps the partition
    arithmetic where it is owned; the alternative would put a rank into every
    loader in this file for the sake of one case.

    The import is FUNCTION-LOCAL, following the landed ``inc-glm53f-023``
    precedent (``model_fp8.py:865-868``): this file's import blocks are earlier
    increments' D14 sections.
    """
    from vllm_neuron.utils.weight_loader import expert_parallel_interleaved_loader

    def transform(slices: list, rank: int) -> torch.Tensor:
        _bank_slice_count(slices, layout, param_name)
        local = list(resolve(rank))
        if not local:
            _refuse(
                param_name,
                f"is a bank of {layout.experts} experts, and its owning module "
                f"reports that rank {rank} owns NONE of them. An empty expert "
                f"subset has no tensor to stack; the shipped expert-parallel "
                f"loader refuses it too (utils/weight_loader.py:480-481), and "
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
) -> SafetensorsWeightLoader:
    """Load a routed expert bank's WEIGHTS as one stacked tensor.

    ``inc-glm53f-095``, and the answer to ``-091a``'s refusal. Returns a loader
    whose transform, for the rank it is called with, selects that rank's experts
    and stacks their weight slices on a new LEADING axis in checkpoint order --
    so element ``[e]`` of the result is expert ``local[e]``'s weight, and the
    result's element count is the sum of its own slices' counts rather than one
    expert's (which is what ``-091a`` measured going wrong:
    ``LOADED_NUMEL=16384`` of ``65536``).

    IT IS COMPOSED UNDER :func:`wrap_with_blockwise_fp8_downscale`, and that
    order is safe for a stated reason: :func:`downscale_fp8_weight_bytes` is
    elementwise -- multiply, clamp, cast (``:1169-1173``) -- so squeezing the
    stack and stacking the squeezed experts give the same bytes. Wrapping the
    stack costs one pass instead of E, and on a platform needing no squeeze the
    wrapper returns this loader untouched.

    EVERY REFUSAL HAPPENS AT CONSTRUCTION, NOT AT LOAD, and ``-091``'s two-pass
    materialiser depends on it: ``_materialise_declared_parameters`` chooses
    every loader BEFORE registering any parameter, so a refusal raised now leaves
    the tree byte-for-byte as it arrived and ``named_parameters()`` still reads
    exactly zero. Deferring one to transform time would land mid-load with half a
    tree registered.
    """
    layout = bank_layout(checkpoint_keys, param_name=param_name)
    resolve = _bank_expert_indices(owner, layout, param_name)
    return wrap_with_blockwise_fp8_downscale(
        SafetensorsWeightLoader(
            transform=_stacked_bank_transform(
                layout, resolve, param_name, _stack_local_expert_weights
            )
        )
    )


def stacked_expert_scale_loader(
    checkpoint_keys: str | Sequence[str],
    *,
    param_name: str | None = None,
    owner: object | None = None,
) -> SafetensorsWeightLoader:
    """Load a routed expert bank's SCALE GRIDS as one stacked tensor.

    ``inc-glm53f-095``. The mirror of :func:`stacked_expert_bank_loader` over
    the odd positions: this rank's experts, their grids compensated one expert at
    a time by :func:`compensate_block_scales`, stacked on the same leading axis
    in the same order. So row ``[e]`` of this result belongs to the weight at
    ``[e]`` of that one, which is what conjunct (2) reads.

    NOT WRAPPED IN THE WEIGHT DOWNSCALE, and the asymmetry is the point: the 240
    squeeze applies to weight BYTES, while a grid's own compensation is
    :func:`compensate_block_scales` -- the same split
    :func:`blockwise_scale_loader` and :func:`wrap_with_blockwise_fp8_downscale`
    already make for a non-bank weight and its grid.

    WHAT IT IS ATTACHED TO TODAY: NOTHING, DISCLOSED RATHER THAN PAPERED OVER.
    ``Glm5NextRoutedExperts`` declares five parameters and no bank scale
    parameter (``model_fp8.py:892-899``), and ``_add_moe_mlp`` writes map entries
    for exactly two shapes, ``experts.router_weight`` and
    ``experts.<leaf>_weight`` (``:661``, ``:679``). So a bank's stacked scales
    have no declared parameter to land in at this increment, and this loader
    exists to be CALLED -- by conjunct (2), against the checkpoint's own grids --
    rather than attached. Declaring and mapping that parameter is outside this
    increment's surface; the kernel-side consumption of E grids is
    ``inc-glm53f-095b``. Both are recorded in ``increments/evidence-095.md``.
    """
    layout = bank_layout(checkpoint_keys, param_name=param_name)
    resolve = _bank_expert_indices(owner, layout, param_name)
    return SafetensorsWeightLoader(
        transform=_stacked_bank_transform(
            layout, resolve, param_name, _stack_local_expert_scales(param_name)
        )
    )
