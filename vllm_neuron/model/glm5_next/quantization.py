# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) quantization abstraction.

Parses the HuggingFace ``quantization_config`` into a per-module-queryable
:class:`QuantizationSpec` that modeling code consults to pick weight dtypes,
scale handling and kernel calls, and resolves that spec into the
:class:`BlockFp8QuantMethod` a call site acts on.

Follows the shape of :mod:`vllm_neuron.model.llama3.quantization`: a
``str``-valued :class:`QuantScheme` enum, a frozen :class:`QuantizationSpec`
dataclass with a uniform :meth:`QuantizationSpec.get_scheme` lookup, a
``from_hf_quantization_config`` parser that returns ``None`` for an unquantized
checkpoint, and defensive invariants in ``__post_init__``. Kept torch-free:
parsing a ``quantization_config`` does not need ``torch.nn``, and the numerics
live in :mod:`vllm_neuron.model.glm5_next.weight_loaders_fp8`.

<-- MODEL-SPECIFIC: where llama3 carries NVIDIA ModelOpt static per-tensor FP8,
this checkpoint is blockwise FP8 with dynamic activation scales --
``quant_method = "fp8"``, ``weight_block_size = [128, 128]``,
``activation_scheme = "dynamic"``. The scheme therefore carries a 2-D block
shape, and a per-tensor scale is not just a different number but a different
layout: one fp32 scale per ``[128, 128]`` tile of the weight, shipped as the
``weight_scale_inv`` companion key
(:data:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.FP8_SCALE_SUFFIX`).

llama3's ``resolve_attention_mlp_classes`` has no counterpart here. That
function dispatches onto per-scheme module classes; this package's modeling
module is selected by ``factory.py`` instead.

Why a supported set rather than "any two positive ints"
-------------------------------------------------------
:data:`SUPPORTED_WEIGHT_BLOCK_SIZES` is exactly ``{(128, 128)}``, because the
substrate's block-quantisation granularity is a fixed 256 (see
:data:`SUBSTRATE_BLOCK_QUANT_SIZE`) and a 256-granular block is exactly four
``[128, 128]`` checkpoint blocks -- 2 H-tiles by 2 I-tiles, which is the mapping
the retile producer writes and the block-FP8 kernel reads. Any other well-formed
block shape would need a different mapping, so it is refused at method
resolution with a named error rather than carried into a kernel that cannot
represent it.

:meth:`QuantizationSpec.from_hf_quantization_config` stays permissive by design:
any two positive ints still parse into a spec. Narrowing the parser would move
that refusal off this arch's kernel path and onto the platform's config-time
admission, which a different module owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------
class QuantScheme(str, Enum):
    """Quantization scheme applied to a single module / tensor.

    Values are stable strings so they are safe to log, serialize, and compare.
    """

    #: No quantization; tensors stay in the model's compute dtype.
    NONE = "none"

    #: Blockwise FP8 (``float8_e4m3fn`` bytes) with one fp32 scale per 2-D weight
    #: tile and dynamically quantized activations -- the DeepSeek-V3-style layout
    #: GLM-5.3-Flash ships. The scale companion is a 2-D grid, not a scalar, so it
    #: cannot be broadcast the way llama3's per-tensor scale is.
    FP8_BLOCK_DYNAMIC = "fp8_block_dynamic"


#: Schemes a KV cache may use. The blockwise weight scheme is not among them: a
#: KV cache has no weight-tile grid to carry scales for, and this checkpoint's
#: ``quantization_config`` carries no ``kv_cache_quant_algo``. A blockwise KV
#: scheme is rejected rather than taken as a synonym for "FP8 cache".
_VALID_KV_CACHE_SCHEMES: frozenset[QuantScheme] = frozenset({QuantScheme.NONE})

#: This checkpoint's block shape, and the parser's fallback for a config that
#: sets ``quant_method = "fp8"`` but omits ``weight_block_size``. Mirrors the
#: default in ``config.py``.
DEFAULT_WEIGHT_BLOCK_SIZE: tuple[int, int] = (128, 128)

#: ``activation_scheme`` values this module accepts. Only ``"dynamic"`` is
#: supported: a static activation scale would need a calibrated per-tensor
#: input scale in the checkpoint, which this one does not ship.
_SUPPORTED_ACTIVATION_SCHEMES: frozenset[str] = frozenset({"dynamic"})

#: ``quant_method`` values that mean "blockwise FP8" in this checkpoint family.
_FP8_QUANT_METHODS: frozenset[str] = frozenset({"fp8"})

#: The weight block shapes this arch has a block-fp8 path for, as
#: ``(rows, cols)``; see the module docstring for why the set is a singleton. A
#: spec may legally carry another shape -- :func:`resolve_quant_method` is what
#: refuses to build a method for one.
SUPPORTED_WEIGHT_BLOCK_SIZES: frozenset[tuple[int, int]] = frozenset(
    {DEFAULT_WEIGHT_BLOCK_SIZE}
)

#: The substrate's block-quantisation granularity, fixed by ``nkilib``'s
#: ``BLOCK_QUANT_SIZE`` in ``core/moe/moe_cte/bwmm_shard_on_I.py``. Quoted in the
#: refusal message below so a reader of the failure learns why the shape has no
#: path.
SUBSTRATE_BLOCK_QUANT_SIZE: int = 256


class UnsupportedWeightBlockSize(ValueError):
    """A well-formed ``weight_block_size`` this arch has no authored path for.

    A :class:`ValueError` subclass, because callers of
    :meth:`QuantizationSpec.from_hf_quantization_config` already catch that. The
    distinct type lets a call site tell "this shape is unsupported" from "this
    config is malformed" without matching on message text.
    """


# ---------------------------------------------------------------------------
# The BF16 skip-list predicate
# ---------------------------------------------------------------------------
def keeps_bf16(name: str, skip: Sequence[str] | None) -> bool:
    """True when ``name`` is one the checkpoint keeps in BF16.

    The rule is the fork's own, from ``neuron_config.py``'s
    ``modules_to_not_convert``: a parameter keeps bf16 when any entry in the list
    is a substring of its fully-qualified name. No regex, no per-family table, no
    normalisation. ``qwen3_vl/model_mxfp8.py``'s ``_keep_bf16`` applies the same
    predicate, so the two paths cannot drift apart.

    An empty or ``None`` skip list keeps nothing in BF16.

    The substring rule is what lets a module-namespace entry
    (``model.layers.0.self_attn.q_proj``) match a checkpoint-namespace key
    (``model.language_model.layers.0.self_attn.q_proj``): the entry is a substring
    of the key because ``language_model`` ends in the literal ``model``.
    """
    return bool(skip) and any(token in name for token in skip)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantizationSpec:
    """Per-module-queryable view of the model's quantization configuration.

    A ``None`` spec means "not quantized". When a spec is present, modeling
    code should query it uniformly via :meth:`get_scheme` regardless of which
    upstream producer built it.

    Attributes:
        linear_scheme:
            Scheme applied to quantizable linear modules, meaning the modules
            :attr:`modules_to_not_convert` does not name. One scheme, because the
            checkpoint carries one; mixed precision is expressed through the skip
            list instead. Read this field only when you mean the whole-model
            scheme -- a call site that reads it directly bypasses the skip list and
            will call a BF16 module block-FP8. Query :meth:`get_scheme`.
        kv_cache_scheme:
            Scheme applied to the KV cache. ``NONE`` for this checkpoint --
            see :data:`_VALID_KV_CACHE_SCHEMES`.
        weight_block_size:
            The ``(rows, cols)`` weight tile one fp32 scale covers, or ``None``
            when :attr:`linear_scheme` is not a blockwise scheme. Stored as a
            tuple rather than the config's list so the frozen dataclass stays
            hashable and the shape cannot be mutated by a caller.
        activation_scheme:
            The checkpoint's ``activation_scheme``, or ``None`` when
            unquantized.
        modules_to_not_convert:
            The checkpoint's own list of module names to keep in BF16, taken
            verbatim off ``quantization_config.modules_to_not_convert``. Empty
            means "quantize everything the scheme supports". Stored as a tuple
            rather than the config's list so the frozen dataclass stays hashable,
            for the same reason :attr:`weight_block_size` is a tuple.
    """

    linear_scheme: QuantScheme
    kv_cache_scheme: QuantScheme
    weight_block_size: tuple[int, int] | None = None
    activation_scheme: str | None = None
    modules_to_not_convert: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Uniform per-module query
    # ------------------------------------------------------------------
    def get_scheme(
        self,
        layer_index: int | None,
        prefix: str,
    ) -> QuantScheme:
        """Return the scheme applied to the module at ``(layer_index, prefix)``.

        The answer is the checkpoint's own: a module named by
        :attr:`modules_to_not_convert` keeps BF16 and gets
        :attr:`QuantScheme.NONE`; everything else gets :attr:`linear_scheme`. In
        the real checkpoint the skip list covers ``lm_head``, the KDA projections
        and the DSA indexer, about a thousand tensors in all.

        Args:
            layer_index: Transformer-block index (0-based) for modules inside a
                block (e.g. ``self_attn.q_b_proj``). ``None`` for modules
                outside any block (``lm_head``, ``embed_tokens``). Not consulted:
                the skip list carries its own layer qualification, so the layer
                is already inside ``prefix`` for every entry that names one.
            prefix: Qualified module name, e.g.
                ``"model.layers.3.self_attn.kv_b_proj"``. Qualify it: the match is
                a substring rule and nearly every skip-list entry is a qualified
                dotted path, so a leaf-style ``"kv_b_proj"`` matches nothing and
                would be reported quantized. The handful of bare entries
                (``lm_head``, ``router``, ``visual`` and a few more) match either
                way.

        Returns:
            The :class:`QuantScheme` to apply.

        Notes:
            The matcher is :func:`keeps_bf16`; this method adds no rule of its own.
        """
        del layer_index  # the skip list qualifies its own entries; see Args
        if keeps_bf16(prefix, self.modules_to_not_convert):
            return QuantScheme.NONE
        return self.linear_scheme

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def is_block_quantized(self) -> bool:
        """True when the linear scheme carries a 2-D weight block shape."""
        return (
            self.linear_scheme is QuantScheme.FP8_BLOCK_DYNAMIC
            and self.weight_block_size is not None
        )

    # ------------------------------------------------------------------
    # Construction from HuggingFace ``quantization_config``
    # ------------------------------------------------------------------
    @classmethod
    def from_hf_quantization_config(
        cls, quantization_config: dict[str, Any] | None
    ) -> QuantizationSpec | None:
        """Parse a HuggingFace ``quantization_config`` dict.

        Returns ``None`` when ``quantization_config`` is ``None`` or falsy
        (the checkpoint is not quantized).

        Raises:
            ValueError: when the config is a quantized format that is
                recognized but not supported, or when a required field is
                missing or malformed. A clear error beats a silent fallback to
                bf16, which is how a "why is this model slow and wrong"
                investigation starts.
        """
        if not quantization_config:
            return None
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "Expected quantization_config to be a dict, got "
                f"{type(quantization_config).__name__}."
            )

        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if quant_method in _FP8_QUANT_METHODS:
            return _parse_fp8_block(quantization_config)

        raise ValueError(
            f"Unsupported quantization_config.quant_method={quant_method!r}. "
            "GLM-5.3-Flash currently supports: 'fp8' (blockwise)."
        )

    @classmethod
    def from_model_config(cls, config: Any) -> QuantizationSpec | None:
        """Build from the fields ``Glm5NextConfig`` already lifted.

        ``config.py`` lifts ``quant_method``, ``activation_scheme``,
        ``weight_block_size`` and ``modules_to_not_convert`` off the top-level
        ``quantization_config`` without modelling the spec. This is the bridge
        back, so a caller holding a parsed
        :class:`~vllm_neuron.model.glm5_next.config.Glm5NextConfig` need not keep
        the raw HF dict alive to get a spec.

        ``config`` is untyped here on purpose: importing it would turn a one-way
        dependency into a cycle, and the four attribute names are the whole
        contract. All four are forwarded -- dropping the skip list would hand back
        a spec that quantizes everything.
        """
        quant_method = getattr(config, "quant_method", None)
        if not quant_method:
            return None
        return cls.from_hf_quantization_config(
            {
                "quant_method": quant_method,
                "activation_scheme": getattr(config, "activation_scheme", None),
                "weight_block_size": getattr(config, "weight_block_size", None),
                "modules_to_not_convert": getattr(
                    config, "modules_to_not_convert", None
                ),
            }
        )

    # ------------------------------------------------------------------
    # Defensive invariants
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.kv_cache_scheme not in _VALID_KV_CACHE_SCHEMES:
            raise ValueError(
                f"Unsupported kv_cache_scheme={self.kv_cache_scheme!r}; "
                f"expected one of {sorted(s.value for s in _VALID_KV_CACHE_SCHEMES)}."
            )
        if self.linear_scheme is QuantScheme.FP8_BLOCK_DYNAMIC:
            if self.weight_block_size is None:
                raise ValueError(
                    f"linear_scheme={self.linear_scheme.value!r} requires a "
                    "weight_block_size; got None."
                )
            if len(self.weight_block_size) != 2 or any(
                not isinstance(dim, int) or dim <= 0
                for dim in self.weight_block_size
            ):
                raise ValueError(
                    "weight_block_size must be two positive ints, got "
                    f"{self.weight_block_size!r}."
                )
        elif self.weight_block_size is not None:
            raise ValueError(
                f"weight_block_size={self.weight_block_size!r} is meaningless "
                f"for linear_scheme={self.linear_scheme.value!r}."
            )


# ---------------------------------------------------------------------------
# Blockwise-FP8 parsing (private)
# ---------------------------------------------------------------------------
def _parse_fp8_block(quantization_config: dict[str, Any]) -> QuantizationSpec:
    """Parse a blockwise-FP8 ``quantization_config``.

    Accepted shape -- the flat form this checkpoint ships::

        {
          "quant_method": "fp8",
          "activation_scheme": "dynamic",
          "weight_block_size": [128, 128]
        }

    ``weight_block_size`` is checked as two positive ints rather than merely
    truthy: a 1-D or 3-D block shape would index the scale grid with the wrong
    rank, and the failure would surface far away in the loader as a shape mismatch
    on an unrelated tensor.
    """
    activation_scheme = quantization_config.get("activation_scheme")
    activation_scheme = (
        str(activation_scheme).lower() if activation_scheme is not None else None
    )
    if activation_scheme not in _SUPPORTED_ACTIVATION_SCHEMES:
        raise ValueError(
            "GLM-5.3-Flash blockwise FP8 requires activation_scheme in "
            f"{sorted(_SUPPORTED_ACTIVATION_SCHEMES)}, got "
            f"{activation_scheme!r}."
        )

    raw_block = quantization_config.get("weight_block_size")
    if raw_block is None:
        block_size = DEFAULT_WEIGHT_BLOCK_SIZE
    else:
        if not isinstance(raw_block, (list, tuple)):
            raise ValueError(
                "quantization_config.weight_block_size must be a list of two "
                f"ints, got {type(raw_block).__name__}."
            )
        if len(raw_block) != 2:
            raise ValueError(
                "quantization_config.weight_block_size must have exactly two "
                f"entries (rows, cols), got {list(raw_block)!r}."
            )
        if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in raw_block):
            raise ValueError(
                "quantization_config.weight_block_size entries must be ints, "
                f"got {list(raw_block)!r}."
            )
        block_size = (int(raw_block[0]), int(raw_block[1]))

    # The checkpoint's BF16 skip list, carried verbatim and only shape-checked. A
    # list of non-strings would match nothing, and match nothing silently, so it
    # raises instead.
    raw_skip = quantization_config.get("modules_to_not_convert") or ()
    if not isinstance(raw_skip, (list, tuple)):
        raise ValueError(
            "quantization_config.modules_to_not_convert must be a list of "
            f"strings, got {type(raw_skip).__name__}."
        )
    if any(not isinstance(token, str) for token in raw_skip):
        raise ValueError(
            "quantization_config.modules_to_not_convert entries must be "
            "strings; got a non-string entry."
        )

    return QuantizationSpec(
        linear_scheme=QuantScheme.FP8_BLOCK_DYNAMIC,
        kv_cache_scheme=QuantScheme.NONE,
        weight_block_size=block_size,
        activation_scheme=activation_scheme,
        modules_to_not_convert=tuple(raw_skip),
    )


# ---------------------------------------------------------------------------
# Quant-method resolution
#
# A spec says what the checkpoint carries; a method says what this arch does about
# it. They are separate objects because they answer to different things: the spec
# to the checkpoint's ``quantization_config``, the method to what the block-fp8
# path (retile, kernels, call site) can actually run.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockFp8QuantMethod:
    """The blockwise-FP8 method resolved for a module.

    Attributes:
        block_h: Rows of the weight tile one fp32 scale covers.
        block_w: Columns of the same tile.
        activation_scheme: The checkpoint's ``activation_scheme``, carried so a
            call site does not have to keep the spec alive to know whether
            activations are quantized dynamically.

    The two extents are separate ints because that is how call sites index a scale
    grid; :attr:`block_shape` gives the tuple form where one is wanted.
    """

    block_h: int
    block_w: int
    activation_scheme: str

    #: Fixed -- this class exists for exactly one scheme. Present so a call site
    #: can branch on ``method.scheme`` uniformly across future methods.
    scheme: QuantScheme = QuantScheme.FP8_BLOCK_DYNAMIC

    @property
    def block_shape(self) -> tuple[int, int]:
        """``(block_h, block_w)`` as a tuple."""
        return (self.block_h, self.block_w)

    def __post_init__(self) -> None:
        """Refuse a block shape this arch has no authored path for.

        The membership test lives here rather than in :func:`resolve_quant_method`
        so that no construction path, including a direct call, can produce a method
        for an unsupported shape.
        """
        if self.scheme is not QuantScheme.FP8_BLOCK_DYNAMIC:
            raise ValueError(
                f"BlockFp8QuantMethod carries scheme={self.scheme.value!r}; only "
                f"{QuantScheme.FP8_BLOCK_DYNAMIC.value!r} is meaningful here."
            )
        if self.activation_scheme not in _SUPPORTED_ACTIVATION_SCHEMES:
            raise ValueError(
                "GLM-5.3-Flash blockwise FP8 requires activation_scheme in "
                f"{sorted(_SUPPORTED_ACTIVATION_SCHEMES)}, got "
                f"{self.activation_scheme!r}."
            )
        if self.block_shape not in SUPPORTED_WEIGHT_BLOCK_SIZES:
            raise UnsupportedWeightBlockSize(
                f"weight_block_size={list(self.block_shape)!r} has no block-fp8 "
                f"path in this build. Supported: "
                f"{sorted(SUPPORTED_WEIGHT_BLOCK_SIZES)}. The substrate's "
                f"block-quantisation granularity is fixed at "
                f"{SUBSTRATE_BLOCK_QUANT_SIZE}, which this arch's scale mapping "
                f"covers with four "
                f"{list(sorted(SUPPORTED_WEIGHT_BLOCK_SIZES)[0])!r} checkpoint "
                f"blocks; another shape would need a different mapping, which is "
                f"a design change rather than a configuration."
            )


def resolve_quant_method(
    spec: QuantizationSpec | None,
    layer_index: int | None = None,
    prefix: str = "",
) -> BlockFp8QuantMethod | None:
    """Return the quantisation method for the module at ``(layer_index, prefix)``.

    The route is deliberately not the vendor ``quantization_type=`` keyword:
    ``nkilib``'s quantisation enum carries no blockwise member, so the block-quant
    path calls the inner kernel directly. Nothing here names or imports that enum.

    Args:
        spec: Result of :meth:`QuantizationSpec.from_hf_quantization_config` or
            :meth:`QuantizationSpec.from_model_config`. ``None`` means the
            checkpoint is not quantized.
        layer_index: Zero-based transformer-block index, or ``None`` for modules
            outside any block. Forwarded to :meth:`QuantizationSpec.get_scheme`
            so per-layer dispatch lands there rather than here.
        prefix: Qualified or leaf module name, forwarded the same way.

    Returns:
        A :class:`BlockFp8QuantMethod` for a blockwise-FP8 module, or ``None``
        when the module is not quantized. ``None`` means "run the unquantized
        path", which is the same convention
        :meth:`QuantizationSpec.from_hf_quantization_config` uses for an
        unquantized checkpoint.

    Raises:
        UnsupportedWeightBlockSize: the scheme is blockwise FP8 but the block
            shape has no authored path.
        ValueError: the spec is internally inconsistent (a blockwise scheme with
            no block shape). :meth:`QuantizationSpec.__post_init__` already
            forbids that state, so this arm guards a spec built by some future
            path that bypasses it.
        NotImplementedError: the resolved scheme is one no method is wired for.
    """
    if spec is None:
        return None

    scheme = spec.get_scheme(layer_index, prefix)

    if scheme is QuantScheme.NONE:
        return None

    if scheme is QuantScheme.FP8_BLOCK_DYNAMIC:
        block = spec.weight_block_size
        if block is None:
            raise ValueError(
                f"spec.linear_scheme={scheme.value!r} carries no "
                "weight_block_size; a blockwise method cannot be resolved."
            )
        return BlockFp8QuantMethod(
            block_h=int(block[0]),
            block_w=int(block[1]),
            activation_scheme=str(spec.activation_scheme),
        )

    raise NotImplementedError(
        f"No quantisation method is wired for scheme {scheme.value!r} in "
        "vllm_neuron.model.glm5_next."
    )
