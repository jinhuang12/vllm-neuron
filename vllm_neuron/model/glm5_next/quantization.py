# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) quantization abstraction.

Parses the HuggingFace ``quantization_config`` and produces a
per-module-queryable :class:`QuantizationSpec` that modeling code can consult
to pick weight dtypes, scale handling, and kernel calls. This is the file
``config.py``'s own comment defers to, verbatim: *"Only the two fields this
module needs are lifted here; the quantization* **spec** *(scheme resolution,
per-module policy) is a separate concern and a separate file."*

Shape precedent
---------------
:mod:`vllm_neuron.model.llama3.quantization` is the shape this module follows --
a ``str``-valued :class:`QuantScheme` enum, a frozen
:class:`QuantizationSpec` dataclass with a uniform
:meth:`QuantizationSpec.get_scheme` lookup, a
``from_hf_quantization_config`` parser that returns ``None`` for an
unquantized checkpoint, and defensive invariants in ``__post_init__``.
Deliberately kept **torch-free** for the same reason llama3's is: parsing a
``quantization_config`` does not need ``torch.nn``, and the numerics that do
live in :mod:`vllm_neuron.model.glm5_next.weight_loaders_fp8`.

<-- MODEL-SPECIFIC: where llama3 carries NVIDIA ModelOpt **static per-tensor**
FP8, this checkpoint is **BLOCKWISE** FP8 with **dynamic** activation scales --
``quant_method = "fp8"``, ``weight_block_size = [128, 128]``,
``activation_scheme = "dynamic"``. So the scheme carries a 2-D block shape,
and a per-tensor scale is not merely a different number but a different
layout: one fp32 scale per ``[128, 128]`` tile of the weight, shipped in the
checkpoint as the ``weight_scale_inv`` companion key
(:data:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.FP8_SCALE_SUFFIX`).

What is NOT here, and where it lands
------------------------------------
llama3's ``resolve_attention_mlp_classes`` has no counterpart yet: it dispatches
onto ``model_static_fp8`` / ``model_mx_fp8`` module classes, and this package's
modeling file (``model_fp8.py``) is ``inc-glm53f-013``. Adding a dispatcher
against modules that do not exist would be a stub asserting nothing, so the
class dispatch lands with the module tree it dispatches to.

<-- ``inc-glm53f-023`` (WP6) adds the QUANT-METHOD half of that dispatch, and
only that half: :class:`BlockFp8QuantMethod` and :func:`resolve_quant_method`
below turn a parsed :class:`QuantizationSpec` into the *method* a call site
consults, while the module-class dispatch above stays absent. The distinction is
load-bearing -- parsing a ``weight_block_size`` into a spec (which this file
already did) is NOT the same as resolving a method for it, and until
``inc-glm53f-023`` a spec carrying ``(128, 128)`` resolved to nothing at all.

Why a SUPPORTED SET rather than "any two positive ints"
-------------------------------------------------------
:data:`SUPPORTED_WEIGHT_BLOCK_SIZES` is exactly ``{(128, 128)}``. Three grounds,
each cited rather than re-derived:

* the campaign's target is frozen at ``zai-org/GLM-5.3-Flash`` native blockwise
  FP8 e4m3 with ``weight_block_size [128, 128]`` (``approvals/DECISIONS.md``
  section 2);
* the substrate's block-quantisation granularity is the single constant
  ``BLOCK_QUANT_SIZE = 256`` (``nkilib`` ``core/moe/moe_cte/bwmm_shard_on_I.py``
  line 50, measured at ``increments/evidence-007.md``) -- it is structural and
  this campaign cannot change it;
* a 256-granular block is therefore exactly **four** ``[128, 128]`` checkpoint
  blocks (2 H-tiles x 2 I-tiles), which is the mapping ``inc-glm53f-024``
  authors and ``inc-glm53f-025`` consumes.

So a well-formed block shape that is not ``(128, 128)`` has no authored path.
It is refused HERE, at method resolution, with a named error -- not silently
carried into a retile and a kernel that cannot represent it.

:meth:`QuantizationSpec.from_hf_quantization_config` deliberately stays
PERMISSIVE (any two positive ints still parse into a spec). Narrowing the parser
would move a refusal that belongs to this arch's authored kernel path onto the
platform's config-time admission, which ``inc-glm53f-019`` owns and which this
increment does not touch.
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

    #: Blockwise FP8 (``float8_e4m3fn`` bytes) with one fp32 scale per 2-D
    #: weight tile and **dynamically** quantized activations. This is the
    #: DeepSeek-V3-style layout GLM-5.3-Flash ships: the scale companion is a
    #: 2-D grid, not a scalar, so it cannot be broadcast the way llama3's
    #: per-tensor scale is.
    FP8_BLOCK_DYNAMIC = "fp8_block_dynamic"


#: Schemes that KV caches are permitted to use. The blockwise weight scheme is
#: **not** among them: a KV cache has no weight-tile grid to carry scales for,
#: and this checkpoint's ``quantization_config`` declares no
#: ``kv_cache_quant_algo`` at all. A blockwise KV scheme is therefore rejected
#: rather than silently accepted as a synonym for "FP8 cache".
_VALID_KV_CACHE_SCHEMES: frozenset[QuantScheme] = frozenset({QuantScheme.NONE})

#: The block shape this checkpoint declares. Kept as the parser's fallback for
#: a config that omits ``weight_block_size`` while declaring ``quant_method
#: = "fp8"``, mirroring the default ``config.py`` already carries.
DEFAULT_WEIGHT_BLOCK_SIZE: tuple[int, int] = (128, 128)

#: ``activation_scheme`` values this module accepts. Only ``"dynamic"`` is
#: supported: a static activation scale would need a calibrated per-tensor
#: input scale in the checkpoint, which this one does not ship.
_SUPPORTED_ACTIVATION_SCHEMES: frozenset[str] = frozenset({"dynamic"})

#: ``quant_method`` values that mean "blockwise FP8" in this checkpoint family.
_FP8_QUANT_METHODS: frozenset[str] = frozenset({"fp8"})

#: The weight block shapes this arch has an authored block-fp8 path for, as
#: ``(rows, cols)``. See the module docstring for the three grounds. A spec may
#: legally *carry* another shape; :func:`resolve_quant_method` is what refuses to
#: build a method for one.
SUPPORTED_WEIGHT_BLOCK_SIZES: frozenset[tuple[int, int]] = frozenset(
    {DEFAULT_WEIGHT_BLOCK_SIZE}
)

#: The substrate's block-quantisation granularity, CITED not derived:
#: ``BLOCK_QUANT_SIZE = 256`` at ``nkilib`` ``core/moe/moe_cte/bwmm_shard_on_I.py``
#: line 50, the sole such constant in the ``bwmm_*`` family
#: (``increments/evidence-007.md``). Used in the refusal message below so a
#: reader of the failure learns why the shape has no path.
SUBSTRATE_BLOCK_QUANT_SIZE: int = 256


class UnsupportedWeightBlockSize(ValueError):
    """A well-formed ``weight_block_size`` this arch has no authored path for.

    A subclass of :class:`ValueError` on purpose: every existing caller of
    :meth:`QuantizationSpec.from_hf_quantization_config` catches ``ValueError``,
    and this error is a refusal of the same family. The distinct type exists so
    a call site can tell "this shape is unsupported" apart from "this config is
    malformed" without matching on message text.
    """


# ---------------------------------------------------------------------------
# The BF16 skip-list predicate -- ``inc-glm53f-079``
# ---------------------------------------------------------------------------
def keeps_bf16(name: str, skip: Sequence[str] | None) -> bool:
    """True when ``name`` is one the checkpoint keeps in BF16.

    THE FORK'S OWN RULE, REUSED RATHER THAN REPLACED. ``neuron_config.py``
    declares ``modules_to_not_convert`` as a *"Substring-match list of parameter
    FQNs to KEEP in bf16 … A parameter keeps bf16 iff any entry in this list is
    a substring of its fully-qualified name"*, and
    ``qwen3_vl/model_mxfp8.py``'s ``_keep_bf16`` is the landed consumer. This is
    that predicate, with the same semantics, so the two paths cannot drift into
    two different answers. No regex, no per-family table, no normalisation.

    An empty or ``None`` skip list keeps nothing in BF16, which is the
    behaviour every caller had before this increment.

    The substring rule is what lets a module-namespace entry
    (``model.layers.0.self_attn.q_proj``) match a checkpoint-namespace key
    (``model.language_model.layers.0.self_attn.q_proj``) at all: the entry is a
    substring of the key because ``language_model`` ends in the literal
    ``model``. That alignment is recorded in the design and no increment
    tightens it.
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
            Scheme applied to quantizable linear modules -- that is, to the
            modules :attr:`modules_to_not_convert` does NOT name.

            MIXED PRECISION IS NO LONGER A TODO HERE, and the decision moved
            where the TODO said it would: :meth:`get_scheme` answers per module
            off the checkpoint's own skip list (``inc-glm53f-079``). This field
            stays one scheme because the checkpoint declares one, so **a call
            site that reads ``linear_scheme`` directly now bypasses the skip
            list and will call a BF16 module block-FP8.** Query
            :meth:`get_scheme`.
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
            The checkpoint's own list of module names to KEEP in BF16, taken
            verbatim off ``quantization_config.modules_to_not_convert``
            (``inc-glm53f-079``). Empty means "quantize everything the scheme
            supports", which is what this class assumed before that increment.
            Stored as a tuple rather than the config's list so the frozen
            dataclass stays hashable, the same reason
            :attr:`weight_block_size` is a tuple.
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

        The answer is the checkpoint's own, since ``inc-glm53f-079``: a module
        named by :attr:`modules_to_not_convert` keeps BF16 and gets
        :attr:`QuantScheme.NONE`; everything else gets :attr:`linear_scheme`.
        Before that increment this returned :attr:`linear_scheme`
        unconditionally, so the answer was "block-FP8" for ``lm_head``, for the
        KDA projections and for the DSA indexer alike -- for 1,067 of the real
        checkpoint's 37,534 base tensors, all of them BF16.

        Args:
            layer_index: Transformer-block index (0-based) for modules inside a
                block (e.g. ``self_attn.q_b_proj``). ``None`` for modules
                outside any block (``lm_head``, ``embed_tokens``). Not consulted:
                the skip list carries its own layer qualification, so the layer
                is already inside ``prefix`` for every entry that names one.
            prefix: Qualified module name, e.g.
                ``"model.layers.3.self_attn.kv_b_proj"``. **QUALIFY IT.** The
                match is the fork's substring rule and 1,500 of the real
                checkpoint's 1,509 entries are qualified dotted paths (1,150
                under ``model.layers.``, 347 under ``visual.``, 3 top-level
                modules), so a leaf-style ``"kv_b_proj"`` cannot match one and
                would be reported quantized. The nine bare entries (``lm_head``,
                ``router``, ``visual``, ``dt_bias``, ``weights_proj``, and four
                more) match either way.

        Returns:
            The :class:`QuantScheme` to apply.

        Notes:
            The matcher is the fork's own and this method adds none of its own:
            see :func:`keeps_bf16`.
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

        ``config.py`` lifts ``quant_method`` / ``activation_scheme`` /
        ``weight_block_size`` / ``modules_to_not_convert`` off the top-level
        ``quantization_config`` while deliberately not modelling the spec. This
        is the bridge back, so a caller holding a parsed
        :class:`~vllm_neuron.model.glm5_next.config.Glm5NextConfig`
        does not have to keep the raw HF dict alive to get a spec. Untyped
        deliberately: importing the config module here would make a cycle out of
        a one-way dependency (``config`` -> nothing, this module -> ``config``),
        and the attribute names are the whole contract.

        ``inc-glm53f-079`` added the fourth name. A bridge that forwarded three
        of four would hand back a spec whose skip list is empty, and an empty
        skip list quantizes everything -- the exact defect this increment
        repairs, reintroduced one layer up.
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

    ``weight_block_size`` is validated as **two positive ints** rather than
    merely truthy: a 1-D or 3-D block shape would index the scale grid with the
    wrong rank, and the failure would land far away in the loader as a shape
    mismatch on a tensor nobody was looking at.
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

    # inc-glm53f-079: the checkpoint's BF16 skip list, carried verbatim and
    # only shape-checked. A list of non-strings would match nothing and would do
    # it silently, so it raises instead.
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
# Quant-method resolution -- ``inc-glm53f-023``
#
# A SPEC says what the checkpoint declares. A METHOD says what this arch will
# actually do about it. The two are separate objects because they answer to
# different authorities: the spec answers to the checkpoint's
# ``quantization_config``, the method answers to the block-fp8 path this
# campaign authors (``inc-glm53f-024`` retile, ``-025``/``-026`` kernels,
# ``-027`` call site).
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

    The two block extents are stored as separate ints rather than as a tuple
    because that is how call sites index a scale grid, and because a tuple field
    invites the shape being passed around as opaque data. :attr:`block_shape`
    gives the tuple form where one is wanted.
    """

    block_h: int
    block_w: int
    activation_scheme: str

    #: Fixed: this class exists for exactly one scheme. Present so a call site
    #: can branch on ``method.scheme`` uniformly across future methods.
    scheme: QuantScheme = QuantScheme.FP8_BLOCK_DYNAMIC

    @property
    def block_shape(self) -> tuple[int, int]:
        """``(block_h, block_w)`` as a tuple."""
        return (self.block_h, self.block_w)

    def __post_init__(self) -> None:
        """Refuse a block shape this arch has no authored path for.

        The membership test lives here rather than in
        :func:`resolve_quant_method` so that NO construction path -- including a
        direct call from a future call site -- can produce a method for an
        unsupported shape.
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

    This is the dispatcher gap ``inc-glm53f-023`` closes: before it, a parsed
    spec carrying ``weight_block_size (128, 128)`` reached no method at all.

    The route is deliberately NOT the vendor ``quantization_type=`` keyword.
    ``nkilib``'s quantisation enum carries no blockwise member at this pin, and
    adding one is a vendor change this campaign cannot make (plan section 11,
    constraint B.6), so the block-quant path calls the inner kernel directly
    (design decision D5(b)). Nothing here names or imports that enum.

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
