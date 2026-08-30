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
"""

from __future__ import annotations

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
            Scheme applied to quantizable linear modules.

            TODO(quant, mixed-precision): a single scheme for the whole model
            today, as llama3's spec also is. Per-layer / per-module mixed
            precision extends this field (or moves the decision into
            :meth:`get_scheme`); call sites that read ``linear_scheme``
            directly will need revisiting then.
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
    """

    linear_scheme: QuantScheme
    kv_cache_scheme: QuantScheme
    weight_block_size: tuple[int, int] | None = None
    activation_scheme: str | None = None

    # ------------------------------------------------------------------
    # Uniform per-module query
    # ------------------------------------------------------------------
    def get_scheme(
        self,
        layer_index: int | None,
        prefix: str,
    ) -> QuantScheme:
        """Return the scheme applied to the module at ``(layer_index, prefix)``.

        Args:
            layer_index: Transformer-block index (0-based) for modules inside a
                block (e.g. ``self_attn.q_b_proj``). ``None`` for modules
                outside any block (``lm_head``, ``embed_tokens``).
            prefix: Qualified module name. Either the full dotted path
                (``"model.layers.7.linear_attn.out_proj"``) or a leaf-style
                name (``"out_proj"``) is accepted.

        Returns:
            The :class:`QuantScheme` to apply. Today this is always
            :attr:`linear_scheme`; the signature is fixed now so call sites can
            be wired without churn when dispatch becomes richer.

        Notes:
            TODO(quant, mixed-precision): the hybrid stack is a live reason
            this will get richer -- the DSA indexer and the KDA convolution
            are the two families most likely to stay unquantized while the
            projections around them do not.
        """
        del layer_index, prefix  # reserved for future per-module dispatch
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
        ``weight_block_size`` off the top-level ``quantization_config`` while
        deliberately not modelling the spec. This is the bridge back, so a
        caller holding a parsed :class:`~vllm_neuron.model.glm5_next.config.Glm5NextConfig`
        does not have to keep the raw HF dict alive to get a spec. Untyped
        deliberately: importing the config module here would make a cycle out of
        a one-way dependency (``config`` -> nothing, this module -> ``config``),
        and the three attribute names are the whole contract.
        """
        quant_method = getattr(config, "quant_method", None)
        if not quant_method:
            return None
        return cls.from_hf_quantization_config(
            {
                "quant_method": quant_method,
                "activation_scheme": getattr(config, "activation_scheme", None),
                "weight_block_size": getattr(config, "weight_block_size", None),
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

    return QuantizationSpec(
        linear_scheme=QuantScheme.FP8_BLOCK_DYNAMIC,
        kv_cache_scheme=QuantScheme.NONE,
        weight_block_size=block_size,
        activation_scheme=activation_scheme,
    )
