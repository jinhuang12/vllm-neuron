# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 quantization spec parsing
=====================================

The checkpoint ``deepseek-ai/DeepSeek-V4-Flash-0731`` carries::

    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
        "activation_scheme": "dynamic"
    },
    "expert_dtype": "fp4"

Two distinct weight formats therefore coexist in one model:

* **block-128x128 FP8** (``float8_e4m3fn`` weights plus a
  ``[ceil(N/128), ceil(K/128)]`` UE8M0 scale grid) for the attention
  projections, the shared-expert MLP and the DSA indexer ``wq_b``. Served
  through :func:`vllm_neuron.functional.block_fp8_linear`.
* **1-byte FP8 with one power-of-two scale per output channel** for the 256
  routed experts, requantized from the checkpoint's MXFP4 + group-32 E8M0 at
  load time (LD-23). Trainium2 has neither an FP4 datapath nor an MX datapath
  — ``nisa.nc_matmul_mx`` is a NeuronCore-v4 instruction and this venue is v3
  (R-13) — so the group scales are folded into one per channel, which is what
  the gen3-legal MoE entry points take (``moe_cte`` PER_CHANNEL, ``moe_tkg``
  ROW). The fold is bit-exact or it raises at load; see the LD-23 section of
  ``weight_loaders.py``.

Everything else — the KV compressor, the indexer ``weights_proj``, norms,
the router gate, embedding and LM head — stays unquantized. That split is
upstream's (``vllm/model_executor/models/deepseek_v4.py`` at tag
``v0.21.0``, which passes ``quant_config`` to exactly those linears and
``None`` to the rest), so it is reproduced here as data rather than
re-derived per call site.

This module is the single place that maps a module name to a scheme. It
never touches the KV cache dtype: that comes from the serve flag
``--kv-cache-dtype`` and is applied by the runner, not by the checkpoint's
``quantization_config`` (which carries no ``kv_cache_quant_algo``).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "QuantScheme",
    "QuantizationSpec",
    "BLOCK_FP8_MODULES",
    "UNQUANTIZED_MODULES",
]


class QuantScheme(str, Enum):
    """Quantization scheme applied to a single module / tensor.

    Values are stable strings so they are safe to log, serialize, and
    compare.
    """

    #: No quantization; tensors stay in the model's compute dtype.
    NONE = "none"

    #: Block-128x128 FP8 (``float8_e4m3fn``) weights with a UE8M0
    #: (power-of-two) per-block dequant scale grid and dynamic per-token,
    #: per-128-element-K-group FP8 activation quantization.
    #:
    #: The bytes are LEGACY ``nl.float8_e4m3`` (bias 7, amax 240), not OCP
    #: ``float8_e4m3fn`` (amax 448). The torch dtype is a 1-byte carrier only;
    #: on trn2 the plugin maps between the two names
    #: (``nki/nki_dtype.py:43,51-53``). The checkpoint stores OCP-448 bytes by
    #: design, so the loader re-encodes every element into the legacy grid by
    #: HALVING it and DOUBLES the paired block scale to absorb that -- LD-24,
    #: ``weight_loaders.py`` ``_OCP_TO_LEGACY_HALVED_BYTES``. The doubled scale
    #: is still an exact power of two, so it is still exactly UE8M0.
    #:
    #: The dense/attention slice is NOT exempt from the carrier doctrine that
    #: :attr:`FP8_E4M3_PER_CHANNEL` records below. Believing it was is what
    #: produced this campaign's third replan.
    FP8_BLOCK_128X128 = "fp8_block_128x128"

    #: 1-byte FP8 weights with ONE power-of-two dequant scale per OUTPUT
    #: CHANNEL. The scheme the routed experts are actually served under: the
    #: loader requantizes the checkpoint's MXFP4 + group-32 E8M0 to this form
    #: (LD-23), and the gen3-legal MoE entry points consume it as
    #: ``moe_cte`` PER_CHANNEL / ``moe_tkg`` ROW.
    #:
    #: The bytes are LEGACY ``nl.float8_e4m3`` (bias 7, amax 240), not OCP
    #: ``float8_e4m3fn`` (amax 448). The torch dtype is a 1-byte carrier only;
    #: on trn2 the plugin maps between the two names
    #: (``nki/nki_dtype.py:43,51-53``).
    FP8_E4M3_PER_CHANNEL = "fp8_e4m3_per_channel"

    #: MXFP8: ``float8_e4m3fn`` weights with group-32 E8M0 scales.
    #:
    #: UNREFERENCED ON PURPOSE — kept in the enum, assigned to nothing. No MX
    #: weight path is reachable for THIS family on this campaign's trn2
    #: (= NeuronCore-v3) venue, for two separately measured reasons (R-13; probe
    #: artifact ``author_model_family-iter3/iter3-moe-gen3-probe.txt`` §3.3):
    #:
    #: - DECODE is refused outright: ``functional/moe/moe_tkg.py:455`` asserts
    #:   NeuronCore-v4 for MX, at every shape.
    #: - PREFILL is refused by SHAPE, not by generation. The MX prefill kernel
    #:   ``bwmm_shard_on_block_mx`` DOES lower on gen3 (measured PASS at
    #:   H=512), but gen3 caps a matmul moving free dimension at 512 and this
    #:   family's hidden size is 4096, so every family-shaped MX prefill matmul
    #:   violates the cap. Do not restate this as "MX cannot lower on gen3" —
    #:   that overclaim is contradicted by the probe.
    #:
    #: It stays declared because it is the correct scheme name for a Trn3 venue,
    #: where it becomes assignable again with no other change to this module;
    #: deleting it would erase that fact from the type.
    MXFP8_GROUP32 = "mxfp8_group32"


#: Leaf module names served by the block-FP8 linear path. Matched against
#: the last dotted component of a module prefix, so both
#: ``"model.layers.7.self_attn.wq_b"`` and ``"wq_b"`` resolve.
BLOCK_FP8_MODULES: frozenset[str] = frozenset(
    {
        "fused_wqa_wkv",  # 4096 -> [q_lora_rank 1024, kv_lora_rank 512], replicated
        "wq_b",  # q_lora_rank -> 64 heads x 512 (also the indexer's wq_b)
        "wo_a",  # 4096 -> o_groups x o_lora_rank, group-parallel
        "wo_b",  # o_groups x o_lora_rank -> 4096, row-parallel
        "gate_up_proj",  # shared-expert MLP, 16-way subgroup
        "down_proj",  # shared-expert MLP, 16-way subgroup
    }
)

#: Leaf module names that upstream explicitly constructs with
#: ``quant_config=None`` and that therefore stay in the compute dtype.
UNQUANTIZED_MODULES: frozenset[str] = frozenset(
    {
        "weights_proj",  # DSA indexer score weights
        "fused_wkv_wgate",  # KV compressor, consumed as a raw .weight.T
        "gate",  # MoE router
        "embed_tokens",
        "lm_head",
    }
)

#: Leaf module names whose weights are the routed experts.
_ROUTED_EXPERT_MODULES: frozenset[str] = frozenset({"experts", "routed_experts"})

_SUPPORTED_QUANT_METHODS = ("fp8",)
_SUPPORTED_EXPERT_DTYPES = ("fp4",)


@dataclass(frozen=True)
class QuantizationSpec:
    """Per-module-queryable view of the model's quantization configuration.

    A ``None``-valued ``DeepseekV4Config.quant_spec`` means "not
    quantized". When a spec is present, modeling code queries it uniformly
    via :meth:`get_scheme`.

    Attributes:
        linear_scheme: Scheme applied to the block-quantized linears.
        expert_scheme: Scheme applied to the routed experts.
        weight_block_size: ``(block_n, block_k)`` of the weight scale grid.
        scale_fmt: Weight-scale format string from the checkpoint.
        activation_scheme: ``"dynamic"`` or ``"static"``; the block path
            requires dynamic.
        expert_group_size: MX scale group size for the routed experts.
        checkpoint_expert_dtype: The checkpoint's ``expert_dtype`` value,
            kept so the loader knows what it is upcasting FROM.
    """

    linear_scheme: QuantScheme
    expert_scheme: QuantScheme
    weight_block_size: tuple[int, int] = (128, 128)
    scale_fmt: str = "ue8m0"
    activation_scheme: str = "dynamic"
    expert_group_size: int = 32
    checkpoint_expert_dtype: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Uniform per-module query
    # ------------------------------------------------------------------
    def get_scheme(self, layer_index: int | None, prefix: str) -> QuantScheme:
        """Return the scheme applied to the module at ``(layer_index, prefix)``.

        Args:
            layer_index: Transformer-block index (0-based) for modules
                inside a block, ``None`` for modules outside any block.
                Accepted and ignored: this model quantizes uniformly
                across layers, and the MTP block follows the same rule as
                the main stack.
            prefix: Qualified module name. Either the full dotted path
                (``"model.layers.7.self_attn.wq_b"``) or a leaf-style name
                (``"wq_b"``) is accepted.

        Returns:
            The :class:`QuantScheme` to apply to this module.

        Raises:
            ValueError: when ``prefix`` names no module this spec knows
                about. Failing loudly here is deliberate: a silent
                :attr:`QuantScheme.NONE` would serve a block-FP8 weight
                through a bf16 path and be discovered only as an accuracy
                miss after a full compile.
        """
        del layer_index  # uniform across layers, incl. the MTP block
        leaf = prefix.rsplit(".", 1)[-1]

        if leaf in BLOCK_FP8_MODULES:
            return self.linear_scheme
        if leaf in _ROUTED_EXPERT_MODULES:
            return self.expert_scheme
        if leaf in UNQUANTIZED_MODULES:
            return QuantScheme.NONE

        raise ValueError(
            f"Unknown module prefix={prefix!r} (leaf={leaf!r}) in "
            "DeepseekV4 QuantizationSpec.get_scheme. Add it to "
            "BLOCK_FP8_MODULES, UNQUANTIZED_MODULES or "
            "_ROUTED_EXPERT_MODULES so the scheme is explicit."
        )

    @property
    def block_n(self) -> int:
        """Weight-scale block extent along the output (N) dimension."""
        return self.weight_block_size[0]

    @property
    def block_k(self) -> int:
        """Weight-scale block extent along the input (K) dimension."""
        return self.weight_block_size[1]

    # ------------------------------------------------------------------
    # Construction from HuggingFace ``quantization_config``
    # ------------------------------------------------------------------
    @classmethod
    def from_hf_quantization_config(
        cls,
        quantization_config: dict[str, Any] | None,
        expert_dtype: str | None = None,
    ) -> "QuantizationSpec | None":
        """Parse a HuggingFace ``quantization_config`` dict.

        Args:
            quantization_config: The checkpoint's ``quantization_config``.
                ``None`` / falsy means the checkpoint is not quantized.
            expert_dtype: The checkpoint's top-level ``expert_dtype``
                field. It lives outside ``quantization_config`` upstream,
                so it is passed separately.

        Returns:
            A spec, or ``None`` when the checkpoint is not quantized.

        Raises:
            ValueError: when the config is a quantized format that is
                recognized but not supported, or when a required field is
                missing or malformed. A clear error here beats a silent
                fallback to bf16, which would change the served dtype
                without changing any recorded decision.
        """
        if not quantization_config:
            return None
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "Expected quantization_config to be a dict, got "
                f"{type(quantization_config).__name__}."
            )

        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if quant_method not in _SUPPORTED_QUANT_METHODS:
            raise ValueError(
                f"Unsupported quantization_config.quant_method={quant_method!r}. "
                f"DeepseekV4 supports: {list(_SUPPORTED_QUANT_METHODS)}."
            )

        fmt = str(quantization_config.get("fmt", "")).lower()
        if fmt != "e4m3":
            raise ValueError(
                f"DeepseekV4 requires quantization_config.fmt='e4m3', got {fmt!r}."
            )

        block_size_raw = quantization_config.get("weight_block_size")
        if not isinstance(block_size_raw, (list, tuple)) or len(block_size_raw) != 2:
            raise ValueError(
                "DeepseekV4 requires quantization_config.weight_block_size to be "
                f"a 2-element list, got {block_size_raw!r}."
            )
        block_size = (int(block_size_raw[0]), int(block_size_raw[1]))
        if block_size != (128, 128):
            raise ValueError(
                "DeepseekV4 on Neuron supports only weight_block_size=[128, 128] "
                f"(the block_fp8_linear kernel envelope), got {list(block_size)!r}."
            )

        scale_fmt = str(quantization_config.get("scale_fmt", "")).lower()
        if scale_fmt != "ue8m0":
            raise ValueError(
                "DeepseekV4 requires quantization_config.scale_fmt='ue8m0' "
                "(power-of-two block scales applied exactly), got "
                f"{scale_fmt!r}."
            )

        activation_scheme = str(
            quantization_config.get("activation_scheme", "")
        ).lower()
        if activation_scheme != "dynamic":
            # Upstream asserts ``not act_q_static`` for the block path
            # (vllm/model_executor/layers/quantization/fp8.py:359 @ v0.21.0).
            raise ValueError(
                "The block-128x128 FP8 path requires "
                "activation_scheme='dynamic' (per-token, per-128-K-group "
                f"activation scales), got {activation_scheme!r}."
            )

        expert_dtype_norm = (
            str(expert_dtype).lower() if expert_dtype is not None else None
        )
        if expert_dtype_norm is None:
            raise ValueError(
                "DeepseekV4 requires the checkpoint's top-level expert_dtype "
                "field; none was supplied. Expected one of "
                f"{list(_SUPPORTED_EXPERT_DTYPES)}."
            )
        if expert_dtype_norm not in _SUPPORTED_EXPERT_DTYPES:
            raise ValueError(
                f"Unsupported expert_dtype={expert_dtype_norm!r}. DeepseekV4 on "
                f"Neuron supports: {list(_SUPPORTED_EXPERT_DTYPES)} "
                "(requantized at load to 1-byte legacy-E4M3 with one "
                "power-of-two scale per output channel: Trainium2 has neither "
                "an FP4 datapath nor an MX datapath)."
            )

        return cls(
            linear_scheme=QuantScheme.FP8_BLOCK_128X128,
            expert_scheme=QuantScheme.FP8_E4M3_PER_CHANNEL,
            weight_block_size=block_size,
            scale_fmt=scale_fmt,
            activation_scheme=activation_scheme,
            checkpoint_expert_dtype=expert_dtype_norm,
        )
