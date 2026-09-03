# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek-V4 model selection and fail-fast config validation.

There is ONE implementation path for this family: the mixed-precision path
(block-128x128 FP8 attention / shared-expert / indexer linears, MXFP8
group-32 routed experts, FP8 KV cache). ``_select_implementation``
therefore validates and returns that single implementation rather than
dispatching between variants.

``_validate_config`` fails at model construction — with a message naming
the offending dimension — rather than inside a kernel launch or, worse,
after a multi-thousand-second compile. Every rule below is either a
platform capability limit or an invariant the block-FP8 numerics depend
on.
"""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

logger = logging.getLogger(__name__)

#: KV cache dtypes this family supports. bf16 KV does not fit the recorded
#: production configuration, and upstream independently forces FP8 MLA KV
#: for this model (deepseek_v4_attention.py:694-716 @ v0.21.0).
_SUPPORTED_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3", "float8_e4m3fn")

#: The dynamic activation quantization group of the block-FP8 path. Every
#: block-FP8 linear must keep its local K extent a whole multiple of this,
#: or the shard boundary splits an activation group and silently changes
#: numerics relative to the GPU incumbent.
_ACTIVATION_GROUP = 128


class DeepseekV4ForCausalLM(nn.Module):
    """Factory that validates config and selects the DeepSeek-V4 implementation.

    Extends ``nn.Module`` to satisfy vLLM's ``ModelRegistry`` requirements.
    The factory stores the selected implementation and delegates
    ``forward()`` to it.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the configuration and instantiate the implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import DeepseekV4ForCausalLM as Model

        model = Model.from_configs(hf_config, neuron_config)
        cls._validate_kv_spec(model)
        return model

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Fail fast on any configuration this family cannot serve.

        Rules, each with the reason it exists:

        1. ``quantization_config`` must parse — wrong ``weight_block_size``,
           ``scale_fmt`` or ``activation_scheme``, or an unsupported
           ``expert_dtype``, is rejected here (see
           :meth:`QuantizationSpec.from_hf_quantization_config`).
        2. ``compress_ratios`` must cover every layer, or the per-layer KV
           class is undefined.
        3. TP must divide the query-head count; the head count is the TP
           ceiling for this architecture.
        4. Every block-FP8 linear must keep ``K_local % 128 == 0``.
        5. The KV cache dtype must be FP8.
        6. ``ep_degree`` must divide the routed-expert count.
        7. ``shared_expert_tp`` must divide both the TP degree and the
           shared-expert intermediate size.
        """
        from .config import DeepseekV4Config

        config = DeepseekV4Config.from_configs(hf_config, neuron_config)

        # Rule 1 is enforced inside from_configs -> QuantizationSpec parse.
        if config.quant_spec is None:
            raise ValueError(
                "DeepseekV4 on Neuron expects a quantized checkpoint "
                "(quantization_config with quant_method='fp8', "
                "weight_block_size=[128, 128]); none was found. An "
                "unquantized DeepSeek-V4 checkpoint is not a supported "
                "configuration for this family."
            )

        # Rule 2: compression schedule covers every layer.
        if len(config.compress_ratios) < config.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(config.compress_ratios)} entries but "
                f"the model has {config.num_hidden_layers} layers; the per-layer "
                "KV compression class would be undefined."
            )

        tp_size = cls._resolve_tp_size()

        # Rule 3: TP ceiling is the query-head count.
        if tp_size is not None:
            if config.num_attention_heads % tp_size != 0:
                raise ValueError(
                    f"tensor_parallel_size={tp_size} does not divide "
                    f"num_attention_heads={config.num_attention_heads}. The "
                    "query-head count is the TP ceiling for this MLA "
                    "architecture."
                )
            if tp_size > config.num_attention_heads:
                raise ValueError(
                    f"tensor_parallel_size={tp_size} exceeds the TP ceiling "
                    f"num_attention_heads={config.num_attention_heads}."
                )

            # Rule 4: block-FP8 activation-group alignment.
            misaligned = [
                (name, k_global, kind, k_local)
                for name, k_global, kind, k_local in config.block_fp8_linear_plan(
                    tp_size
                )
                if k_local % _ACTIVATION_GROUP != 0
            ]
            if misaligned:
                detail = "; ".join(
                    f"{name}: K_global={k_global} {kind}-sharded -> K_local={k_local}"
                    for name, k_global, kind, k_local in misaligned
                )
                raise ValueError(
                    "Block-FP8 activation-group alignment violated at "
                    f"tensor_parallel_size={tp_size}: every block-FP8 linear must "
                    f"keep K_local as a multiple of {_ACTIVATION_GROUP}, because the "
                    "activation quantization group boundary is what would otherwise "
                    f"shift relative to the reference implementation. Offenders: {detail}."
                )

            # Rule 7: shared-expert subgroup.
            if tp_size % config.shared_expert_tp != 0:
                raise ValueError(
                    f"shared_expert_tp={config.shared_expert_tp} must divide "
                    f"tensor_parallel_size={tp_size} so the subgroups tile the TP "
                    "group exactly."
                )
        if config.moe_intermediate_size % config.shared_expert_tp != 0:
            raise ValueError(
                f"shared_expert_tp={config.shared_expert_tp} must divide "
                f"moe_intermediate_size={config.moe_intermediate_size}."
            )

        # Rule 5: FP8 KV cache.
        cache_dtype = cls._resolve_kv_cache_dtype()
        if cache_dtype is not None and cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported by DeepseekV4 on "
                f"Neuron; expected one of {list(_SUPPORTED_KV_CACHE_DTYPES)}. The "
                "compressed-latent cache layout stores FP8 NoPE dims, and a bf16 "
                "cache does not fit the served context at this batch size."
            )

        # Rule 6: expert parallelism divides the routed-expert count.
        ep_degree = neuron_config.ep_degree if neuron_config is not None else 1
        if ep_degree > 1 and config.n_routed_experts % ep_degree != 0:
            raise ValueError(
                f"ep_degree={ep_degree} does not divide "
                f"n_routed_experts={config.n_routed_experts}."
            )

    @classmethod
    def _validate_kv_spec(cls, model: nn.Module) -> None:
        """Cross-check declared KV layer names against the layer list.

        A name mismatch between ``get_kv_spec()`` and the keys the model
        reads out of ``attn_metadata`` surfaces only on hardware, as a
        ``KeyError`` mid-compile or a silently unwritten cache. Checking it
        at construction turns that into a startup error.
        """
        spec = model.get_kv_spec()
        declared = [layer.name for layer in spec.layers]
        expected = set(model.expected_kv_layer_names())

        unknown = [name for name in declared if name not in expected]
        missing = sorted(expected - set(declared))
        duplicated = sorted({n for n in declared if declared.count(n) > 1})

        if unknown or missing or duplicated:
            raise ValueError(
                "get_kv_spec() does not match the model's attention layers: "
                f"unexpected={unknown}, missing={missing}, duplicated={duplicated}. "
                "The KV layer-name convention is "
                "'layers.{i}.self_attn' plus the port's '.rope', '.swa' and "
                "'.indexer' suffixes; both get_kv_spec() and every "
                "attn_metadata lookup must use exactly these names."
            )

    # ------------------------------------------------------------------
    # Environment lookups (best-effort: absent config is not an error)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tp_size() -> int | None:
        """Return the TP degree, or ``None`` when no vLLM config is set.

        Validation runs before the model is built, which normally happens
        inside a live vLLM config context. When it does not (a bare
        construction in a unit test), the TP-dependent rules are skipped
        rather than guessed.
        """
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            return int(vllm_config.parallel_config.tensor_parallel_size)
        except Exception:  # noqa: BLE001 - absent config is not a failure
            logger.debug("No vLLM config available; skipping TP-dependent validation.")
            return None

    @staticmethod
    def _resolve_kv_cache_dtype() -> str | None:
        """Return the configured KV cache dtype string, or ``None``."""
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            return str(vllm_config.cache_config.cache_dtype)
        except Exception:  # noqa: BLE001 - absent config is not a failure
            return None


class DeepSeekV4MTP(nn.Module):
    """Factory for the DeepSeek-V4 DSpark draft module.

    Named ``MTP`` because that is the architecture string upstream vLLM 0.21.0
    registers for this checkpoint's draft module
    (``"DeepSeekV4MTPModel"``) and the name the speculative config carries; what
    it actually builds is DeepSeek's DSpark block-parallel drafter
    (:class:`~.dspark_model.DeepseekV4DSparkDrafter`), which is a different
    design from upstream's one-extra-layer MTP. ``DSparkProposer``
    (``vllm_neuron/vllm/spec_decode/dspark.py``) resolves this class and drives
    one BLOCK of ``dspark_block_size`` drafted tokens per target step.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        """Create the draft model from configs."""
        return cls._select_implementation(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    @classmethod
    def _select_implementation(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        # The replan (ladder rows LD-18/19/20) settled what this builds. The
        # earlier version of this method raised NotImplementedError, because the
        # port plan's speculative row was transcribed from upstream's
        # DeepSeekV4MultiTokenPredictorLayer (``enorm``/``hnorm`` +
        # ``e_proj``/``h_proj`` + ``shared_head``) and NONE of those parameters
        # exists in the pinned checkpoint. Its ``mtp.*`` namespace holds DSpark:
        # three full decoder stages with ``main_proj``/``main_norm`` on stage 0
        # and ``markov_head``/``confidence_head`` on stage 2. The diagnosis and
        # the key census that settled it are in
        # ``artifacts/repairs/author_model_family-iter1/``.
        from .dspark_model import DeepseekV4DSparkDrafter

        cls._validate_config(config, neuron_config)
        return DeepseekV4DSparkDrafter.from_configs(
            config, start_layer_idx, neuron_config
        )

    @classmethod
    def _validate_config(
        cls, config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate the draft configuration.

        The draft shares the target's dimensions and therefore inherits every
        target rule; what is checked here is only what is draft-specific.

        ``num_nextn_predict_layers`` is deliberately NOT checked. The HF config
        declares it 1 while the checkpoint ships THREE draft stages
        (``mtp.{0,1,2}``) and the reference config declares ``n_mtp_layers = 3``;
        the field is recorded as contradicted by the weights
        (:data:`~.config.CONTRADICTED_CHECKPOINT_FIELDS`) and
        ``config.num_dspark_stages`` is the authority. An earlier version of this
        method enforced the field, which would have rejected the real
        checkpoint.
        """
        from .config import DeepseekV4Config

        parsed = DeepseekV4Config.from_configs(config, neuron_config)

        if parsed.num_dspark_stages < 1:
            raise ValueError(
                "DSpark needs at least one draft stage; config.num_dspark_stages "
                f"= {parsed.num_dspark_stages}."
            )
        if parsed.dspark_block_size < 1:
            raise ValueError(
                "DSpark drafts a block of at least one token; "
                f"config.dspark_block_size = {parsed.dspark_block_size}."
            )
        if not 0 <= parsed.dspark_noise_token_id < parsed.vocab_size:
            raise ValueError(
                f"dspark_noise_token_id={parsed.dspark_noise_token_id} is outside "
                f"the vocabulary [0, {parsed.vocab_size})."
            )
        bad_layers = [
            layer
            for layer in parsed.dspark_target_layer_ids
            if not 0 <= layer < parsed.num_hidden_layers
        ]
        if bad_layers:
            raise ValueError(
                f"dspark_target_layer_ids {bad_layers} are outside the target's "
                f"{parsed.num_hidden_layers} layers; the Eagle3 aux-hidden "
                "collection would never fire for them."
            )
        # Every stage must be MoE-and-SWA-shaped by its OWN layer index. This is
        # the load-bearing consequence of numbering the stages
        # ``num_hidden_layers + s``: it is what makes them take the gate.bias
        # branch (no ``tid2eid``) and declare no compressor or indexer, matching
        # the ``mtp.*`` key census. Checked here because getting it wrong is a
        # missing-key failure thousands of seconds into a load, not a crash.
        for stage in range(parsed.num_dspark_stages):
            layer_idx = parsed.num_hidden_layers + stage
            if parsed.is_hash_moe_layer(layer_idx):
                raise ValueError(
                    f"DSpark stage {stage} maps to layer_idx {layer_idx}, which "
                    "config classifies as a hash-MoE layer; the stages must take "
                    "the gate.bias branch, and no mtp.* key carries a tid2eid."
                )
            if parsed.has_compressed_cache(layer_idx):
                raise ValueError(
                    f"DSpark stage {stage} maps to layer_idx {layer_idx}, whose "
                    f"compress_ratio is {parsed.compress_ratio(layer_idx)}; the "
                    "stages are SWA-only (compress_ratios[43:46] == [0, 0, 0]) "
                    "and no mtp.* key carries a compressor."
                )

        # The same K_local % 128 == 0 obligation the target's Rule 4 carries,
        # over the DSpark stage's own linear inventory. tp_size comes from the
        # SAME resolver the target uses, and the check is skipped when it is
        # unavailable -- exactly as Rule 3 does -- because an absent vLLM config
        # is a CPU-side import, not a misconfiguration.
        tp_size = DeepseekV4ForCausalLM._resolve_tp_size()
        if tp_size is not None:
            for name, k_full, sharding, k_local in parsed.dspark_block_fp8_linear_plan(
                tp_size
            ):
                if k_local % _ACTIVATION_GROUP != 0:
                    raise ValueError(
                        f"DSpark block-FP8 linear {name!r} would have "
                        f"K_local={k_local} at tensor_parallel_size={tp_size} "
                        f"(K_full={k_full}, {sharding}-sharded); block-128x128 "
                        f"FP8 requires K_local % {_ACTIVATION_GROUP} == 0."
                    )

        DeepseekV4ForCausalLM._validate_config(config, neuron_config)
