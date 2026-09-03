# SPDX-License-Identifier: Apache-2.0
"""
Neuron Platform implementation for vLLM

This module provides the NeuronPlatform class that integrates vLLM Neuron with vLLM's
platform plug-in system.
"""

import logging
import os
import sys
import time as _time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING

import torch
from vllm_neuron import envs

from vllm.platforms import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.inputs import EngineInput
    from vllm.sampling_params import SamplingParams
    from vllm.pooling_params import PoolingParams
else:
    VllmConfig = None

logger = logging.getLogger(__name__)

SO_DISABLED_MESSAGE = (
    "Structured outputs were requested, but this Neuron server was started "
    "with additional_config.neuron_config.enable_structured_outputs=false. "
    "Restart the server with enable_structured_outputs=true to serve "
    "structured-output requests. For this SO-off-only server, retry the "
    "request without structured_outputs."
)


# Relax DCP config validation for prefill-side DCP where TP <= num_kv_heads.
# Upstream asserts TP > num_kv_heads when DCP is enabled, but prefill DCP
# shards attention across the DCP sub-group, not full TP.
#
# During plugin registration, vllm.config.model may not be importable due to
# circular imports. We attempt eagerly; if that fails we use sys.addaudithook
# to patch as soon as the module loads. The hook self-disables after one
# successful patch, so ongoing overhead is a single boolean check per import.
_dcp_config_patched = False


def _patch_dcp_config_validation():
    """Patch ModelConfig.verify_with_parallel_config to skip DCP checks on prefill.

    Called during plugin registration. If the import fails (circular import),
    installs an audit hook that patches on first successful import of
    vllm.config.model. The hook is lightweight (one boolean check) after
    firing and cannot be removed per CPython API, but the cost is negligible
    compared to import overhead itself.
    """
    global _dcp_config_patched
    if _dcp_config_patched:
        return
    _dcp_config_patched = True

    try:
        from vllm.config.model import ModelConfig

        _apply_dcp_patch(ModelConfig)
    except (ImportError, AttributeError):
        # Circular import during plugin registration. Use an audit hook to
        # apply the patch as soon as vllm.config.model finishes loading.
        # The hook fires on every "import" event but self-disables (via
        # _applied flag) after the first successful patch, reducing ongoing
        # cost to a single boolean check per subsequent import.
        import sys

        _applied = [False]

        def _audit_hook(event, args):
            if _applied[0]:
                return
            if event == "import" and "vllm.config" in str(args[0]):
                mod = sys.modules.get("vllm.config.model")
                if mod is not None and hasattr(mod, "ModelConfig"):
                    _applied[0] = True
                    _apply_dcp_patch(mod.ModelConfig)

        sys.addaudithook(_audit_hook)


def _apply_dcp_patch(ModelConfig):
    """Monkey-patch verify_with_parallel_config to bypass DCP assertion for prefill."""
    if getattr(
        ModelConfig.__dict__.get("verify_with_parallel_config"),
        "_neuron_dcp_patched",
        False,
    ):
        return

    _orig_verify = ModelConfig.verify_with_parallel_config

    def _patched_verify(self, parallel_config):
        dcp = parallel_config.decode_context_parallel_size
        if dcp > 1 and not self.use_mla:
            saved_dcp = parallel_config.decode_context_parallel_size
            parallel_config.decode_context_parallel_size = 1
            _orig_verify(self, parallel_config)
            parallel_config.decode_context_parallel_size = saved_dcp
            return
        _orig_verify(self, parallel_config)

    _patched_verify._neuron_dcp_patched = True
    ModelConfig.verify_with_parallel_config = _patched_verify


class NeuronPlatform(Platform):
    # OOT (Out-Of-Tree) refers to hardware platforms that are integrated with vLLM
    # through plugin projects.
    _enum = PlatformEnum.OOT
    # Note we need to make sure libtorch_neuronx_lite has been imported before
    # getting here. This is to guarantee the privateuseone binding and
    # neuron renaming has been registered with torch.
    device_name: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    device_type: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = [
        "neuron_quant",
        "compressed-tensors",
        "modelopt",
        "fp8",
    ]
    # Neuron quantization paths that CPU-dequant compressed-tensors weights to
    # BF16 on the loader thread. On these paths the device only ever sees BF16
    # (the graph is identical to bf16), so an on-disk compressed-tensors weight/
    # activation quant config describes the checkpoint, not what the device runs.
    # The KV-cache-only guard in _validate_quantization_config is waived for
    # them -- gated on operator intent (neuron_config.quantization), NOT the
    # checkpoint format, so pointing at an MXFP8 checkpoint WITHOUT selecting
    # this path still fails loud.
    _cpu_dequant_quantizations: frozenset[str] = frozenset({"mxfp8"})
    device_control_env_var: str = "NEURON_VISIBLE_DEVICES"
    _device_count: int = -1
    _termination_timeout_patched: bool = False
    # Max embeds a single image can produce. Set once in check_and_update_config.
    # None means "no vision validation". In embed (post-merge) space so
    # validate_request can compare directly against mm_pos.length.
    _max_embeds_per_image: int | None = None
    # Structured outputs are request-level, while the SO-off perf mode is a
    # server-level compile choice. Validate requests before they reach the
    # engine so a bad request returns a normal request error.
    _enable_structured_outputs: bool = False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # TODO: Return based on hardware generation like GPUs and TPUs
        # It is only used for logging and debugging - hence not critical
        return f"neuron:{device_id}"

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """Register Neuron model architectures before ModelConfig validation."""
        import os

        from vllm.model_executor.models.registry import ModelRegistry

        # Lazy string, not a class object: the pre-validation hook must stay
        # import-free, and the string resolves through the package re-export.
        ModelRegistry.register_model(
            "Glm5NextForConditionalGeneration",
            "vllm_neuron.model.glm5_next:Glm5NextForConditionalGeneration",
        )

        if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
            ModelRegistry.register_model(
                "SyntheticNeuronModel",
                "vllm_neuron.model.synthetic:SyntheticNeuronModel",
            )

    #: The page this platform defaults to when the operator supplies none.
    #: Named so the two readers below share one number instead of each carrying
    #: its own literal.
    UNIFORM_NEURON_PAGE = 32

    @classmethod
    def resolved_uniform_page(cls, vllm_config: "VllmConfig") -> int:
        """The page ``update_block_size_for_backend`` will leave in place.

        READ THIS RATHER THAN ASSUMING 32. The default only applies when the
        operator supplied no block size; ``--block-size 64`` latches
        ``user_specified_block_size`` and the default is skipped, so a caller
        that hard-codes 32 is right only for the unlatched half of the domain.
        The non-engagement warning in :meth:`check_and_update_config` used to be
        such a caller.

        Safe to call before ``update_block_size_for_backend`` has run, which is
        why it exists: the warning fires earlier in ``check_and_update_config``
        than the block-size resolution below it.
        """
        cache_config = vllm_config.cache_config
        if cache_config.user_specified_block_size:
            return int(cache_config.block_size)
        return cls.UNIFORM_NEURON_PAGE

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        """Default block_size to 32 for Neuron when the user didn't override."""
        cache_config = vllm_config.cache_config
        if cache_config.user_specified_block_size:
            return
        cache_config.block_size = cls.UNIFORM_NEURON_PAGE

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        """Default optimization_level to O1 on Neuron (vLLM defaults to O2).

        O1 preserves the optlevel vllm-neuron has always compiled with.
        vLLM doesn't track whether the user set the level explicitly, so an
        explicit O2 is detected from argv (CLI only); other levels pass
        through. Offline LLM(optimization_level=O2) is indistinguishable
        from the default and also maps to O1 — pass O3 to force -O3.
        """
        from vllm.config.vllm import OptimizationLevel

        explicit = any(a.startswith("--optimization-level") for a in sys.argv)
        if explicit:
            logger.info(
                "Explicit --optimization-level: keeping O%s",
                vllm_config.optimization_level.value,
            )
            return
        # Not explicit on the CLI. Non-O2 values can only come from the
        # offline API — respect them; only the ambiguous default lowers.
        if vllm_config.optimization_level == OptimizationLevel.O2:
            vllm_config.optimization_level = OptimizationLevel.O1
            logger.info("Defaulting optimization level to O1 on Neuron")

    @classmethod
    def _resolve_vision_auto_config(
        cls, vllm_config: "VllmConfig", model_config
    ) -> None:
        """Auto-generate vision token buckets when not explicitly configured. Only called for models with hf_config.vision_config."""
        from vllm_neuron.model.neuron_config import VisionNeuronConfig
        from vllm_neuron.utils.bucket_utils import (
            get_default_num_vision_tokens_buckets,
            next_power_of_2,
        )

        if "vision_neuron_config" not in vllm_config.additional_config:
            vllm_config.additional_config["vision_neuron_config"] = {}
        vnc_dict = vllm_config.additional_config["vision_neuron_config"]
        block_size = vnc_dict.get(
            "vision_attention_block_size",
            VisionNeuronConfig.vision_attention_block_size,
        )

        mm_config = getattr(model_config, "multimodal_config", None)
        user_max_pixels = None
        if mm_config and mm_config.mm_processor_kwargs:
            user_max_pixels = mm_config.mm_processor_kwargs.get("max_pixels")
        if user_max_pixels:
            from vllm_neuron.utils.vision_utils import get_max_pixels_token_count

            per_image_tokens = get_max_pixels_token_count(
                model_config.hf_config, user_max_pixels
            )
            if per_image_tokens is not None and per_image_tokens > block_size:
                block_size = next_power_of_2(per_image_tokens)
                vnc_dict["vision_attention_block_size"] = block_size
                logger.warning(
                    "Block size increased to %d to fit max_pixels=%d. Performance may be degraded for lower-resolution inputs.",
                    block_size,
                    user_max_pixels,
                )

        if vnc_dict.get("num_vision_tokens_buckets"):
            if vnc_dict.get("max_vision_seq_len"):
                raise ValueError(
                    "Cannot set both num_vision_tokens_buckets and max_vision_seq_len in vision_neuron_config. Use num_vision_tokens_buckets for explicit control, or max_vision_seq_len to cap auto-generated buckets."
                )
            max_explicit = max(vnc_dict["num_vision_tokens_buckets"])
            if max_explicit < block_size:
                raise ValueError(
                    f"Largest bucket ({max_explicit}) is smaller than vision_attention_block_size ({block_size}). Increase num_vision_tokens_buckets or reduce max_pixels."
                )
            return

        from vllm_neuron.utils.vision_utils import get_vision_token_merge_factor

        merge_factor = get_vision_token_merge_factor(model_config.hf_config)
        max_model_len = vllm_config.model_config.max_model_len
        max_bucket = next_power_of_2(max_model_len * merge_factor)

        max_vision_seq_len = vnc_dict.get("max_vision_seq_len") or block_size
        if max_vision_seq_len < block_size:
            raise ValueError(
                f"max_vision_seq_len ({max_vision_seq_len}) must be >= "
                f"vision_attention_block_size ({block_size})."
            )
        max_bucket = min(max_bucket, max_vision_seq_len)

        buckets = get_default_num_vision_tokens_buckets(max_bucket, block_size)
        vnc_dict["num_vision_tokens_buckets"] = buckets

        logger.info(
            "Auto-generated vision buckets %s (max_model_len=%d, max_vision_seq_len=%d).",
            buckets,
            max_model_len,
            max_vision_seq_len,
        )
        if max_vision_seq_len < next_power_of_2(max_model_len * merge_factor):
            logger.info(
                "Vision buckets are capped by max_vision_seq_len=%d. "
                "Set max_vision_seq_len in vision_neuron_config to increase.",
                max_vision_seq_len,
            )

    @staticmethod
    def _reject_v2_model_runner() -> None:
        """Reject an explicit request for vLLM's V2 model runner.

        Neuron always runs its own v1-style NeuronModelRunner and has no V2 model
        runner. When V2 is enabled, the base scheduler stops sending
        CachedRequestData.all_token_ids, which our worker reads -- so honoring V2
        would crash mid-decode with a KeyError -> EngineDeadError. The plugin's
        _init_backend defaults VLLM_USE_V2_MODEL_RUNNER to "0" so V2 is never
        turned on implicitly; this guard catches the case where a user sets it
        truthy explicitly, failing fast with an actionable message instead of the
        confusing runtime crash.
        """
        raw = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
        if raw is not None and raw.strip().lower() in ("1", "true", "yes", "on"):
            raise ValueError(
                "VLLM_USE_V2_MODEL_RUNNER is set to a truthy value, but the "
                "Neuron platform does not implement vLLM's V2 model runner "
                "(it always uses the v1-style NeuronModelRunner). Unset the "
                "variable or set VLLM_USE_V2_MODEL_RUNNER=0."
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Configure vLLM for vLLM Neuron platform."""
        cls._reject_v2_model_runner()

        # Patches for bidirectional KV transfer and streaming KV (DI features).

        cls._patch_termination_timeouts()

        model_config = vllm_config.model_config
        if model_config is None:
            return

        # Frontend multimodal patchify runs under vLLM's
        # set_default_torch_num_threads(), which pins torch to OMP_NUM_THREADS or
        # 1 when unset -- so patchify runs single-threaded by default. Default OMP
        # to a small count so it parallelizes. Workers inherit this and reset to 1
        # in NeuronWorker.init_device; setdefault lets an operator value win.
        if model_config.multimodal_config is not None:
            os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))

        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        enable_structured_outputs = neuron_config.get(
            "enable_structured_outputs", False
        )
        if not isinstance(enable_structured_outputs, bool):
            raise ValueError(
                "enable_structured_outputs must be a boolean, "
                f"got {enable_structured_outputs}"
            )
        cls._enable_structured_outputs = enable_structured_outputs

        # ---- Does THIS architecture get the hybrid KDA/DSA KV cache? --------
        # inc-glm53f-081. The knob's own comment in
        # vllm_neuron/model/neuron_config.py promises "the platform turns it on
        # for the archs that need it"; this is that decision. It is taken HERE,
        # above the limb below, so that not one byte of section 6's registered
        # derivation moves: the decision is about WHETHER TO ENTER the limb,
        # never about what the limb computes.
        #
        # Three questions in order, and the first "no" ends the decision.
        # Question 3 pre-checks the ONE precondition that decides whether the
        # registered value is valid at all. The bfloat16 precondition is
        # deliberately NOT pre-checked: it stays inside the limb, so a default
        # run at the registered degree with a non-bf16 KV cache still reaches
        # the loud guard there instead of being quietly skipped here.
        HYBRID_KV_REGISTERED_TP_DEGREE = 64  # DECISIONS.md section 6
        architectures = getattr(model_config.hf_config, "architectures", None) or ()
        if (
            # 1. Is the resolved architecture this campaign's? Tested on the
            #    same string pre_register_and_update registers above. The
            #    config-side spelling glm5_next is NOT a second predicate --
            #    one identity, read in one place.
            "Glm5NextForConditionalGeneration" in architectures
            # 2. Did the operator leave the knob unset? PRESENCE, not
            #    falsiness: an explicit False is an operator decision this path
            #    honours and an explicit True is passed through untouched, so
            #    an operator-explicit run stays byte-for-byte what it was.
            and "enable_hybrid_kv_cache" not in neuron_config
        ):
            # 3. Is the tensor-parallel degree the registered one?
            resolved_tp_size = vllm_config.parallel_config.tensor_parallel_size
            if resolved_tp_size == HYBRID_KV_REGISTERED_TP_DEGREE:
                # Set in the SAME mapping the limb reads through its own .get,
                # so the limb needs no edit -- and stored back under
                # additional_config, because the .get above returns a FRESH {}
                # when the key is absent and a later NeuronConfig construction
                # would otherwise read a decision this path never published.
                neuron_config["enable_hybrid_kv_cache"] = True
                vllm_config.additional_config["neuron_config"] = neuron_config
            else:
                # The case that must not be silent. No section 6 guard is
                # reached, the run keeps whatever uniform page it would have
                # had, and the operator is TOLD once. The marker below is
                # deliberately not a substring of the engagement record the limb
                # logs, so a differential that counts engagements cannot be
                # inflated here.
                #
                # THE PAGE IS READ, NOT ASSUMED. This sentence used to name the
                # literal 32 unconditionally, which is wrong whenever the
                # operator passed a block size of their own: that latches
                # user_specified_block_size, update_block_size_for_backend
                # returns without touching the page, and the run allocates the
                # operator's value. An operator sizing KV memory was being told
                # the wrong number by the one line that went out of its way to
                # name it. Both the page and where it came from are read here.
                uniform_page = cls.resolved_uniform_page(vllm_config)
                page_origin = (
                    "the block size supplied on the command line"
                    if vllm_config.cache_config.user_specified_block_size
                    else "the default update_block_size_for_backend sets"
                )
                logger.warning(
                    "Hybrid KDA/DSA KV cache left OFF for "
                    "Glm5NextForConditionalGeneration: the registered block "
                    "size is valid only at tensor_parallel_size=%d and this "
                    "run resolved tensor_parallel_size=%d, so the KV cache "
                    "keeps the uniform %d-token page -- %s. To enable it at "
                    "this degree, re-derive the block size for that degree "
                    "(approvals/DECISIONS.md section 6) and set "
                    "enable_hybrid_kv_cache explicitly.",
                    HYBRID_KV_REGISTERED_TP_DEGREE,
                    resolved_tp_size,
                    uniform_page,
                    page_origin,
                )

        # ---- Hybrid KDA/DSA KV-cache block size -----------------------------
        # Resolved here, before _validate_dcp_config reads cache_config.
        # block_size for its ownership stride, so a hybrid run validates DCP
        # against the block size it will actually allocate with.
        #
        # The VALUE is a frozen user decision -- approvals/DECISIONS.md section
        # 6, verbatim "128 (Recommended)" -- and is valid ONLY under two
        # registered preconditions: tensor-parallel degree 64 and a bfloat16 KV
        # cache. Both are READ here; neither is chosen here, and a seat that
        # needs a different value re-derives it rather than editing this
        # literal. The DSA indexer's own cache is a separate page-size question
        # this value does not answer.
        #
        # Every guard below RAISES rather than warns. On Neuron nothing
        # downstream catches an under-sized page: update_block_size_for_backend
        # hard-sets 32 and never calls the vendor's
        # Platform._align_hybrid_block_size, so a warning would leave the run
        # proceeding on a page too small for the KDA recurrent state and
        # surface as corrupt output far from its cause.
        if neuron_config.get("enable_hybrid_kv_cache", False):
            cache_config = vllm_config.cache_config

            # hybrid_kv_block_size documents itself as an OVERRIDE, so an
            # operator value is HONOURED -- but only when it satisfies both
            # constraints section 6's registered derivation names: the KDA
            # state-page floor expressed in tokens, and the DSA indexer's
            # kernel-granularity multiple. Both values are READ from section 6
            # and neither is re-derived here; the decided 128 is section 6's
            # own result and satisfies both. An unset knob resolves to that
            # decided value.
            #
            # `is None` rather than falsiness, deliberately: an operator 0 is
            # an offending value this path must report, never a value that
            # silently becomes the default.
            HYBRID_BLOCK_SIZE_FLOOR_TOKENS = 67  # DECISIONS.md section 6
            HYBRID_BLOCK_SIZE_GRANULARITY = 64  # DECISIONS.md section 6
            operator_block_size = neuron_config.get("hybrid_kv_block_size")
            if operator_block_size is None:
                hybrid_block_size = 128
            else:
                hybrid_block_size = operator_block_size
                # -- BEGIN section-6 constraint validation --
                if hybrid_block_size < HYBRID_BLOCK_SIZE_FLOOR_TOKENS:
                    raise ValueError(
                        f"The operator-supplied hybrid KDA/DSA KV-cache block "
                        f"size {hybrid_block_size} is below the registered KDA "
                        f"state-page floor of "
                        f"{HYBRID_BLOCK_SIZE_FLOOR_TOKENS} tokens "
                        f"(approvals/DECISIONS.md section 6). A shorter page "
                        f"cannot hold the KDA recurrent state, and nothing "
                        f"downstream on Neuron catches an under-sized page. "
                        f"Supply a value at or above the floor, or unset "
                        f"hybrid_kv_block_size to use the decided value."
                    )
                if hybrid_block_size % HYBRID_BLOCK_SIZE_GRANULARITY:
                    raise ValueError(
                        f"The operator-supplied hybrid KDA/DSA KV-cache block "
                        f"size {hybrid_block_size} is not a multiple of the "
                        f"registered DSA indexer kernel granularity "
                        f"{HYBRID_BLOCK_SIZE_GRANULARITY} "
                        f"(approvals/DECISIONS.md section 6). Supply a "
                        f"multiple of that granularity, or unset "
                        f"hybrid_kv_block_size to use the decided value."
                    )
                # -- END section-6 constraint validation --

            tp_size = vllm_config.parallel_config.tensor_parallel_size
            if tp_size != 64:
                raise ValueError(
                    f"The hybrid KDA/DSA KV-cache block size "
                    f"{hybrid_block_size} is registered ONLY at "
                    f"tensor_parallel_size=64; got tensor_parallel_size="
                    f"{tp_size}. The value is derived from the per-rank KDA "
                    f"recurrent-state page at TP=64, so another TP degree "
                    f"invalidates it. Re-derive the block size for this TP "
                    f"degree before enabling enable_hybrid_kv_cache."
                )

            # Resolve the KV cache dtype exactly as vLLM's own hybrid
            # alignment resolves it (vllm/platforms/interface.py): "auto"
            # follows the model dtype, anything else maps through
            # STR_DTYPE_TO_TORCH_DTYPE. An unmapped spelling resolves to None
            # and fails the guard rather than passing unexamined.
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

            if cache_config.cache_dtype == "auto":
                kv_cache_dtype = model_config.dtype
            else:
                kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE.get(cache_config.cache_dtype)
            if kv_cache_dtype is not torch.bfloat16:
                raise ValueError(
                    f"The hybrid KDA/DSA KV-cache block size "
                    f"{hybrid_block_size} is registered ONLY for a bfloat16 KV "
                    f"cache; the resolved KV cache dtype is {kv_cache_dtype} "
                    f"(cache_dtype={cache_config.cache_dtype!r}). The value is "
                    f"derived from the bf16 per-token page, so electing "
                    f"another KV dtype invalidates it and it must be "
                    f"re-derived rather than reused."
                )

            # Both preconditions hold, so publish the block size. Nothing here
            # sets cache_config.mamba_page_size_padded and nothing calls
            # Platform._align_hybrid_block_size: the KV-cache specs report
            # their pages from shapes and dtypes, and this path leaves that
            # intact.
            cache_config.block_size = hybrid_block_size
            # update_block_size_for_backend runs LATER (from the executor, not
            # from VllmConfig.__post_init__) and hard-sets 32 unless this latch
            # is set, which would silently undo the line above before a single
            # block is allocated. Setting the latch here is what makes the
            # resolved value reach the allocator; that method itself is
            # unchanged.
            cache_config.user_specified_block_size = True
            logger.info(
                "Hybrid KDA/DSA KV cache enabled: block_size=%d "
                "(tensor_parallel_size=%d, kv_cache_dtype=%s)",
                hybrid_block_size,
                tp_size,
                kv_cache_dtype,
            )

        # Component DP on dense models needs the MoE/DP engine path to
        # preserve data_parallel_size across engine core subprocesses.
        if cls._has_neuron_component_dp(vllm_config):
            cls._validate_component_dp_config(vllm_config)
        cls._maybe_enable_moe_for_component_dp(vllm_config)

        cls._validate_dcp_config(vllm_config)
        cls._auto_set_neuron_connector_module_path(vllm_config)
        cls._auto_set_neuron_ec_connector_module_path(vllm_config)
        cls._validate_quantization_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        # TODO: Implement a CPU fallback based on the value from DeviceConfig
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm_neuron.vllm.worker.neuron_worker.NeuronWorker"
            )
            # block_size is set in `update_block_size_for_backend`.
            # TODO: Move all Neuron specific config overrides into a separate function
            # TODO: Can vLLM to vLLM Neuron config mapping be done here after the overrides are applied?

        allowed_backends = ("mp", "uni")
        if parallel_config.distributed_executor_backend not in allowed_backends:
            # multi-process executor is now the default executor in vLLM. But want
            # to explicitly set it here for clarity
            parallel_config.distributed_executor_backend = "mp"

        logger.warning(
            f"Neuron Platform only supports the following distributed executor backends: {allowed_backends}, using {parallel_config.distributed_executor_backend}"
        )

        # Set custom scheduler for Neuron platform
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is enabled on Neuron platform. Currently Neuron only supports"
                " chunking prefills with batch size of 1. Mixing prefill and decode in the same"
                " batch is not supported"
            )

        # Note: we do NOT set disable_chunked_mm_input=True here because
        # upstream vLLM enforces max_tokens_per_mm_item <= max_num_batched_tokens
        # when that flag is set — which fails for models like Qwen3-VL where a
        # single image can theoretically produce more tokens than our
        # max_num_batched_tokens. The encoder budget cap in NeuronScheduler
        # already prevents overflow without needing this flag.

        if hasattr(model_config.hf_config, "vision_config"):
            cls._resolve_vision_auto_config(vllm_config, model_config)

        # Compute per-image embed limit for request validation.
        # Buckets are in raw-token space; mm_pos.length is in embed space.
        # Convert once here: max_embeds = max_bucket / merge_factor.
        vision_config_dict = vllm_config.additional_config.get("vision_neuron_config")
        if vision_config_dict:
            buckets = vision_config_dict.get("num_vision_tokens_buckets")
            if buckets:
                from vllm_neuron.utils.vision_utils import (
                    get_vision_token_merge_factor,
                )

                merge_factor = get_vision_token_merge_factor(model_config.hf_config)
                cls._max_embeds_per_image = max(buckets) // max(merge_factor, 1)

        if envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
            # Capture copies a forward's inputs to host via a standalone op run
            # just before a plain execute. Under async scheduling the input
            # buffers may still hold, or be overwritten by, another in-flight
            # forward, so the op could serialize the wrong tensors — require
            # synchronous scheduling.
            if scheduler_config.async_scheduling:
                raise ValueError(
                    "Snapshot capture (VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE) is "
                    "incompatible with async scheduling. Disable async "
                    "scheduling (async_scheduling=False or "
                    "--no-async-scheduling) to capture snapshots."
                )
            # Resolve (and cache) the snapshot config now so a malformed
            # selection or FORMAT fails at startup, not on the first forward.
            from vllm_neuron.snapshot.config import get_snapshot_config

            get_snapshot_config()

        # Override the default vLLM scheduler with our custom Neuron scheduler
        # We need to override even if it's already set to the default scheduler
        if scheduler_config.scheduler_cls is None or scheduler_config.scheduler_cls in (
            "vllm.v1.core.sched.scheduler.Scheduler",
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        ):
            if scheduler_config.async_scheduling:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronAsyncScheduler"
                )
            else:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronScheduler"
                )
        else:
            logger.warning(
                "scheduler_cls already set to non-default: %s, "
                "NOT overriding with custom Neuron scheduler",
                scheduler_config.scheduler_cls,
            )

    @classmethod
    def validate_request(
        cls,
        processed_inputs: "EngineInput",
        params: "SamplingParams | PoolingParams",
    ) -> None:
        """Reject requests unsupported by the current Neuron server config.

        Called per-request before scheduling. Raises ValueError before model
        execution so SO-off servers do not enter the SO mask path.
        """
        if (
            not cls._enable_structured_outputs
            and getattr(params, "structured_outputs", None) is not None
        ):
            raise ValueError(SO_DISABLED_MESSAGE)

        if cls._max_embeds_per_image is None:
            return

        mm_placeholders = processed_inputs.get("mm_placeholders")
        if not mm_placeholders:
            return

        for modality, positions in mm_placeholders.items():
            for mm_pos in positions:
                # Use get_num_embeds(), not length: a video placeholder span
                # interleaves timestamp TEXT tokens with the vision embeds
                # (is_embed mask), so length overcounts what the vision
                # encoder actually produces. For images they are equal.
                num_embeds = mm_pos.get_num_embeds()
                if num_embeds > cls._max_embeds_per_image:
                    raise ValueError(
                        f"A {modality} in this request produces "
                        f"{num_embeds} embedding tokens, which exceeds "
                        f"the maximum supported by the compiled vision "
                        f"encoder ({cls._max_embeds_per_image}). Reduce the "
                        f"image resolution (e.g. set max_pixels in "
                        f"mm_processor_kwargs) or increase "
                        f"num_vision_tokens_buckets in vision_neuron_config."
                    )

    @classmethod
    def _validate_quantization_config(cls, vllm_config) -> None:
        """Validate quantization config. Only KV cache quantization
        (q_scale/k_scale/v_scale) is supported for compressed-tensors, except on
        Neuron CPU-dequant paths (e.g. quantization="mxfp8") where the device
        never sees on-device weight/activation quant.

        Two checkpoint-format branches sit between the CPU-dequant waiver and
        the compressed-tensors guard, and both are keyed on the CHECKPOINT's
        ``quant_method`` rather than on operator intent:

        * **Block-scaled fp8 is admitted explicitly.** A ``quant_method="fp8"``
          checkpoint carrying ``weight_block_size`` is what this platform's
          block-fp8 path consumes, so the admission is logged. Without the log
          the config is still accepted -- by falling through the
          compressed-tensors guard without ever being examined -- and there is
          no observable difference between "recognised and admitted" and "not
          recognised at all".
        * **MX quantisation methods are refused.** The compile target is trn2
          (gen3), whose matmul substrate carries no MX path, so an MX
          checkpoint has to fail at config time with a readable message rather
          than reach a kernel. The refusal runs AFTER the CPU-dequant waiver by
          design: the waiver is keyed on ``neuron_config["quantization"]``
          (operator intent), so the pin's CPU-dequant path keeps its behaviour
          and this method refuses no configuration the pin accepts.
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        if neuron_config.get("quantization") in cls._cpu_dequant_quantizations:
            return
        model_config = vllm_config.model_config
        quant_cfg = getattr(model_config.hf_config, "quantization_config", None)
        if not quant_cfg:
            return
        quant_method = quant_cfg.get("quant_method")
        # Substring, not prefix: the MX family is spelled both ways in vLLM's
        # own registry ("mxfp8" and "modelopt_mxfp8"), so a prefix test would
        # admit half of it. No method this platform allowlists contains "mx".
        if isinstance(quant_method, str) and "mx" in quant_method.lower():
            raise ValueError(
                f"Neuron does not support MX quantization: quant_method="
                f"{quant_method!r}. The compile target is trn2 (gen3), whose "
                f"matmul substrate has no MX path. Use one of "
                f"{cls.supported_quantization}, or select a Neuron CPU-dequant "
                f"path with additional_config['neuron_config']['quantization']."
            )
        if quant_method == "fp8":
            weight_block_size = quant_cfg.get("weight_block_size")
            if weight_block_size:
                logger.info(
                    "Admitting block-scaled fp8 checkpoint: "
                    "weight_block_size=%s, activation_scheme=%s",
                    weight_block_size,
                    quant_cfg.get("activation_scheme"),
                )
            return
        if quant_method != "compressed-tensors":
            return
        for group_name, group in quant_cfg.get("config_groups", {}).items():
            if group.get("weights"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Weight quantization "
                    f"in config_groups['{group_name}'] is not supported."
                )
            if group.get("output_activations"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Output activation "
                    f"quantization in config_groups['{group_name}'] is not "
                    f"supported."
                )
            targets = group.get("targets", [])
            if group.get("input_activations") and not any(
                "Attention" in t for t in targets
            ):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Input activation "
                    f"quantization on non-attention targets in "
                    f"config_groups['{group_name}'] is not supported."
                )

    @staticmethod
    def _has_neuron_component_dp(vllm_config) -> bool:
        """Check if any Neuron component DP size > 1 in additional_config."""
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        return (
            neuron_config.get("attention_dp_size", 1) > 1
            or neuron_config.get("embedding_dp_size", 1) > 1
            or neuron_config.get("mlp_dp_size", 1) > 1
            or neuron_config.get("lm_head_dp_size", 1) > 1
        )

    @classmethod
    def _maybe_enable_moe_for_component_dp(cls, vllm_config) -> None:
        """Mark model as MoE when component DP needs cross-DP process groups.

        vLLM's run_engine_core only keeps data_parallel_size > 1 for MoE models.
        Component DP creates cross-DP process groups that require all DP workers
        in the same torch distributed world. Setting num_experts=1 causes
        run_engine_core to take the DPEngineCoreProc path.
        """
        if (
            vllm_config.parallel_config.data_parallel_size > 1
            and not vllm_config.model_config.is_moe
            and cls._has_neuron_component_dp(vllm_config)
        ):
            vllm_config.model_config.model_arch_config.num_experts = 1
            logger.info(
                "Component DP enabled — setting num_experts=1 to preserve "
                "data_parallel_size for cross-DP process groups"
            )

    @classmethod
    def _validate_component_dp_config(cls, vllm_config) -> None:
        """Validate component DP (attention/embedding/mlp/lm_head DP) constraints.

        Component DP is decode-only and requires disaggregated inference (DI)
        with data_parallel_size > 1. Single-node non-DI serving is not supported.
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        dp_configs = {
            "attention_dp_size": neuron_config.get("attention_dp_size", 1),
            "embedding_dp_size": neuron_config.get("embedding_dp_size", 1),
            "mlp_dp_size": neuron_config.get("mlp_dp_size", 1),
            "lm_head_dp_size": neuron_config.get("lm_head_dp_size", 1),
        }
        active = {k: v for k, v in dp_configs.items() if v > 1}
        if not active:
            return

        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            raise ValueError(
                f"Component DP ({active}) requires disaggregated inference "
                f"(--kv-transfer-config). Single-node serving without DI is "
                f"not supported with component DP."
            )

        kv_role = kv_config.kv_role
        if kv_role not in ("kv_consumer", "kv_both"):
            raise ValueError(
                f"Component DP ({active}) is only supported on the decode "
                f"server (kv_role='kv_consumer' or 'kv_both'), got "
                f"kv_role='{kv_role}'. The prefill server should not set "
                f"component DP sizes > 1."
            )

        if vllm_config.parallel_config.data_parallel_size < 2:
            raise ValueError(
                f"Component DP ({active}) requires --data-parallel-size >= 2. "
                f"Component DP shards modules across DP replicas and needs "
                f"multiple DP ranks to function."
            )

    @classmethod
    def _validate_dcp_config(cls, vllm_config) -> None:
        """Validate all constraints for decode_context_parallel_size > 1."""
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            return

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        num_q_heads = vllm_config.model_config.hf_config.num_attention_heads

        block_size = vllm_config.cache_config.block_size
        long_prefill_token_threshold = (
            vllm_config.scheduler_config.long_prefill_token_threshold
        )
        ownership_stride = dcp_size * block_size
        if (
            long_prefill_token_threshold > 0
            and long_prefill_token_threshold % ownership_stride != 0
        ):
            raise ValueError(
                "DCP requires long_prefill_token_threshold to be divisible by "
                "decode_context_parallel_size * block_size "
                f"({dcp_size} * {block_size} = {ownership_stride}); got "
                f"{long_prefill_token_threshold}."
            )

        # DI is optional: prefill and decode share one layout/semantics, so DCP
        # runs single-node. If a connector IS set (P/D split) it must be Nixl.
        kv_config = vllm_config.kv_transfer_config
        if kv_config is not None and kv_config.kv_connector != "NeuronNixlConnector":
            raise ValueError(
                "DCP requires kv_connector='NeuronNixlConnector'. "
                'Set "kv_connector": "NeuronNixlConnector" in --kv-transfer-config'
            )

        # Each KV head must be replicated across a complete DCP group, and its
        # GQA query-head group must split evenly across those ranks.
        if tp_size < dcp_size * num_kv_heads:
            raise ValueError(
                "DCP requires tensor_parallel_size >= "
                "decode_context_parallel_size * num_kv_heads "
                f"(tp={tp_size}, dcp={dcp_size}, num_kv_heads={num_kv_heads}; "
                f"need tp >= {dcp_size * num_kv_heads}). KV must be replicated "
                "across the DCP group so each group member shares a KV head."
            )

        gqa_ratio = num_q_heads // num_kv_heads
        if gqa_ratio % dcp_size != 0:
            raise ValueError(
                "DCP requires (num_q_heads // num_kv_heads) % dcp == 0 "
                f"(gqa_ratio={gqa_ratio}, dcp={dcp_size})."
            )

        # DCP groups cannot cover every TP rank unless the sizes divide evenly.
        if tp_size % dcp_size != 0:
            raise ValueError(
                f"DCP requires tp % decode_context_parallel_size == 0 so the replica "
                f"mesh covers every rank (tp={tp_size}, dcp={dcp_size})"
            )

        if tp_size > num_q_heads:
            logger.warning(
                "tensor_parallel_size (%d) > num_attention_heads (%d). "
                "Multiple ranks will share the same Q heads, wasting attention "
                "computation FLOPs. This may still be useful to shard the KV "
                "sequence across more DCP ranks for long-context workloads.",
                tp_size,
                num_q_heads,
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        import torch

        torch.manual_seed(seed)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_neuron.parallel.neuron_communicator.NeuronDeviceCommunicator"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        logger.info("Using NEURON custom attention backend.")
        return "vllm_neuron.vllm.attention.attn.NeuronAttentionBackend"

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        # "cuda" here means device. Neuron DI uses device-to-device
        # direct transfer.
        return {NeuronPlatform.device_type: {"cuda"}}

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """
        Whether the platform supports hybrid KV cache with multiple cache groups.

        When True, models with mixed attention types (e.g., full attention and
        sliding window attention) will use separate KV cache groups, enabling
        memory optimization by freeing blocks outside the sliding window.

        Returns:
            True to enable hybrid KV cache support on Neuron.
        """
        return True

    @classmethod
    def device_count(cls) -> int:
        """
        Gets the visible device count. This returns the configured device count
        if set, otherwise falls back to the number of visible devices returned
        by the torch runtime.
        """
        if cls._device_count != -1:
            return cls._device_count

        runtime = torch.classes.neuron.Runtime()
        nc_count = runtime.get_nc_count()
        if nc_count == -1:
            raise RuntimeError(
                "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"
            )
        return nc_count

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def set_device_count(cls, device_count: int) -> None:
        """
        Sets the visible device count.

        In the new vLLM Neuron backend, each worker thread has a single visible device,
        so the torch runtime can only see one device. This is a problem because
        vLLM expects that the device count returns the number of devices visible
        to a node, such as when initializing MultiprocExecutor for a multi-node
        server. To fix that issue, each worker thread sets the device count to
        the number of visible devices on the local node.
        """
        cls._device_count = device_count

    @classmethod
    def _auto_set_neuron_connector_module_path(cls, vllm_config) -> None:
        """Auto-inject kv_connector_module_path for NeuronNixlConnector.

        The factory needs module_path to locate the class in worker subprocesses.
        We set it here so users only need to specify kv_connector name.
        """
        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            return
        if (
            kv_config.kv_connector == "NeuronNixlConnector"
            and not kv_config.kv_connector_module_path
        ):
            kv_config.kv_connector_module_path = (
                "vllm_neuron.vllm.kv_connector.neuron_nixl_connector"
            )
        if (
            kv_config.kv_connector == "NeuronDecodeBenchConnector"
            and not kv_config.kv_connector_module_path
        ):
            kv_config.kv_connector_module_path = (
                "vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector"
            )

    @classmethod
    def _auto_set_neuron_ec_connector_module_path(cls, vllm_config) -> None:
        """Auto-inject ec_connector_module_path for NeuronNixlECConnector.

        The EC connector factory imports the class by module path in worker
        subprocesses (it is not in the built-in registry), so we set it here when
        the user selects the connector by name. No-op when no EC transfer is
        configured (the default monolith path).
        """
        ec_config = getattr(vllm_config, "ec_transfer_config", None)
        if ec_config is None:
            return
        if (
            ec_config.ec_connector == "NeuronNixlECConnector"
            and not ec_config.ec_connector_module_path
        ):
            ec_config.ec_connector_module_path = (
                "vllm_neuron.vllm.ec_connector.neuron_nixl_ec_connector"
            )

    @classmethod
    def _patch_termination_timeouts(cls) -> None:
        """Patch vLLM process termination to use a configurable timeout.

        Upstream vLLM has two hardcoded SIGTERM-to-SIGKILL timeout paths:

        1. ``v1/utils.py:shutdown()`` (5 s) — Main process kills the
           EngineCore process tree via ``kill_process_tree()``.
        2. ``MultiprocExecutor._ensure_worker_termination()`` (4 s) —
           EngineCore kills individual worker processes.

        When Neuron profiling is enabled (``NEURON_RT_INSPECT_ENABLE=1``),
        the runtime needs extra time after SIGTERM to flush profiling data
        to disk.  This method monkey-patches both paths so the timeout is
        controlled by ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT`` (default
        5 s, matching upstream).

        For path (1), ``v1/engine/utils.py`` does
        ``from vllm.v1.utils import shutdown`` which creates a module-level
        binding later captured by ``weakref.finalize``.  We must replace
        the name in **both** modules so the finalizer gets our version.
        """
        if cls._termination_timeout_patched:
            return
        cls._termination_timeout_patched = True

        timeout = envs.VLLM_NEURON_WORKER_TERMINATION_TIMEOUT

        if timeout == 5 and os.getenv("NEURON_RT_INSPECT_ENABLE") == "1":
            logger.warning(
                "Neuron profiling is enabled but VLLM_NEURON_WORKER_TERMINATION_TIMEOUT "
                "is at default (5s). Profile data may be incomplete. "
                "Consider setting VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60."
            )
            return

        if timeout == 5:
            return

        logger.info(
            "Patching vLLM termination timeouts to %ss (pid=%s)", timeout, os.getpid()
        )
        cls._patch_shutdown(timeout)
        cls._patch_ensure_worker_termination(timeout)

    @classmethod
    def _patch_shutdown(cls, neuron_timeout: int) -> None:
        """Patch ``v1.utils.shutdown`` (5 s → *timeout*)."""
        import vllm.v1.engine.utils as engine_utils
        import vllm.v1.utils as v1_utils
        from vllm.utils.system_utils import kill_process_tree

        def _shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
            logger.info(
                "patched shutdown: pid=%s, timeout=%ss, %s proc(s)",
                os.getpid(),
                timeout,
                len(procs),
            )
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            timeout = max(neuron_timeout, timeout) if timeout else neuron_timeout
            deadline = _time.monotonic() + timeout
            for proc in procs:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if proc.is_alive():
                    proc.join(remaining)

            for proc in procs:
                if proc.is_alive() and (pid := proc.pid) is not None:
                    logger.warning(
                        "%s (pid=%s) still alive after %ss — sending SIGKILL",
                        proc.name,
                        pid,
                        timeout,
                    )
                    kill_process_tree(pid)

        v1_utils.shutdown = _shutdown
        engine_utils.shutdown = _shutdown

    @classmethod
    def _patch_ensure_worker_termination(cls, timeout: int) -> None:
        """Patch ``MultiprocExecutor._ensure_worker_termination`` (4 s → *timeout*)."""
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        @staticmethod  # type: ignore[misc]
        def _ensure_worker_termination(worker_procs: list[BaseProcess]) -> None:
            active = [p for p in worker_procs if p.is_alive()]
            logger.info(
                "patched _ensure_worker_termination: pid=%s, timeout=%ss, %s worker(s)",
                os.getpid(),
                timeout,
                len(active),
            )
            for p in active:
                p.terminate()

            deadline = _time.monotonic() + timeout
            for p in active:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if p.is_alive():
                    p.join(remaining)

            still_alive = [p for p in active if p.is_alive()]
            if still_alive:
                for p in still_alive:
                    logger.warning(
                        "Worker %s (pid=%s) still alive after %ss — sending SIGKILL",
                        p.name,
                        p.pid,
                        timeout,
                    )
                    p.kill()

        MultiprocExecutor._ensure_worker_termination = _ensure_worker_termination

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        """Map logical device ID to physical Neuron core ID."""
        if cls.device_control_env_var in os.environ:
            # Parse NEURON_VISIBLE_DEVICES range (e.g., "0-7")
            core_range = os.environ[cls.device_control_env_var]
            if "-" in core_range:
                start, end = map(int, core_range.split("-"))
                available_cores = list(range(start, end + 1))
            else:
                # Single core or comma-separated list
                available_cores = [int(c.strip()) for c in core_range.split(",")]
            logger.debug(
                "device id: %s, physical device id: %s",
                device_id,
                available_cores[device_id],
            )
            return available_cores[device_id]
        return device_id
