# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative draft proposer for the Kimi K3 drafter."""

from __future__ import annotations

import contextlib
import logging
import time

import torch
import torch.nn as nn
from libtorch_neuronx_lite import envs as libtorch_envs
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadata

from vllm_neuron import envs
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.kimi_k3.config import KimiK3TextConfig
from vllm_neuron.model.kimi_k3.model import aux_layers_for_draft
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.registry import get_models
from vllm_neuron.utils.neuron_utils import model_forward_context
from vllm_neuron.vllm.spec_decode.dflash import DFlashProposer

logger = logging.getLogger(__name__)

# The distributed argmax kernel takes at most this many verification rows.
DSPARK_MAX_VERIFY_ROWS = 256


class DSparkProposer(DFlashProposer):
    """DSpark proposer: the DFlash flow with an anchor-first block and in-order picks.

    Each round writes the verify rows' taps into the draft's context cache, runs one
    causal block pass over the anchor and k - 1 mask tokens, and picks the k drafts in
    order with the draft's Markov head. Warm-up and graph extraction are DFlash's.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ) -> None:
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        if self.method != "dspark":
            raise ValueError(
                f"DSparkProposer requires method='dspark'; got {self.method!r}"
            )

        self.device = device
        self.on_device_sampling = on_device_sampling
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        num_verify_tokens = max_num_seqs * (self.num_speculative_tokens + 1)
        if num_verify_tokens > DSPARK_MAX_VERIFY_ROWS:
            raise ValueError(
                "DSpark supports at most 256 verification tokens; "
                f"max_num_seqs={max_num_seqs} with "
                f"num_speculative_tokens={self.num_speculative_tokens} would verify "
                f"{num_verify_tokens} tokens at once"
            )

        # Every block row is a pick, so k may reach the block size.
        block_size = self.draft_model_config.hf_config.block_size
        if not 1 <= self.num_speculative_tokens <= block_size:
            raise ValueError(
                "DSpark num_speculative_tokens must be 1 to the draft block size "
                f"{block_size}; got {self.num_speculative_tokens}"
            )

        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.capture_backend_model: nn.Module | None = None

    def load_model(
        self,
        target_hidden_size: int | None = None,
        target_hidden_pad_groups: int = 1,
        compile_options: dict | None = None,
    ) -> None:
        """Build, load and compile the draft; its caches follow the target's layers.

        Args:
            target_hidden_size: Hidden size of the target's taps.
            target_hidden_pad_groups: Layout of the target's padded hidden dim; the
                drafter takes the unpadded layout (1) only.
            compile_options: Target model's torch.compile options, so the draft
                graph is built with the same compiler args.
        """
        draft_hf_config = self.draft_model_config.hf_config
        if target_hidden_size != draft_hf_config.target_hidden_size:
            raise ValueError(
                f"the draft reads taps of width {draft_hf_config.target_hidden_size}; "
                f"the target's hidden size is {target_hidden_size}"
            )
        if target_hidden_pad_groups != 1:
            raise NotImplementedError(
                f"the target deinterleaves its hidden dim into {target_hidden_pad_groups} "
                "groups, which the DSpark draft does not read"
            )

        model_cls = dict(get_models())[self.draft_model_config.architecture]
        target_num_layers = self.vllm_config.model_config.hf_text_config.num_hidden_layers
        with torch.device("meta"):
            model = model_cls.from_configs(
                config=draft_hf_config,
                start_layer_idx=target_num_layers,
                neuron_config=NeuronConfig(),
                max_positions=self.vllm_config.model_config.max_model_len
                + self.num_speculative_tokens,
            )
        model.num_speculative_tokens = self.num_speculative_tokens
        model.mask_token_id = draft_hf_config.mask_token_id

        load_start = time.perf_counter()
        if not envs.VLLM_NEURON_CPU_COMPILE:
            model.load_weights(
                self.speculative_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )
            model = model.to(self.device)
            MODEL_LOAD_TIME.labels(model_name=self.speculative_config.model).set(
                time.perf_counter() - load_start
            )
            MODEL_LOAD_SIZE.labels(model_name=self.speculative_config.model).set(
                sum(p.nbytes for p in model.parameters())
            )
        else:
            if hasattr(model, "load_weights_lite"):
                model.load_weights_lite(
                    self.speculative_config.model,
                    torch.device("cpu"),
                    self.vllm_config.load_config.download_dir,
                )
                logger.info("Light weight loading complete for DSpark CPU compilation.")
            logger.info("CPU compilation is enabled. Skipping full DSpark weight loading.")
            model = model.to("meta")

        compile_kwargs: dict = {"fullgraph": not envs.VLLM_NEURON_DEBUG_MODE}
        if compile_options is not None:
            compile_kwargs["options"] = compile_options
        if (
            self.vllm_config.model_config.enforce_eager
            or envs.VLLM_NEURON_CPU_MODE
            or envs.VLLM_NEURON_DEBUG_MODE
            or libtorch_envs.NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND
        ):
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                model, backend="neuron_libtorch_graph_capture", **compile_kwargs
            )
        # Kept so the runner can wrap the draft model again after the package flush.
        self.compile_kwargs = {"backend": envs.get_compile_backend_name(), **compile_kwargs}
        self.model = torch.compile(model, **self.compile_kwargs)
        self.attn_layer_names = [layer.cache_name for layer in model.layers]
        logger.info("Completed compilation for DSpark draft model.")

    def get_target_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """The target layers whose input streams the draft reads: its layer ids plus 1."""
        text_config = KimiK3TextConfig.from_hf_config(
            self.vllm_config.model_config.hf_text_config
        )
        return aux_layers_for_draft(self.draft_model_config.hf_config, text_config)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: AttentionMetadata,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
        cur_num_draft_tokens: torch.Tensor | None = None,
        is_warmup: bool = False,
        model_override: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose the drafts for the current target step.

        Takes the arguments of :meth:`DFlashProposer.propose`.

        Returns:
            ``(draft_token_ids, drafts_only)``: the anchor with the drafts
            ``[num_reqs, 1 + k]``, and the drafts ``[num_reqs, k]`` that rejection
            sampling reads.
        """
        assert self.model is not None
        assert attn_metadata, "DSpark requires draft attention metadata"
        model = model_override if model_override is not None else self.model

        batch_size = last_token_indices.shape[0]
        device = target_hidden_states.device
        if prev_sampled_token_ids is None:
            noop = self._build_noop_async_spec_correction_kwargs(
                target_hidden_states.shape[0], batch_size, device=device
            )
            prev_sampled_token_ids = noop["prev_sampled_token_ids"]
            prev_num_draft_tokens = noop["prev_num_draft_tokens"]
            req_indices_per_token = noop["req_indices_per_token"]
        if cur_num_draft_tokens is None:
            cur_num_draft_tokens = torch.full(
                (batch_size,), self.num_speculative_tokens, dtype=torch.int32, device=device
            )

        start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            draft_token_ids, drafts_only = model(
                target_positions=target_positions.to(device),
                target_hidden_states=target_hidden_states,
                last_token_indices=last_token_indices.to(device),
                attn_metadata=attn_metadata,
                raw_sampled_token_ids=raw_sampled_token_ids.to(device),
                prev_sampled_token_ids=prev_sampled_token_ids.to(device),
                prev_num_draft_tokens=prev_num_draft_tokens.to(device),
                req_indices_per_token=req_indices_per_token.to(device),
                cur_num_draft_tokens=cur_num_draft_tokens.to(device),
            )
        bucket_name = f"dspark_block{self.num_speculative_tokens}"
        labels = {"model_name": self.speculative_config.model, "bucket_name": bucket_name}
        if is_warmup:
            COMPILATION_TIME.labels(**labels).set(time.perf_counter() - start)
        else:
            NEFF_EXECUTION_COUNT.labels(**labels).inc()
        return draft_token_ids, drafts_only
