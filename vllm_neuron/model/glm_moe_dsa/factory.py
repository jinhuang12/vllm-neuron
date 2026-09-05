# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 factory and frozen TP=64 contract validation."""

import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class GlmMoeDsaForCausalLM(nn.Module):
    """Registry factory for the pinned GLM-5.2 Neuron implementation."""

    SUPPORTED_TENSOR_PARALLEL_SIZE = 64

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        tensor_parallel_size: int | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, neuron_config, tensor_parallel_size
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def get_kv_spec(self):
        return self._model.get_kv_spec()

    def bind_kv_cache(self, kv_caches):
        return self._model.bind_kv_cache(kv_caches)

    def load_weights(self, checkpoint_path, device, cache_dir=None):
        return self._model.load_weights(checkpoint_path, device, cache_dir)

    def load_weights_lite(self, checkpoint_path, device, cache_dir=None):
        return self._model.load_weights_lite(checkpoint_path, device, cache_dir)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        tensor_parallel_size: int | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, neuron_config, tensor_parallel_size
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        tensor_parallel_size: int | None,
    ) -> nn.Module:
        resolved_tp = cls._resolve_tensor_parallel_size(tensor_parallel_size)
        cls._validate_tensor_parallel_size(resolved_tp)

        from .model import GlmMoeDsaForCausalLM as Model

        return Model.from_configs(
            hf_config,
            neuron_config,
            tensor_parallel_size=resolved_tp,
            expert_parallel_rank=cls._resolve_tensor_parallel_rank(),
        )

    @staticmethod
    def _resolve_tensor_parallel_rank() -> int:
        if not dist.is_initialized():
            return 0

        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    @classmethod
    def _resolve_tensor_parallel_size(cls, tensor_parallel_size: int | None) -> int:
        if tensor_parallel_size is not None:
            return tensor_parallel_size

        from vllm.distributed import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size())

    @classmethod
    def _validate_tensor_parallel_size(cls, tensor_parallel_size: int) -> None:
        if tensor_parallel_size != cls.SUPPORTED_TENSOR_PARALLEL_SIZE:
            raise ValueError(
                "GLM-5.2 supports tensor_parallel_size=64; "
                f"got {tensor_parallel_size!r}"
            )
