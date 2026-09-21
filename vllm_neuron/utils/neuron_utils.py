# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import torch
from torch._guards import TracingContext
from torch._subclasses.fake_tensor import FakeTensor

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Check if NKI kernels can run on the given device."""
    if envs.VLLM_NEURON_DISABLE_NKI_KERNELS:
        return False
    if envs.VLLM_NEURON_CPU_MODE:
        return os.environ.get("NKI_SIMULATOR") == "1"
    device_str = str(device.device) if isinstance(device, torch.Tensor) else device
    return device_str != "cpu"


def values_are_readable(tensor: torch.Tensor) -> bool:
    """Check if a value can be read off ``tensor`` on the host.

    Guards host-side reads of tensor *data*, which are only valid outside graph
    construction. Each clause covers a different construction path: under Dynamo
    the tensor is a real device tensor, so only the tracing state reveals the
    trace -- and reading a value there creates an unbacked symbol that torch then
    cannot guard on -- while a fake-tensor pass and a ``meta`` graph build open no
    tracing context and are caught by the tensor checks instead. Shape and dtype
    reads stay valid on all three paths and need no guard.
    """
    if torch.compiler.is_compiling() or TracingContext.try_get() is not None:
        return False
    return not isinstance(tensor, FakeTensor) and tensor.device.type != "meta"


def model_forward_context(
    vllm_config: VllmConfig,
) -> contextlib.AbstractContextManager[None]:
    """Context manager for model forward: skips fail_on_recompile in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE and vllm_config.model_config.enforce_eager:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("fail_on_recompile")
