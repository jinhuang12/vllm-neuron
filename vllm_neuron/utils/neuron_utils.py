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
    """True when a VALUE can be read off ``tensor`` on the host.

    A precondition that reads data cannot run while a graph is built, and three
    builders reach this code. Dynamo traces the caller's OWN types, so under it a
    tensor is neither a ``FakeTensor`` instance nor on ``meta`` and the two tensor
    clauses below both pass: a serving run read a gate value that way, and torch
    then refused to guard the unbacked symbol the read had made. A tracing context
    is what answers there. The fake-tensor pass and the ``meta`` graph build are the
    other two, and each keeps its own clause because a tracing context is not open
    for either. Shape and dtype reads stay legitimate under all three and do not
    belong here.
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
