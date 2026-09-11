# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import torch
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

    A precondition that reads data cannot run while a graph is being traced or built:
    Dynamo traces with fake inputs, and a graph built for the device is built on ``meta``,
    where a value does not exist. Both are decided from the tensor rather than from
    ``torch.compiler.is_compiling()``, which ``attention_decode.py:592-597`` records as
    unreliable on this backend. Shape and dtype reads stay legitimate in either case and
    do not belong here.
    """
    return not isinstance(tensor, FakeTensor) and tensor.device.type != "meta"


def model_forward_context(
    vllm_config: VllmConfig,
) -> contextlib.AbstractContextManager[None]:
    """Context manager for model forward: skips fail_on_recompile in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE and vllm_config.model_config.enforce_eager:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("fail_on_recompile")
