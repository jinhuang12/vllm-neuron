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

#: Partitions of one SBUF tile: the partition-axis bound, ``nl.tile_size.pmax`` inside a kernel.
#: Host-side tile arithmetic reads this one number.
SBUF_PARTITIONS = 128
#: SBUF bytes per partition on trn2, before the runtime's reservations.
SBUF_TOTAL_BYTES_PER_PARTITION = 224 * 1024
#: The dynamic-DMA scratch region nkilib names ``DynamicDMAScratchLoc``.
SBUF_DYNAMIC_DMA_SCRATCH_BYTES = 16384
#: The reserved region nkilib names ``EvalAccelReservedLoc``.
SBUF_EVAL_ACCEL_RESERVED_BYTES = 8
#: The last reservation. nkilib gives it as 256 B, the transpose's identity tensor, in
#: ``core/mlp/mlp_cte/mlp_cte_constants.py:62-63`` and as 520 B in
#: ``experimental/moe/bwd/bwmm_bwd_dropless.py:36-37`` (which describes its reservations as
#: DMA header, metadata and alignment overhead). This takes 520, the larger, so a tile sized
#: here fits under either.
SBUF_TAIL_RESERVED_BYTES = 520
#: SBUF bytes one partition offers an NKI kernel on trn2: nkilib's ``MAX_AVAILABLE_SBUF_SIZE``
#: (``bwmm_bwd_dropless.py:37``). Host-side tile-size rules use it, because they run before a
#: kernel is traced; inside a kernel ``nl.tile_size`` is authoritative.
SBUF_BYTES_PER_PARTITION = (SBUF_TOTAL_BYTES_PER_PARTITION - SBUF_DYNAMIC_DMA_SCRATCH_BYTES
                            - SBUF_EVAL_ACCEL_RESERVED_BYTES - SBUF_TAIL_RESERVED_BYTES)


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
