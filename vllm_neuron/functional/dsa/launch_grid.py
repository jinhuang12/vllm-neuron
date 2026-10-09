# SPDX-License-Identifier: Apache-2.0
"""The launch grid of the LNC-aware kernels: one program, or both cores of an LNC2 pair.

The launcher sets ``NEURON_LOGICAL_NC_CONFIG`` for the runtime and the compiler before the
worker starts, so the setting is a property of the process. These kernel seams in
``vllm_neuron/functional`` split their work over an LNC2 pair only when :func:`lnc_pair`
says they may:

- ``dsa``: the decode kernels (``decode_batch``, ``decode_select``, ``decode_trow``),
  ``kpool_hadamard`` and ``score_gemm``;
- ``kda``: ``chunked_recurrence`` (the prefill chunk kernels), ``depthwise_conv1d`` and
  ``fused_decode``;
- ``attention``: ``mla_decode``, ``mla_dense_window`` and ``mla_sparse``;
- ``glue``: ``kda_output``, ``kda_projections`` and ``mhc_pre``;
- ``moe``: ``expert_decode``, ``fused_fp8`` and ``token_gather_combine``;
- ``mtp``: the tail kernels ``tail_in`` and ``tail_out``, through
  ``common.launch_programs``.

They read it once per trace and never per step. An ordinary read inside a
traced function becomes a Dynamo guard, and that guard reads the environment again
(``vllm_neuron.envs.__getattr__``, then ``os.getenv``) before every step.
:func:`lnc_pair` is folded instead (``torch._dynamo.assume_constant_result``): Dynamo calls
it while it traces and keeps the answer as a constant of the graph, with no guard. A graph
keeps the grid of the setting it was traced under; a new setting needs a new trace
(a new process, or ``torch._dynamo.reset()``). Called eagerly, it reads the setting on
every call.
"""

from __future__ import annotations

import torch

from vllm_neuron import envs

#: The settings the LNC-aware kernels serve: one physical core per logical core (unset,
#: or 1) or the LNC2 pair (2), the two trn2 configurations.
SERVED_SETTINGS = (None, 1, 2)


class LaunchGridError(ValueError):
    """``NEURON_LOGICAL_NC_CONFIG`` holds a setting the LNC-aware kernels do not serve."""


@torch._dynamo.assume_constant_result
def lnc_pair() -> bool:
    """Whether the LNC-aware kernels may split their work over an LNC2 pair (setting 2).

    Raises:
        LaunchGridError: ``NEURON_LOGICAL_NC_CONFIG`` is not an integer, or is not one of
            :data:`SERVED_SETTINGS`.
    """
    try:
        setting = envs.NEURON_LOGICAL_NC_CONFIG
    except ValueError as error:
        raise LaunchGridError(
            f"NEURON_LOGICAL_NC_CONFIG must be an integer; {error}") from error
    if setting not in SERVED_SETTINGS:
        raise LaunchGridError(
            f"NEURON_LOGICAL_NC_CONFIG={setting}: the LNC-aware kernels launch one "
            f"program (unset or 1) or two, one per core of an LNC2 pair (2)")
    return setting == 2
