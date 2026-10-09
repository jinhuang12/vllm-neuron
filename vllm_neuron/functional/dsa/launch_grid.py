# SPDX-License-Identifier: Apache-2.0
"""The launch grid of the DSA decode kernels: one program, or both cores of an LNC2 pair.

The launcher sets ``NEURON_LOGICAL_NC_CONFIG`` for the runtime and the compiler before the
worker starts, so the setting is a property of the process. The decode kernels read it
through :func:`lnc_pair`, once per trace and never per step. An ordinary read inside a
traced function becomes a Dynamo guard, and that guard reads the environment again
(``vllm_neuron.envs.__getattr__``, then ``os.getenv``) before every decode step.
:func:`lnc_pair` is folded instead (``torch._dynamo.assume_constant_result``): Dynamo calls
it while it traces and keeps the answer as a constant of the graph, with no guard. A graph
keeps the grid of the setting it was traced under; a new setting needs a new trace
(a new process, or ``torch._dynamo.reset()``). Called eagerly, it reads the setting on
every call.
"""

from __future__ import annotations

import torch

from vllm_neuron import envs

#: The settings the DSA decode kernels serve: one physical core per logical core (unset,
#: or 1) or the LNC2 pair (2), the two trn2 configurations.
SERVED_SETTINGS = (None, 1, 2)


class LaunchGridError(ValueError):
    """``NEURON_LOGICAL_NC_CONFIG`` holds a setting the DSA decode kernels do not serve."""


@torch._dynamo.assume_constant_result
def lnc_pair() -> bool:
    """Whether the decode kernels may split their work over an LNC2 pair (setting 2).

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
            f"NEURON_LOGICAL_NC_CONFIG={setting}: the DSA decode kernels launch one "
            f"program (unset or 1) or two, one per core of an LNC2 pair (2)")
    return setting == 2
