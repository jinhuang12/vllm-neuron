# SPDX-License-Identifier: Apache-2.0
"""The ``/dev/neuron*`` nodes this process holds open (not a test).

A CPU-only test or script asserts that the list is empty after it runs: the simulator
and the CPU compile path must never open a Neuron device. The NKI CPU simulator does
open every node when ``NEURON_PLATFORM_TARGET_OVERRIDE`` is unset (it asks the runtime
for the platform), which ``test/conftest.py`` prevents for pytest runs only.
"""

from __future__ import annotations

import os

_FD_DIR = "/proc/self/fd"
_DEVICE_PREFIX = "/dev/neuron"


def open_neuron_device_nodes() -> list[str]:
    """The device paths behind this process's open file descriptors (Linux ``/proc``)."""
    held = []
    for handle in os.listdir(_FD_DIR):
        target = os.path.realpath(os.path.join(_FD_DIR, handle))
        if target.startswith(_DEVICE_PREFIX):
            held.append(target)
    return held
