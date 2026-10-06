# SPDX-License-Identifier: Apache-2.0
"""Run this directory's NKI kernels on the CPU simulator.

The root conftest selects CPU mode, where a kernel dispatch refuses unless
``NKI_SIMULATOR=1`` is set (``libtorch_neuronx_lite.nki.nki_hop``). Every test here
asserts that its kernel dispatched, so the simulator is this directory's default; an
explicit ``NKI_SIMULATOR`` in the environment still wins. The flag is read at
dispatch time, so setting it at collection covers every test below.
"""

import os

os.environ.setdefault("NKI_SIMULATOR", "1")
