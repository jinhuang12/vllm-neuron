# SPDX-License-Identifier: Apache-2.0
"""``get_efa_interface`` finds a logical core's Neuron device from ``NEURON_LOGICAL_NC_CONFIG``.

The device index is ``core * setting // PHYSICAL_CORES_PER_DEVICE``, and an unset setting
means the runtime's default grouping, ``_DEFAULT_LNC_CONFIG`` (2). That differs from
:func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair`, where unset means one program:
a kernel asks whether it may use both cores of a pair, but the device map needs the
grouping the runtime applies. So this seam reads ``vllm_neuron.envs`` and not ``lnc_pair``.
The device node is never opened: ``os.stat`` of ``/dev/neuron*`` is refused here, and the
refusal names the path.
"""

import os

import pytest

from vllm_neuron.utils import hardware_config

NC_CONFIG = "NEURON_LOGICAL_NC_CONFIG"
#: trn2: physical NeuronCores on one Neuron device.
PHYSICAL_CORES_PER_DEVICE = 8
#: A logical core whose device differs between groupings 1 and 2.
CORE = 12


def _device_node(monkeypatch) -> str:
    """The ``/dev/neuron*`` path ``get_efa_interface`` asks for at :data:`CORE`."""
    real_stat = os.stat
    asked = []

    def stat(path, *args, **kwargs):
        if str(path).startswith("/dev/neuron"):
            asked.append(str(path))
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hardware_config.os, "stat", stat)
    with pytest.raises(RuntimeError, match="Cannot access /dev/neuron"):
        hardware_config.get_efa_interface(0, [CORE])
    assert len(asked) == 1
    return asked[0]


@pytest.mark.parametrize("setting, grouping", [
    (None, hardware_config._DEFAULT_LNC_CONFIG), ("1", 1), ("2", 2)],
    ids=["unset", "lnc1", "lnc2"])
def test_the_core_maps_to_the_device_of_its_grouping(setting, grouping, monkeypatch):
    if setting is None:
        monkeypatch.delenv(NC_CONFIG, raising=False)
    else:
        monkeypatch.setenv(NC_CONFIG, setting)
    assert _device_node(monkeypatch) == (
        f"/dev/neuron{CORE * grouping // PHYSICAL_CORES_PER_DEVICE}")


def test_a_setting_that_is_not_an_integer_is_refused(monkeypatch):
    monkeypatch.setenv(NC_CONFIG, "abc")
    with pytest.raises(ValueError, match="abc"):
        hardware_config.get_efa_interface(0, [CORE])
