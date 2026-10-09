# SPDX-License-Identifier: Apache-2.0
"""Defaults and resolution of the two test-artefact knobs, with the standard library only.

``vllm_neuron/envs.py`` declares ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR`` and
``VLLM_NEURON_GLM5NEXT_CAMPAIGN_DIR`` and resolves them with the functions here.
``test/vllm_neuron/artifacts.py`` loads this file without running
``vllm_neuron/__init__.py`` (which imports torch and vLLM), so tools under ``test/``
that start without them, such as the kernel ledger, read the same values. Keep this
module free of imports outside the standard library. The server reads neither knob.
"""

from __future__ import annotations

import os

#: Names the served GLM-5.3-Flash checkpoint directory (it holds
#: ``model.safetensors.index.json``).
CHECKPOINT_KNOB = "VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR"
#: The serving host's campaign checkpoint.
CHECKPOINT_DEFAULT = "/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9"

#: Names the directory the performance campaign keeps its worktrees and records under.
CAMPAIGN_KNOB = "VLLM_NEURON_GLM5NEXT_CAMPAIGN_DIR"
#: The campaign hosts' directory (``glm53f-wt*``, ``glm53f-decode-breakdown-*`` below it).
CAMPAIGN_DEFAULT = "/home/ubuntu"


def checkpoint_dir() -> str:
    """:data:`CHECKPOINT_KNOB`'s value; :data:`CHECKPOINT_DEFAULT` when unset or empty."""
    return os.getenv(CHECKPOINT_KNOB) or CHECKPOINT_DEFAULT


def campaign_dir() -> str:
    """:data:`CAMPAIGN_KNOB`'s value; :data:`CAMPAIGN_DEFAULT` when unset or empty."""
    return os.getenv(CAMPAIGN_KNOB) or CAMPAIGN_DEFAULT
