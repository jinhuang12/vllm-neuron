# SPDX-License-Identifier: Apache-2.0
"""
Centralized patch management for vllm-neuron.

All monkey-patches to upstream vLLM are applied here via apply_patches().
Called from NeuronPlatform.check_and_update_config() after vLLM is fully
initialized.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_patches():
    """Apply all vllm-neuron patches to upstream vLLM."""
