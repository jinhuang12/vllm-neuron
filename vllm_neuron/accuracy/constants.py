# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Constants for vLLM Neuron accuracy validation.
"""

# Tolerances tend to be tighter at smaller top_k values because the accuracy of
# more likely tokens is more important than less likely tokens.
DEFAULT_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001

#: The architecture string whose entries are registered below. Named once so the
#: same key is used by every arch-scoped map, here and in ``logit_validation``.
GLM5NEXT_ARCH = "Glm5NextForConditionalGeneration"

# Arch-scoped tolerances, ADDED BESIDE the shared defaults above and never
# overwriting them. The defaults are read by every architecture the plugin
# registers (``logit_validation``, ``logit_visualization`` and the
# ``accuracy_debugger`` plugins all consume them), so retuning them in place
# would silently move every other architecture's comparison.
#
# Tuple order is ``(atol, rtol)`` -- ``DEFAULT_TOLERANCE_MAP``'s own order, and
# the REVERSE of ``vllm_neuron.accuracy.testing._DEFAULT_DTYPE_TOLERANCE``'s
# ``(rtol, atol)``. The two orders are different on purpose; do not normalise
# one to the other.
ARCH_TOLERANCE_MAP = {
    GLM5NEXT_ARCH: {
        "5": (1e-5, 0.033),
        "50": (1e-5, 0.06),
        "1000": (1e-5, 0.09),
        "all": (1e-5, 0.15),
    },
}

# Arch-scoped divergence entries. ``divergence_n_ulps`` is registered
# EXPLICITLY as ``None`` and that ``None`` is load-bearing rather than
# decoration: the fixed difference tolerance is consulted only when the ULP
# count is ``None``, so an entry that omitted the key would make the registered
# ``divergence_difference_tol`` dead code and leave the effective tolerance a
# runtime ULP quantity. Registering the key with ``None`` makes the choice
# visible and checkable; a missing key would not be.
ARCH_DIVERGENCE_CONFIG = {
    GLM5NEXT_ARCH: {
        "divergence_difference_tol": 0.003,
        "divergence_n_ulps": None,
    },
}
