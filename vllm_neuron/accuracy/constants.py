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

# Architecture key shared by the per-architecture maps here and in
# logit_validation, so all of them are looked up with the same string.
GLM5NEXT_ARCH = "Glm5NextForConditionalGeneration"

# Per-architecture tolerances, consulted instead of DEFAULT_TOLERANCE_MAP when a
# model's architecture appears here. Every top_k is 3x the default: GLM-5-Next
# runs from an FP8 block-quantized checkpoint, whose per-block dequantization
# adds error a bf16 checkpoint does not carry.
#
# Tuple order is (atol, rtol) -- DEFAULT_TOLERANCE_MAP's own order, and the
# reverse of vllm_neuron.accuracy.testing._DEFAULT_DTYPE_TOLERANCE, which is
# (rtol, atol). Do not normalise one order to the other.
ARCH_TOLERANCE_MAP = {
    GLM5NEXT_ARCH: {
        "5": (1e-5, 0.033),
        "50": (1e-5, 0.06),
        "1000": (1e-5, 0.09),
        "all": (1e-5, 0.15),
    },
}

# Per-architecture divergence settings, looked up the same way. The explicit
# divergence_n_ulps=None is what makes divergence_difference_tol the tolerance
# actually applied: a ULP count, when set, overrides the fixed difference
# tolerance with a value that scales with each logit's magnitude.
ARCH_DIVERGENCE_CONFIG = {
    GLM5NEXT_ARCH: {
        "divergence_difference_tol": 0.003,
        "divergence_n_ulps": None,
    },
}
