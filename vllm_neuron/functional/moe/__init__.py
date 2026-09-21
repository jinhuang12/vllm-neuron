# SPDX-License-Identifier: Apache-2.0
"""Public exports for the MoE functional subpackage.

Only the two entry points of the block-quant MoE path are re-exported: the
kernel seam and the host-side scale publisher. Helpers such as
``to_kernel_scale_layout``, ``flat_scale_index`` and ``kernel_scale_shape``
are deliberately left out -- ``moe_blockwise_fp8`` and the dense
``blockwise_fp8_mm`` both define them at different signatures, so a flat
re-export would resolve to whichever module was imported last and fail as an
arity error far from its cause. Import those by their own module path.
"""

from .blockwise_fp8_retile import retile_block_scales
from .moe_blockwise_fp8 import blockwise_fp8_moe

# Alphabetical, matching the parent package's own convention.
__all__ = [
    "blockwise_fp8_moe",
    "retile_block_scales",
]
