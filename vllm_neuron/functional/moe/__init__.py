# SPDX-License-Identifier: Apache-2.0
"""Public exports for the MoE functional subpackage.

``inc-glm53f-027``. This file was 0 bytes at the pin: every consumer of
``vllm_neuron.functional.moe.*`` reached its module by full dotted path, and the
parent package re-exported single names through that path
(``functional/__init__.py:34`` -- ``from .moe.moe_blockwise import
build_blockwise_mapping``). The block-quant path of WP6 lands two modules that
have no such re-export, so this increment -- the plan's only declared writer of
this file and of its parent's export list -- establishes the hub for them.

WHAT IS EXPORTED, AND WHAT IS DELIBERATELY NOT
----------------------------------------------
Exported: the two **entry points** of the block-quant MoE path -- the kernel
seam (`inc-glm53f-025`) and the host-side retile producer (`inc-glm53f-024`).

NOT exported, and this is the point rather than an omission: every helper whose
NAME IS SHARED BY TWO MODULES OF THIS CAMPAIGN AT DIFFERENT SIGNATURES.
Measured over the three landed WP6 modules:

* ``to_kernel_scale_layout`` -- ``moe/moe_blockwise_fp8.py:181``
  ``(consumer_scales, num_experts, rows, cols, projection)`` versus
  ``blockwise_fp8_mm.py:309`` ``(weight_scale, rows, cols)``;
* ``flat_scale_index`` -- ``moe/blockwise_fp8_retile.py:194``
  ``(h_tile, i_tile, h_256, i_256, projection, gate_or_up)`` versus
  ``blockwise_fp8_mm.py:291`` ``(k_block, n_block, n_n_blocks)``;
* ``kernel_scale_shape``, ``dispatch_counters``, ``reset_dispatch_counters``,
  ``kernel_identity``, ``BLOCK_QUANT_SIZE``, ``TILE_SIZE`` -- each defined or
  re-exported by more than one of the three.

A flat re-export of any of those would resolve to whichever module was imported
last, and the two arities differ, so the failure would surface as a confusing
shape or arity error far from its cause. They stay reachable at their own module
path, which is the qualification that keeps them unambiguous.
"""

from .blockwise_fp8_retile import retile_block_scales
from .moe_blockwise_fp8 import blockwise_fp8_moe

# Alphabetical, matching the parent package's own convention.
__all__ = [
    "blockwise_fp8_moe",
    "retile_block_scales",
]
