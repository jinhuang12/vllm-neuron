# SPDX-License-Identifier: Apache-2.0
#
# Package marker for the GLM-5.3-Flash end-to-end test directory.
#
# WHY THIS FILE EXISTS AT ALL. The directory it marks is new, and every directory above it
# already carries one, so omitting it would make this the only gap in the chain. The plugin's
# test tree is collected as packages, so a missing marker here would let two test modules with
# the same basename in different directories collide at import time. The rule that requires it
# admits no exception outside the vendored ``upstream/`` tree.
#
# THERE IS NO SOURCE-SIDE COUNTERPART. ``e2e/`` is a test-tree node with no mirror under
# ``vllm_neuron/``, so no packaging obligation follows from this file.
