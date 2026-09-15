# SPDX-License-Identifier: Apache-2.0
#
# Package marker for the GLM-5.3-Flash tiny-config forward test directory.
#
# WHY THIS FILE EXISTS AT ALL. The directory it marks is new, and every directory above it
# already carries one, so omitting it would make this the only gap in the chain. The plugin's
# test tree is collected as packages, so a missing marker here would let two test modules with
# the same basename in different directories collide at import time. The rule that requires it
# admits no exception outside the vendored ``upstream/`` tree.
#
# THERE IS NO SOURCE-SIDE COUNTERPART. ``tiny/`` is a test-tree node with no mirror under
# ``vllm_neuron/``, so no packaging obligation follows from this file.
#
# THE DIRECTORY HAS TWO RESIDENTS AND ONE CREATOR. ``inc-glm53f-054a`` adds this marker and
# ``test_tiny_glm5next_forward.py`` beside it, and bears the whole cost of the directory;
# ``inc-glm53f-054b`` later adds ``test_tiny_glm5next_e2e.py`` and bears no part of it. The two
# are ordered and never selectable at the same time, so this directory has one writer at a time.
#
# IT CARRIES NO PYTEST MARKS. Marks belong on the test modules; a package marker declares
# nothing to collect.
