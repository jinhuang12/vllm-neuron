"""MoE functional-kernel tests for the fork's ``test/`` overlay.

Present for the same reason as ``test/vllm_neuron/functional/__init__.py``: it
keeps D15's ``__init__.py`` chain unbroken from ``test/`` down to this package,
so pytest's prepend import mode stops its upward walk at the repository root and
these modules are named ``test.vllm_neuron.functional.moe.*``. A gap anywhere in
that chain puts the 0-byte ``test/vllm_neuron/__init__.py`` overlay ahead of the
real plugin package on ``sys.path``.
"""
