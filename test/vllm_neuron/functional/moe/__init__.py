"""MoE functional-kernel tests for the fork's ``test/`` overlay.

Present for the same reason as ``test/vllm_neuron/functional/__init__.py``: it
keeps the ``__init__.py`` chain unbroken from ``test/`` down to this package,
so pytest's prepend import mode stops its upward walk at the repository root
and ``import vllm_neuron`` here reaches the real plugin package.
"""
