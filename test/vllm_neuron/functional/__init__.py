"""Functional-kernel tests for the fork's ``test/`` overlay.

Present for the import chain, not for its contents. pytest's prepend import
mode walks up from a test file while each directory holds an ``__init__.py``
and puts the first directory without one on ``sys.path``. Keeping this file
makes the walk stop at the repository root, so these modules are named
``test.vllm_neuron.functional.*`` and ``import vllm_neuron`` inside them
resolves to the real plugin package rather than the empty
``test/vllm_neuron/__init__.py`` overlay.
"""
