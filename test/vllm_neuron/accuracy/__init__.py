"""Tests for ``vllm_neuron.accuracy``.

This file exists for the import chain, not for its contents. pytest's prepend
import mode walks up from a test file while each directory holds an
``__init__.py`` and puts the first directory that does not on ``sys.path``. With
``test/__init__.py``, ``test/vllm_neuron/__init__.py`` and this file all present
the walk stops at the repository root, so ``import vllm_neuron`` here resolves to
the real plugin package rather than the empty ``test/vllm_neuron/__init__.py``.
Remove this file and the walk stops at ``test/`` instead, which shadows the
plugin and makes every ``vllm_neuron.accuracy`` import in this directory fail.

``fixtures/`` deliberately carries no ``__init__.py``: it holds data, not modules.
"""
