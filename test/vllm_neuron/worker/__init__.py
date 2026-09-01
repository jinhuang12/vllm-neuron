"""Worker/runner tests for the fork's ``test/`` overlay.

This file exists for D15's import chain, not for its contents. pytest's prepend
import mode walks up from a test file while each directory holds an
``__init__.py`` and inserts the first directory that does not. With
``test/__init__.py``, ``test/vllm_neuron/__init__.py`` and this file all
present, that walk stops at the repository root -- so the root goes on
``sys.path``, these modules are named ``test.vllm_neuron.worker.*``, and a plain
``import vllm_neuron`` inside them resolves to the **real** plugin package
instead of the 0-byte ``test/vllm_neuron/__init__.py`` overlay.

Remove this file and the walk stops at ``test/`` instead, which puts the overlay
first on ``sys.path`` and makes every ``vllm_neuron.vllm.worker`` import here
fail. ``inc-glm53f-016`` is the first increment on this import path, so the
chain cost is owed and paid here.
"""
