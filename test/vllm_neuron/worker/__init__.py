"""Worker and runner tests.

This file exists for the import chain rather than for its contents. pytest's
prepend import mode walks up from a test file while each directory holds an
``__init__.py`` and puts the first directory without one on ``sys.path``. With
this file present the walk stops at the repository root, so these modules are
named ``test.vllm_neuron.worker.*`` and a plain ``import vllm_neuron`` inside
them resolves to the real plugin package instead of the empty
``test/vllm_neuron/__init__.py`` overlay.
"""
