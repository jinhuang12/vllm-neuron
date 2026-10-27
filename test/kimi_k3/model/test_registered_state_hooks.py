# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's block-size step on the class the worker registers.

Each worker registers the plugin's model classes over vLLM's, then runs the platform's block-size
step. The served form is hybrid, so the step reads the state shapes and dtypes from the class the
registry gives. The DCP 4 form's serve flags, from ``test/kimi_k3/serving/cells.json``, build the
engine config through ``vllm serve``'s parser and the platform's config step. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its config and tokenizer only); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import copy
import json
import os
from pathlib import Path

import pytest
from vllm import ModelRegistry
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_neuron.model import registry
from vllm_neuron.vllm.platform import _config_fields_only

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
CELLS = Path(__file__).resolve().parents[1] / "serving" / "cells.json"
# The hooks upstream's hybrid alignment reads from the registered class.
STATE_HOOKS = (
    "get_mamba_specs_from_config",
    "get_mamba_state_shape_from_config",
    "get_mamba_state_dtype_from_config",
)
# The block size the step gives the served config with vLLM's class (test_kv_cache_spec.py).
SERVED_BLOCK_SIZE = 256


@pytest.fixture
def served_config(tmp_path):
    """The engine config ``vllm serve`` builds from the DCP 4 form's flags."""
    flags = [f.replace("{checkpoint}", str(CHECKPOINT)).replace("{measurements}", str(tmp_path))
             for f in json.loads(CELLS.read_text(encoding="utf-8"))["forms"]["dcp4"]["server_flags"]]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([str(CHECKPOINT), *flags])
    args.model = args.model_tag
    validate_parsed_serve_args(args)
    return AsyncEngineArgs.from_cli_args(args).create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER)


def test_block_size_step_on_the_registered_class(served_config, monkeypatch):
    """After the worker's registry step, the platform's block-size step gives the served config
    the block size it gives with vLLM's class, and the registered class's state hooks return
    vLLM's values."""
    model_config = served_config.model_config
    upstream, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config=model_config)
    with_upstream = copy.deepcopy(served_config)
    current_platform.update_block_size_for_backend(with_upstream)

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    for arch, model_cls in registry.get_models():
        ModelRegistry.register_model(arch, model_cls)
    registered, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config=model_config)
    assert registered is dict(registry.get_models())[model_config.architecture]

    current_platform.update_block_size_for_backend(served_config)

    block_sizes = (served_config.cache_config.block_size, with_upstream.cache_config.block_size)
    assert block_sizes == (SERVED_BLOCK_SIZE, SERVED_BLOCK_SIZE)
    view = _config_fields_only(served_config)
    for hook in STATE_HOOKS:
        assert hasattr(registered, hook) == hasattr(upstream, hook), hook
        if hasattr(upstream, hook):
            assert getattr(registered, hook)(view) == getattr(upstream, hook)(view), hook
