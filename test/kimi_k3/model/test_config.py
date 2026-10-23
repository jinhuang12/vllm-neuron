# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 config parse against the checkpoint on disk.

The config is built from the HF config vLLM loads for the checkpoint, the
object the model runner passes to ``from_configs``. Set ``KIMI_K3_CHECKPOINT``
to the checkpoint directory; the default is ``~/models/moonshotai/Kimi-K3``.
"""

import json
import os
from pathlib import Path

import pytest

from vllm_neuron.model.kimi_k3 import KimiK3Config, stored_a_log_length

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)


@pytest.fixture(scope="module")
def stored_text_config() -> dict:
    """The checkpoint's text_config, read from config.json as stored."""
    with open(CHECKPOINT / "config.json") as f:
        return json.load(f)["text_config"]


@pytest.fixture(scope="module")
def text_config():
    """The parsed text config, built from vLLM's HF config for the checkpoint."""
    from vllm.transformers_utils.config import get_config

    hf_config = get_config(str(CHECKPOINT), trust_remote_code=False)
    return KimiK3Config.from_configs(hf_config).text_config


def test_num_hidden_layers(text_config, stored_text_config):
    assert text_config.num_hidden_layers == stored_text_config["num_hidden_layers"]
    assert text_config.num_hidden_layers == 93


def test_kda_and_mla_layers_at_configured_indices(text_config, stored_text_config):
    stored = stored_text_config["linear_attn_config"]
    assert text_config.kda_layer_indices == [i - 1 for i in stored["kda_layers"]]
    assert text_config.mla_layer_indices == [
        i - 1 for i in stored["full_attn_layers"]
    ]
    assert (len(text_config.kda_layer_indices), len(text_config.mla_layer_indices)) == (
        69,
        24,
    )


def test_expert_count_and_top_k(text_config, stored_text_config):
    assert (text_config.num_experts, text_config.num_experts_per_token) == (
        stored_text_config["num_experts"],
        stored_text_config["num_experts_per_token"],
    )
    assert (text_config.num_experts, text_config.num_experts_per_token) == (896, 16)


def test_heads_per_rank_at_tp32(text_config):
    assert text_config.kda_heads_per_rank(32) == 3
    assert text_config.mla_heads_per_rank(32) == 3


def test_latent_expert_dim(text_config, stored_text_config):
    assert (
        text_config.routed_expert_hidden_size
        == stored_text_config["routed_expert_hidden_size"]
    )
    assert text_config.routed_expert_hidden_size == 3584


def test_a_log_length_read_as_stored(text_config):
    assert stored_a_log_length(CHECKPOINT, text_config) == 128
