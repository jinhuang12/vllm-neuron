# SPDX-License-Identifier: Apache-2.0
"""The KDA ``MambaSpec`` carries the cache config's ``mamba_cache_mode``.

vLLM's ``MambaManager`` takes its cache mode from the group's spec, so the mode on
``cache_config`` reaches the manager only if ``get_kv_cache_spec`` puts it on the spec.
Read here, with the served TP 64 layer list (no model weights): each of vLLM's three
modes reaches every KDA spec, ``page_size_padded`` keeps it, the served form (prefix
caching off) reports vLLM's default, "none", and the worker prices the mode the cache
config sets.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/worker/test_kda_spec_cache_mode.py
"""

from __future__ import annotations

import pytest

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv

pytestmark = [pytest.mark.fast]


def _kda_specs(runner):
    from vllm.v1.kv_cache_interface import MambaSpec

    specs = [s for s in runner.get_kv_cache_spec().values() if isinstance(s, MambaSpec)]
    assert len(specs) == kv.KDA_LAYERS
    return specs


@pytest.mark.parametrize("mode", ["none", "align", "all"])
def test_every_kda_spec_carries_the_cache_configs_mode(mode):
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=8192, gate_knobs=False)
    runner.vllm_config.cache_config.mamba_cache_mode = mode

    specs = _kda_specs(runner)

    assert {spec.mamba_cache_mode for spec in specs} == {mode}
    # The attention page is still set on the spec that carries the mode.
    assert {spec.page_size_bytes for spec in specs} == {kv.BLOCK_SIZE_TOKENS
                                                        * kv.LATENT_BYTES_PER_TOKEN}


def test_the_served_form_reports_vllms_default_mode():
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=4096)
    assert runner.vllm_config.cache_config.enable_prefix_caching is False

    assert {spec.mamba_cache_mode for spec in _kda_specs(runner)} == {"none"}


def test_the_worker_prices_the_mode_the_cache_config_sets():
    """Set on the cache config only, "align" reaches the spec and so the KV need."""
    layers, text_config = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=8192, gate_knobs=False,
                            text_config=text_config)
    runner.vllm_config.cache_config.mamba_cache_mode = "align"
    worker = kv.fake_worker(runner)
    page = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN
    kda_groups = -(-kv.KDA_LAYERS // kv.MLA_LAYERS)

    assert worker._kv_cache_need_bytes(log=False) == kv.MLA_LAYERS * page * (
        8192 // kv.BLOCK_SIZE_TOKENS + kda_groups * 2 + 1)
