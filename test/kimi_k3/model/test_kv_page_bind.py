# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KV bind: one page view per KV buffer, on the meta device and on CPU.

The device-free compile traces on meta, where every storage's data pointer is 0, so
the bind must tell the buffers apart there as it does on CPU. Each device gets its own
served model (one rank of TP 32 by DP 2), vLLM's KV config at 220 blocks, and the
runner's own allocation and bind. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint
directory; run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import types
from dataclasses import dataclass

import pytest
import test_kv_cache_spec as spec
import torch
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)

from vllm_neuron.model.kimi_k3 import model as kimi_model
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

pytestmark = spec.pytestmark

BLOCKS, PAGE, BUFFERS, LAYERS = 220, 294_912, 24, 93
CONV_SHAPE = (3, 1152)  # bf16 conv rows, the first bytes of a KDA page
STATE_START = 3 * 1152 * 2
GUARD = "a cache view must lie inside each page of its page buffer"
DEVICES = ["meta", "cpu"]


@dataclass
class Bind:
    """One device's KV config and what its bind left on each layer."""

    config: object
    caches: dict | None = None
    pages: list | None = None
    storages: list | None = None
    own: list | None = None
    layouts: list | None = None
    error: ValueError | None = None


def _model():
    """The 93-layer model on meta, as test_kv_cache_spec.py builds it."""
    group = types.SimpleNamespace(
        world_size=spec.TP, rank_in_group=0, device_group=None
    )
    experts = types.SimpleNamespace(
        world_size=spec.TP * spec.DP, rank_in_group=0, device_group=None
    )
    with pytest.MonkeyPatch.context() as patch, torch.device("meta"):
        patch.setattr(kimi_model, "_tp_group", lambda: group)
        patch.setattr(kimi_model, "get_world_group", lambda: experts)
        patch.setattr(kimi_model, "get_wide_ep_group", lambda: experts)
        model = kimi_model.KimiK3ForConditionalGeneration.from_configs(
            get_config(str(spec.CHECKPOINT), trust_remote_code=False)
        )
        model.prepare_after_load()
        return model


def _bind(device: str) -> Bind:
    """Allocate and bind the served KV caches on ``device`` by the runner's code."""
    vllm_config = spec._engine_config(prefix_caching=False)
    current_platform.update_block_size_for_backend(vllm_config)
    model = _model()
    runner = spec._runner(vllm_config, model)
    specs = runner.get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(vllm_config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(vllm_config, specs)
    budget = BLOCKS * BUFFERS * PAGE
    config = get_kv_cache_config_from_groups(vllm_config, groups, budget)
    layers = model.model.layers
    tensors = config.kv_cache_tensors
    kinds = sorted(
        (len(t.layers), {layers[_index(n)].is_kda for n in t.layers}) for t in tensors
    )
    assert config.num_blocks == BLOCKS
    assert kinds == [(23, {True})] * 3 + [(24, {False})]
    assert {t.layer_stride for t in tensors} == {BLOCKS * PAGE}

    runner.device = torch.device(device)
    try:
        caches = runner.initialize_kv_cache(config)
    except ValueError as error:
        return Bind(config, error=error)
    attns = [layer.self_attn for layer in layers]
    views = [caches[f"layers.{i}.self_attn"] for i in range(len(layers))]
    return Bind(
        config,
        caches,
        pages=[a.pages for a in attns],
        storages=[a.pages.untyped_storage()._cdata for a in attns],
        own=[
            {t.untyped_storage()._cdata for t in (v[:2] if layer.is_kda else v)}
            for v, layer in zip(views, layers)
        ],
        layouts=[
            (a._conv_layout, a._state_layout) if layer.is_kda else (a._cache_layout,)
            for a, layer in zip(attns, layers)
        ],
    )


def _index(name: str) -> int:
    """The layer index in a ``layers.<i>.self_attn`` name."""
    return int(name.split(".")[1])


def _bound(bind: Bind) -> Bind:
    """The bind's readings; a bind that raised fails the test with its error."""
    if bind.error is not None:
        pytest.fail(f"the bind raised: {bind.error}")
    return bind


@pytest.fixture(scope="module")
def meta_bind():
    """The served KV caches allocated and bound on meta."""
    return _bind("meta")


@pytest.fixture(scope="module")
def cpu_bind():
    """The served KV caches allocated and bound on CPU, about 1.56 GB of pages."""
    return _bind("cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_bind_keeps_one_page_view_per_buffer(device, request):
    bind = _bound(request.getfixturevalue(f"{device}_bind"))
    assert len({id(p) for p in bind.pages}) == BUFFERS
    assert len(set(bind.storages)) == BUFFERS


@pytest.mark.parametrize("device", DEVICES)
def test_each_layer_binds_its_own_buffer(device, request):
    bind = _bound(request.getfixturevalue(f"{device}_bind"))
    off = [i for i in range(LAYERS) if bind.own[i] != {bind.storages[i]}]
    assert len(bind.pages) == LAYERS
    assert off == []


def test_buffers_and_offsets_follow_the_config_on_both_devices(meta_bind, cpu_bind):
    # Layers at the same place in their KV tensors share a buffer.
    tensors = meta_bind.config.kv_cache_tensors
    expected = [
        {_index(t.layers[j]) for t in tensors if j < len(t.layers)}
        for j in range(BUFFERS)
    ]
    assert expected == [set(range(4 * j, 4 * j + 4)) for j in range(23)] + [{92}]
    for bind in (_bound(meta_bind), _bound(cpu_bind)):
        shared = {}
        for i, storage in enumerate(bind.storages):
            shared.setdefault(storage, set()).add(i)
        assert sorted(shared.values(), key=min) == expected
        for layer, layout in enumerate(bind.layouts):
            starts = [start for start, _, _ in layout]
            assert starts == ([0, STATE_START] if len(layout) == 2 else [0]), layer
    assert meta_bind.layouts == cpu_bind.layouts


@pytest.mark.parametrize("device", DEVICES)
def test_page_layout_refuses_a_view_of_another_buffer(device, request):
    bind = _bound(request.getfixturevalue(f"{device}_bind"))
    conv, _, pages = bind.caches["layers.0.self_attn"]
    other = bind.caches["layers.4.self_attn"][2]
    assert other.untyped_storage()._cdata != conv.untyped_storage()._cdata
    assert (other.shape, other.stride(), other.dtype) == (
        pages.shape,
        pages.stride(),
        pages.dtype,
    )
    with pytest.raises(ValueError, match=GUARD):
        kimi_model._page_layout(conv, other)
    assert kimi_model._page_layout(conv, pages) == (0, torch.bfloat16, CONV_SHAPE)
