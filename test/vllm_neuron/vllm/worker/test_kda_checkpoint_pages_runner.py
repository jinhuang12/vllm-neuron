# SPDX-License-Identifier: Apache-2.0
"""The runner's KDA checkpoint page views: built with prefix caching on, nothing else.

``initialize_kv_cache`` allocates one raw buffer per pool tensor for the latent layer on
it, and one slot bank per recurrent (KDA) layer (``kv_cache_allocations``). With prefix
caching on, each KDA layer's block checkpoints live in the pages of the pool tensor it
shares (``glm5next_state_banks``, "The KDA checkpoint store"), so the runner hands each
KDA layer a page view of that latent buffer: ``runner.kda_checkpoint_pages[name]``.

Read here, on the real ``initialize_kv_cache`` with the served TP 64 specs (no model
weights, small pool):

1. prefix caching off: no view is built;
2. on or off, the same buffers are allocated and the same caches are bound, so a
   view adds no bytes;
3. on: each KDA layer gets the ``[num_blocks, page_bytes]`` view of the latent buffer
   it shares, one object per pool tensor;
4. a checkpoint written through a view is the latent layer's page of that block id,
   and the KDA layer's own banks are left as they are;
5. a pool tensor that holds a KDA layer and no latent layer, or pages of another size,
   is refused by name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/worker/test_kda_checkpoint_pages_runner.py
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from vllm_neuron.vllm.worker.glm5next_state_banks import (
    kda_checkpoint_layout,
    read_kda_checkpoints,
    write_kda_checkpoints,
)

pytestmark = [pytest.mark.fast]

MAX_NUM_SEQS, MAX_MODEL_LEN = 2, 512
PAGE_BYTES = 131_072


@pytest.fixture(scope="module")
def prefix_caching_point():
    """A runner fake with prefix caching on (128-token KDA blocks) and its KV config."""
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                            gate_knobs=False)
    assert runner.vllm_config.cache_config.enable_prefix_caching is True
    config = kv.served_kv_cache_config(runner, kv.fake_worker(runner))
    return runner, config


def _initialize(monkeypatch, runner, config, *, prefix_caching: bool):
    """The real ``initialize_kv_cache`` with the cache config's flag as given."""
    from vllm_neuron.vllm.worker import neuron_model_runner as module

    monkeypatch.setattr(module, "InputBatch", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(runner.vllm_config.cache_config, "enable_prefix_caching",
                        prefix_caching)
    runner.model.bound.clear()
    caches = module.NeuronModelRunner.initialize_kv_cache(runner, config)
    return caches, runner.kda_checkpoint_pages


def _specs(config):
    return {name: group.kv_cache_spec for group in config.kv_cache_groups
            for name in group.layer_names}


def _kda_names(config):
    from vllm.v1.kv_cache_interface import MambaSpec

    return [name for name, spec in _specs(config).items() if isinstance(spec, MambaSpec)]


def _form(caches):
    """What the model is bound to: every cache tensor's geometry and its buffer's size."""
    return {
        name: [(tuple(t.shape), t.stride(), t.dtype, t.storage_offset(),
                t.untyped_storage().nbytes()) for t in tensors]
        for name, tensors in caches.items()
    }


def _buffers(caches):
    return {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for tensors in caches.values() for t in tensors}


def test_with_prefix_caching_off_no_page_view_is_built(monkeypatch, prefix_caching_point):
    runner, config = prefix_caching_point
    _, pages = _initialize(monkeypatch, runner, config, prefix_caching=False)
    assert pages == {}


def test_the_page_views_add_no_allocation_and_leave_the_caches_as_they_are(
    monkeypatch, prefix_caching_point
):
    runner, config = prefix_caching_point
    off, _ = _initialize(monkeypatch, runner, config, prefix_caching=False)
    on, pages = _initialize(monkeypatch, runner, config, prefix_caching=True)

    assert _form(on) == _form(off)
    assert sorted(_buffers(on).values()) == sorted(_buffers(off).values())
    assert {view.untyped_storage().data_ptr() for view in pages.values()} <= set(_buffers(on))
    assert runner.model.bound == [on]


def test_each_kda_layer_gets_the_page_view_of_the_latent_buffer_it_shares(
    monkeypatch, prefix_caching_point
):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    runner, config = prefix_caching_point
    caches, pages = _initialize(monkeypatch, runner, config, prefix_caching=True)
    specs = _specs(config)

    assert set(pages) == set(_kda_names(config))
    objects = set()
    shared = [t for t in config.kv_cache_tensors if set(t.shared_by) & set(pages)]
    # 34 KDA layers in groups of 9, 9, 8 and 8 over 11 tensors: two hold a latent layer only.
    assert len(shared) == 9
    for tensor in shared:
        (latent,) = [n for n in tensor.shared_by if isinstance(specs[n], MLAAttentionSpec)]
        kda = [n for n in tensor.shared_by if n != latent]
        view = pages[kda[0]]
        assert all(pages[name] is view for name in kda)
        assert view.untyped_storage().data_ptr() == caches[latent][0].untyped_storage().data_ptr()
        assert tuple(view.shape) == (config.num_blocks, PAGE_BYTES)
        assert view.dtype == torch.int8 and view.is_contiguous()
        objects.add(id(view))
    assert len(objects) == len(shared)


def test_a_checkpoint_written_through_the_runner_view_is_the_latent_layers_page(
    monkeypatch, prefix_caching_point
):
    runner, config = prefix_caching_point
    caches, pages = _initialize(monkeypatch, runner, config, prefix_caching=True)
    specs = _specs(config)
    name = _kda_names(config)[0]
    (tensor,) = [t for t in config.kv_cache_tensors if name in t.shared_by]
    (latent,) = [n for n in tensor.shared_by if n not in pages]
    layout = kda_checkpoint_layout(specs[name])
    generator = torch.Generator().manual_seed(0)
    rows = [torch.randn((1, *shape), generator=generator).to(dtype)
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)]
    block = config.num_blocks - 1

    write_kda_checkpoints(pages[name], torch.tensor([block]), rows, layout)

    page = caches[latent][0][block].reshape(-1).view(torch.int8)
    expected = torch.cat([r.reshape(-1).view(torch.int8) for r in rows])
    assert torch.equal(page[: layout.used_bytes], expected)
    assert not caches[latent][0][:block].view(torch.int8).any()
    assert all(not bank.view(torch.int8).any() for bank in caches[name])
    got = read_kda_checkpoints(pages[name], torch.tensor([block]), layout)
    assert all(torch.equal(g, r) for g, r in zip(got, rows, strict=True))


def _config_with(config, **changes):
    return dataclasses.replace(config, **changes)


def test_a_pool_tensor_with_a_kda_layer_and_no_latent_layer_is_refused(prefix_caching_point):
    from vllm_neuron.vllm.worker.neuron_model_runner import kda_checkpoint_page_views

    _, config = prefix_caching_point
    name = _kda_names(config)[0]
    alone = dataclasses.replace(config.kv_cache_tensors[0], shared_by=[name])
    config = _config_with(config, kv_cache_tensors=[alone])
    with pytest.raises(ValueError, match=f"{name}.*no attention layer"):
        kda_checkpoint_page_views(config, {})


def test_pages_of_another_size_than_the_kda_page_are_refused(prefix_caching_point):
    from vllm_neuron.vllm.worker.neuron_model_runner import kda_checkpoint_page_views

    _, config = prefix_caching_point
    tensor = config.kv_cache_tensors[0]
    # Whole 131072-byte pages, twice as many as blocks: the view would cut the wrong rows.
    double = dataclasses.replace(tensor, size=tensor.size * 2)
    raw = {name: torch.zeros(double.size, dtype=torch.int8) for name in tensor.shared_by}
    with pytest.raises(ValueError, match="131072-byte pages and the KV cache tensor it "
                                         "shares has 262144-byte pages"):
        kda_checkpoint_page_views(_config_with(config, kv_cache_tensors=[double]), raw)
