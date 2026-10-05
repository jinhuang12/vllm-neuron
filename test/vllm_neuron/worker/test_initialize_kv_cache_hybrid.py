# SPDX-License-Identifier: Apache-2.0
"""``NeuronModelRunner.initialize_kv_cache`` on a hybrid recurrent/latent stack.

The allocator turns the spec dict into real buffers. Every expected number is
measured against something the code reports rather than written out here: the
pages against ``page_size_bytes`` read off the spec objects
``get_kv_cache_spec`` builds, the element counts against the two ``LayerSpec``
shape fields, the latent head size against the model's own reported value.

The 45 ``LayerSpec`` objects, the fake model and the spec dict come from
``test_get_kv_cache_spec_hybrid``, so there is one construction of the fake
rather than two free to drift apart. Two stages: a small layer count with real
CPU tensors, where both state buffers are written and read back, then the same
assertions at 45 layers with allocation wrapped in a counter. The wrapper
delegates to the real allocator -- the whole footprint is about 10 MB at two
blocks per entry -- so the buffers stay real.

Run with ``VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2
VLLM_SSM_CONV_STATE_LAYOUT=SD``: the conv layout is pinned only so the derived
shapes are deterministic. Every count here is a byte total or a product of
extents, so no assertion depends on the conv orientation. This is a host-side
allocation test; it says nothing about the model's use of the buffers or about
any device behaviour.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from test.vllm_neuron.worker.test_get_kv_cache_spec_hybrid import (
    DSA_ENTRIES,
    HYBRID_BLOCK_SIZE,
    KDA_ENTRIES,
    KDA_STATE_PAGE_BYTES,
    TOTAL_ENTRIES,
    _call,
    _fake_layers,
    _FakeModel,
    _non_none,
    _padded_page_expected_for_kda,
    _raw_fixture,
)

#: Two blocks per entry at the full stack: enough that a ``num_blocks`` stuck at
#: 1 cannot pass as the computed value, and about 10 MB in total.
NUM_BLOCKS_FULL = 2

#: The small stage -- the first eight layers of the fixture's own schedule, which
#: carries both families there (six KDA, two DSA), at three blocks per entry.
TINY_LAYER_COUNT = 8
NUM_BLOCKS_TINY = 3

#: The longest sequence the fake runner admits, handed to ``InputBatch``.
MAX_MODEL_LEN = 256

#: The fake runner's concurrent-sequence bound, ``max_num_reqs``. A recurrent bank
#: holds one slot per concurrent sequence, so this, not the block count, is its
#: leading extent. Different from both block counts above, so a bank sized by
#: blocks cannot pass as one sized by slots.
STATE_SLOTS = 4

#: What the allocator raises for a spec class it has no branch for.
UNSUPPORTED_SPEC_MESSAGE = "Unsupported Attention spec type"


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


class _StubInputBatch:
    """``initialize_kv_cache`` builds one; nothing here measures it."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _drive(kv_cache_config, layers, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Call the unbound ``initialize_kv_cache`` on CPU; no runner is built.

    The two helpers on the allocator's own surface are the real ones, bound onto
    the fake self, so the FP8-packed decision and the K allocation shape are
    exercised rather than stubbed.
    """
    module = _runner_module()
    runner = module.NeuronModelRunner
    monkeypatch.setattr(module, "InputBatch", _StubInputBatch)
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)

    bound: list[dict] = []
    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        neuron_config=SimpleNamespace(fp8_packed_kv=False),
        speculative_config=None,
        drafter=None,
        device=torch.device("cpu"),
        max_num_reqs=STATE_SLOTS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=256,
        vocab_size=128,
        is_pooling_model=False,
        model=_FakeModel(layers),
        _kv_cache_full_tensors={},
    )
    fake.model.bind_kv_cache = bound.append
    fake._kv_cache_is_fp8_packed = MethodType(runner._kv_cache_is_fp8_packed, fake)
    fake._k_cache_alloc_shape = runner._k_cache_alloc_shape

    caches = runner.initialize_kv_cache(fake, kv_cache_config)
    # The dict handed to the model is the dict returned, so nothing below reads a
    # structure the model never sees.
    assert bound and bound[0] is caches
    return caches


def _config(specs: dict, num_blocks: int):
    """One ``KVCacheTensor`` per layer, each sized to that layer's own page."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    tensors = [
        KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[name])
        for name, spec in specs.items()
    ]
    groups = [
        KVCacheGroupSpec(layer_names=names, kv_cache_spec=specs[names[0]])
        for names in _split(specs)
        if names
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
    )


def _split(specs: dict) -> tuple[list[str], list[str]]:
    """``(kda_names, dsa_names)``, read off the returned spec classes."""
    from vllm.v1.kv_cache_interface import MambaSpec

    kda = [name for name, spec in specs.items() if isinstance(spec, MambaSpec)]
    dsa = [name for name, spec in specs.items() if not isinstance(spec, MambaSpec)]
    return kda, dsa


def _counting_zeros(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size of every raw-buffer allocation, then delegate to torch.

    The real allocator does the work because the full footprint is about 10 MB,
    so the buffers stay real while every request is counted.
    """
    real_zeros = torch.zeros
    requested: list[int] = []

    def counting(*args, **kwargs):
        tensor = real_zeros(*args, **kwargs)
        requested.append(tensor.numel() * tensor.element_size())
        return tensor

    monkeypatch.setattr(torch, "zeros", counting)
    return requested


def _allocated_bytes(caches: dict) -> int:
    """Bytes the returned buffers span."""
    return sum(
        buffer.numel() * buffer.element_size()
        for buffers in caches.values()
        for buffer in buffers
    )


def _model_reported_head_size(raw: dict) -> int:
    """``kv_lora_rank + qk_rope_head_dim`` off the fixture."""
    text = raw["text_config"]
    return int(text["kv_lora_rank"]) + int(text["qk_rope_head_dim"])


def _addressable_page_bytes(spec) -> int:
    """Bytes of one page (attention) or one slot (recurrent) a buffer can reach.

    A recurrent state occupies only its own geometry while the page it reports to
    vLLM is padded up to the attention page. The allocator gives each recurrent
    layer one slot per concurrent sequence of exactly that geometry, so a returned
    view spans the geometry and there is no pad to skip. An attention page has no
    pad, so ``page_size_bytes`` is already the addressable answer there.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    if not isinstance(spec, MambaSpec):
        return spec.page_size_bytes
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(spec.shapes, spec.dtypes)
    )


def _entries_per_buffer(spec, num_blocks: int) -> int:
    """Pages (attention) or slots (recurrent) one entry's buffer holds."""
    from vllm.v1.kv_cache_interface import MambaSpec

    return STATE_SLOTS if isinstance(spec, MambaSpec) else num_blocks


def _addressable_bytes(specs: dict, num_blocks: int) -> int:
    """Every entry's addressable page or slot, times the pages or slots it holds."""
    return sum(
        _addressable_page_bytes(spec) * _entries_per_buffer(spec, num_blocks)
        for spec in specs.values()
    )


def _kda_natural_pages(specs: dict, kda_names: list) -> list:
    """The distinct addressable page across the KDA entries, sorted."""
    return sorted({_addressable_page_bytes(specs[name]) for name in kda_names})


def test_every_layer_is_allocated_through_its_own_spec_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """45 allocated entries, the 34 recurrent ones with two state buffers each."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, dsa_names = _split(specs)
    _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    assert len(caches) == TOTAL_ENTRIES
    assert (len(kda_names), len(dsa_names)) == (KDA_ENTRIES, DSA_ENTRIES)
    assert all(len(caches[name]) == 2 for name in kda_names)

    # Attention specs on the same path take the attention branch, which allocates
    # 4-D buffers rather than the two state buffers above.
    attention_everywhere = {
        name: (
            FullAttentionSpec(
                block_size=HYBRID_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=_model_reported_head_size(raw),
                dtype=torch.bfloat16,
                sliding_window=None,
                attention_chunk_size=None,
            )
            if isinstance(spec, MambaSpec)
            else spec
        )
        for name, spec in specs.items()
    }
    control = _drive(_config(attention_everywhere, NUM_BLOCKS_FULL), layers, monkeypatch)
    control_shapes = {tuple(control[name][0].shape) for name in kda_names}
    assert len(control) == TOTAL_ENTRIES
    assert {len(shape) for shape in control_shapes} == {4}

    # A spec class with no branch raises rather than allocating something wrong.
    class _Unmatched:
        pass

    monkeypatch.setattr(_runner_module(), "MambaSpec", _Unmatched)
    with pytest.raises(NotImplementedError) as raised:
        _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)
    assert UNSUPPORTED_SPEC_MESSAGE in str(raised.value)


def test_recurrent_entries_allocate_both_state_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two buffers per recurrent entry, sized by the two shape fields."""
    from vllm.v1.kv_cache_interface import MambaSpec

    raw = _raw_fixture()

    # ---- Stage 1: a small layer count, with real CPU tensors. ----------------
    tiny_layers = _fake_layers(raw)[:TINY_LAYER_COUNT]
    tiny_specs = _call(tiny_layers)
    tiny_kda, tiny_dsa = _split(tiny_specs)
    tiny = _drive(_config(tiny_specs, NUM_BLOCKS_TINY), tiny_layers, monkeypatch)
    by_name = {layer.name: layer for layer in tiny_layers}
    assert len(tiny) == TINY_LAYER_COUNT
    assert tiny_kda and tiny_dsa

    for name in tiny_kda:
        layer = by_name[name]
        # Element counts are products of extents, so nothing here reads an
        # orientation.
        conv_elements = math.prod(layer.kda_conv_state_shape)
        recurrent_elements = math.prod(layer.kda_recurrent_state_shape)
        assert len(tiny[name]) == 2
        conv_buffer, recurrent_buffer = tiny[name]
        assert conv_buffer.numel() == STATE_SLOTS * conv_elements
        assert recurrent_buffer.numel() == STATE_SLOTS * recurrent_elements
        assert conv_buffer.dtype is layer.kda_conv_state_dtype
        assert recurrent_buffer.dtype is layer.kda_recurrent_state_dtype
        assert conv_buffer.shape[0] == STATE_SLOTS
        assert recurrent_buffer.shape[0] == STATE_SLOTS

    # The two states must not overlap inside the page: write one, read the other
    # back. A wrong storage offset shows here as a corrupted read.
    conv_buffer, recurrent_buffer = tiny[tiny_kda[0]]
    conv_buffer.fill_(1)
    recurrent_buffer.fill_(2)
    assert torch.all(conv_buffer == 1)
    assert torch.all(recurrent_buffer == 2)
    # Distinct slots are distinct storage, so a collapsed slot stride shows.
    for slot in range(STATE_SLOTS):
        conv_buffer[slot].fill_(slot + 3)
    per_slot = [float(conv_buffer[b].flatten()[0]) for b in range(STATE_SLOTS)]
    assert per_slot == [float(b + 3) for b in range(STATE_SLOTS)]
    # And the recurrent state of every slot survived the conv writes untouched.
    assert torch.all(recurrent_buffer == 2)

    # ---- Stage 2: the same assertions at 45 layers, allocation counted. ------
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, _ = _split(specs)
    requested = _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)
    full_by_name = {layer.name: layer for layer in layers}
    two_buffer_entries = [name for name in kda_names if len(caches[name]) == 2]
    element_counts = {
        (
            caches[n][0].numel() // STATE_SLOTS,
            caches[n][1].numel() // STATE_SLOTS,
        )
        for n in kda_names
    }
    expected_counts = {
        (
            math.prod(full_by_name[n].kda_conv_state_shape),
            math.prod(full_by_name[n].kda_recurrent_state_shape),
        )
        for n in kda_names
    }
    assert len(two_buffer_entries) == KDA_ENTRIES
    assert element_counts == expected_counts
    assert len(requested) == TOTAL_ENTRIES

    # One shape in, one buffer out: the entry count follows the carriers the spec
    # holds rather than the larger state alone. Built at the allocator's boundary
    # because ``get_kv_cache_spec`` refuses a partial geometry outright, measured
    # just below, so a cleared conv field never reaches this method.
    recurrent_only = {
        name: (
            MambaSpec(
                block_size=spec.block_size,
                shapes=(spec.shapes[1],),
                dtypes=(spec.dtypes[1],),
            )
            if isinstance(spec, MambaSpec)
            else spec
        )
        for name, spec in specs.items()
    }
    control = _drive(_config(recurrent_only, NUM_BLOCKS_FULL), layers, monkeypatch)
    assert sorted({len(control[name]) for name in kda_names}) == [1]

    partial = [
        replace(layer, kda_conv_state_shape=None)
        if layer.name in set(kda_names)
        else layer
        for layer in layers
    ]
    with pytest.raises(ValueError):
        _call(partial)


def test_latent_entries_allocate_the_head_size_the_model_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 11 latent entries allocate one bank at the reported head size."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, dsa_names = _split(specs)
    _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    by_name = {layer.name: layer for layer in layers}
    reported = {by_name[name].head_size for name in dsa_names}
    allocated = {int(caches[name][0].shape[-1]) for name in dsa_names}
    config_value = _model_reported_head_size(raw)

    assert len(dsa_names) == DSA_ENTRIES
    # A relation, not a constant: the allocation matches what the model reports,
    # which matches the configured kv_lora_rank (plus rope) it is derived from, so
    # changing the geometry moves all three together.
    assert len(reported) == 1
    assert allocated == reported
    assert reported == {config_value}
    # One latent bank per attention layer, two state banks per recurrent layer.
    assert all(len(caches[name]) == 1 for name in dsa_names)
    assert all(len(caches[name]) == 2 for name in kda_names)


def test_the_allocated_bytes_reconcile_with_the_pages_the_specs_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allocation total equals the addressable pages, entry by entry."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, _ = _split(specs)
    requested = _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    # What the allocator was asked for: the latent pages vLLM's config sized, and
    # one slot per concurrent sequence for each recurrent layer.
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes

    expected_bytes = sum(
        recurrent_state_slot_bytes(spec) * STATE_SLOTS
        if isinstance(spec, MambaSpec)
        else spec.page_size_bytes * NUM_BLOCKS_FULL
        for spec in specs.values()
    )
    allocated = _allocated_bytes(caches)

    assert len(caches) == TOTAL_ENTRIES
    assert allocated - _addressable_bytes(specs, NUM_BLOCKS_FULL) == 0
    assert _kda_natural_pages(specs, kda_names) == [KDA_STATE_PAGE_BYTES]

    # Per entry too, so a compensating pair of errors cannot net to zero.
    per_entry = {
        name: sum(b.numel() * b.element_size() for b in buffers)
        - _addressable_page_bytes(specs[name])
        * _entries_per_buffer(specs[name], NUM_BLOCKS_FULL)
        for name, buffers in caches.items()
    }
    assert set(per_entry.values()) == {0}

    # The counter records what was asked for: no recurrent pad is requested, so
    # the buffers handed back span exactly what was allocated. The recurrent spec
    # still reports the padded attention page to vLLM's pool; one recurrent slot
    # is smaller than that page.
    assert sum(requested) == expected_bytes
    assert allocated == expected_bytes
    assert all(
        recurrent_state_slot_bytes(specs[n]) < specs[n].page_size_bytes
        for n in kda_names
    )

    # page_size_padded is set on the recurrent entries only, and both carriers are
    # present, so the vendor's pairing cannot have truncated the sum above.
    padded = {name: spec.page_size_padded for name, spec in specs.items()}
    assert _non_none(padded) == _padded_page_expected_for_kda(specs)
    assert {(len(specs[n].shapes), len(specs[n].dtypes)) for n in kda_names} == {(2, 2)}
