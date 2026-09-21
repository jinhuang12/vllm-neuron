# SPDX-License-Identifier: Apache-2.0
"""The MLA latent cache is allocated as one buffer per layer.

MLA keeps one latent vector per token and has no value half. A plain
``FullAttentionSpec`` budgets two buffers per page -- upstream hardcodes the 2 in
``AttentionSpec.real_page_size_bytes`` and adds no override on that subclass -- so
reporting it for a latent cache buys a second buffer nothing reads and halves the
pages a fixed byte budget can hold. ``MLAAttentionSpec`` is upstream's own
one-vector page.

Every expected number is read from something the code reports: the two candidate
pages off vendor spec objects, the block counts off the returned buffers, the
bytes read off the model's own layer records, the recurrent-state page off the
value the hybrid spec module carries. The 45-layer fixture, its derived
``LayerSpec`` list and the unbound-method driver come from
``test_get_kv_cache_spec_hybrid``, so there is one construction of the fake; the
config helper below is this file's own, because it sizes every buffer by a fixed
byte budget where the shared one sizes it per page.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_SSM_CONV_STATE_LAYOUT=SD``. This is a
host-side allocation and mapping test; it covers no device or serving behaviour.
"""

from __future__ import annotations

import copy
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
    _raw_fixture,
)

#: The field on ``LayerSpec`` by which a layer declares a latent cache. Named
#: once so the whole file follows a rename of it.
LATENT_FIELD = "latent_kv"

#: The longest sequence the fake runner admits, handed to ``InputBatch``.
MAX_MODEL_LEN = 256

#: Blocks the budget below buys at the one-buffer page. Two is enough to catch a
#: count stuck at 1; eight keeps the halved count a whole number and the whole
#: 45-layer footprint under 50 MB of host memory.
BLOCKS_AT_ONE_BUFFER = 8


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


class _StubInputBatch:
    """The allocator builds one; nothing here measures it."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _latent_geometry(raw: dict) -> dict:
    """The latent cache's own geometry, off the fixture and the model."""
    text = raw["text_config"]
    return dict(
        block_size=HYBRID_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=int(text["kv_lora_rank"]) + int(text["qk_rope_head_dim"]),
        dtype=torch.bfloat16,
    )


def _vendor_pages(raw: dict) -> tuple[int, int]:
    """``(one_buffer, two_buffer)`` pages, both reported by vendor spec objects.

    Neither number is written down here: the pair is what the two candidate spec
    classes say about one identical geometry, which is the whole point.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

    geometry = _latent_geometry(raw)
    return (
        MLAAttentionSpec(**geometry).page_size_bytes,
        FullAttentionSpec(**geometry).page_size_bytes,
    )


def _without_latent(layers: list) -> list:
    """The same layers with every latent declaration cleared.

    This builds the absence rather than inheriting it, so the comparison cannot turn
    into a no-op if the producer starts setting the field somewhere new.
    """
    return [replace(layer, **{LATENT_FIELD: False}) for layer in layers]


def _split(specs: dict) -> tuple[list[str], list[str]]:
    """``(kda_names, attention_names)``, read off the returned spec classes."""
    from vllm.v1.kv_cache_interface import MambaSpec

    kda = [name for name, spec in specs.items() if isinstance(spec, MambaSpec)]
    recurrent = set(kda)
    return kda, [name for name in specs if name not in recurrent]


def _fixed_budget_config(specs: dict, budget_bytes: int):
    """One equally sized buffer per layer, so the page decides the block count.

    The shared config helper sizes each buffer as ``page x blocks``, which can
    never show a block count moving. Here every layer gets the same byte budget
    and the allocator derives its own count from the page it reports.
    """
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    tensors = [
        KVCacheTensor(size=budget_bytes, shared_by=[name]) for name in specs
    ]
    groups = [
        KVCacheGroupSpec(layer_names=names, kv_cache_spec=specs[names[0]])
        for names in _split(specs)
        if names
    ]
    blocks = min(budget_bytes // spec.page_size_bytes for spec in specs.values())
    return KVCacheConfig(
        num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
    )


def _drive(kv_cache_config, model, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Call the unbound allocator on CPU against ``model``; no runner is built.

    The two helpers this file reads are the real ones, bound onto the fake self,
    so the packed-K decision and the K allocation shape are exercised rather than
    stubbed.
    """
    module = _runner_module()
    runner = module.NeuronModelRunner
    monkeypatch.setattr(module, "InputBatch", _StubInputBatch)
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)

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
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=256,
        vocab_size=128,
        is_pooling_model=False,
        model=model,
        _kv_cache_full_tensors={},
    )
    fake._kv_cache_is_fp8_packed = MethodType(runner._kv_cache_is_fp8_packed, fake)
    fake._k_cache_alloc_shape = runner._k_cache_alloc_shape

    # A model with no mapper of its own gets a collector, so the tests that do not
    # read the mapper still exercise the real bind call rather than skipping it.
    bound: list[dict] = []
    if not hasattr(model, "bind_kv_cache"):
        model.bind_kv_cache = bound.append
    caches = runner.initialize_kv_cache(fake, kv_cache_config)
    # The dict handed to the model is the dict returned, so nothing below reads a
    # structure the model never saw.
    assert not bound or bound[0] is caches
    return caches


def _real_model(raw: dict):
    """The real model, whose own spec and own mapper are read together."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    return Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))


def _drive_real(raw: dict, monkeypatch: pytest.MonkeyPatch, *, latent: bool):
    """Allocate against the real model's own spec, then let its mapper read it.

    Spec and mapper come from one ``get_kv_spec`` call chain, so nothing compares
    a buffer built from one geometry against a reader expecting another.

    The recurrent geometry comes from the shared fixture rather than from the model
    built here: this process has no tensor-parallel world, so a model built in it
    reports the unsharded recurrent state -- 64 times the per-rank shard a serve
    carries -- and page unification refuses a recurrent page larger than the
    attention page, a geometry no serve has. ``_fake_layers`` rebuilds each real
    layer with ``replace``, so the layer's own latent declaration survives while
    the four recurrent fields become the shard at the served world size.
    """
    model = _real_model(raw)
    layers = _fake_layers(raw)
    if not latent:
        layers = _without_latent(layers)
    # One geometry for the allocation and for the mapper that reads it.
    model.get_kv_spec = lambda: SimpleNamespace(layers=layers)
    specs = _call(layers)
    _, attention_names = _split(specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    budget = one_buffer * BLOCKS_AT_ONE_BUFFER
    caches = _drive(_fixed_budget_config(specs, budget), model, monkeypatch)
    return model, specs, caches, attention_names, budget, (one_buffer, two_buffer)


def _bytes_of(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def test_latent_layers_report_the_one_buffer_page() -> None:
    """11 latent entries on the MLA spec class, at half the two-buffer page."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec

    raw = _raw_fixture()
    specs = _call(_fake_layers(raw))
    kda_names, attention_names = _split(specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    pages = {specs[name].page_size_bytes for name in attention_names}

    assert len(specs) == TOTAL_ENTRIES
    assert (len(kda_names), len(attention_names)) == (KDA_ENTRIES, DSA_ENTRIES)
    assert all(isinstance(specs[n], MLAAttentionSpec) for n in attention_names)
    assert all(isinstance(specs[n], MambaSpec) for n in kda_names)
    assert pages == {one_buffer}
    # The two-buffer class on the same geometry reports twice as much, so the page
    # above is the one-vector page and not simply whatever both classes agree on.
    assert two_buffer == 2 * one_buffer


def test_a_fixed_budget_buys_twice_the_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same bytes hold 8 blocks at the one-buffer page and 4 at two buffers."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    _, attention_names = _split(specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    budget = one_buffer * BLOCKS_AT_ONE_BUFFER

    caches = _drive(
        _fixed_budget_config(specs, budget), _FakeModel(layers), monkeypatch
    )
    blocks = {caches[name][0].shape[0] for name in attention_names}

    control_specs = _call(_without_latent(layers))
    control = _drive(
        _fixed_budget_config(control_specs, budget),
        _FakeModel(layers),
        monkeypatch,
    )
    control_blocks = {control[name][0].shape[0] for name in attention_names}

    assert blocks == {BLOCKS_AT_ONE_BUFFER}
    assert budget // one_buffer == BLOCKS_AT_ONE_BUFFER
    # The identical budget at the two-buffer page holds exactly half as many.
    assert control_blocks == {BLOCKS_AT_ONE_BUFFER // 2}
    assert budget // two_buffer == BLOCKS_AT_ONE_BUFFER // 2


def test_every_allocated_latent_byte_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapper reaches all of the latent allocation, and half of two buffers."""
    raw = _raw_fixture()
    model, _, caches, attention_names, _, _ = _drive_real(
        raw, monkeypatch, latent=True
    )
    banks = {
        record["name"]: record["latent_bank"]
        for record in model.glm5next_layer_banks
        if record["family"] == "self_attn"
    }
    read = _bytes_of(banks.values())
    allocated = sum(_bytes_of(caches[name]) for name in attention_names)

    control_model, _, control_caches, control_names, _, _ = _drive_real(
        raw, monkeypatch, latent=False
    )
    control_banks = {
        record["name"]: record["latent_bank"]
        for record in control_model.glm5next_layer_banks
        if record["family"] == "self_attn"
    }
    control_read = _bytes_of(control_banks.values())
    control_allocated = sum(_bytes_of(control_caches[name]) for name in control_names)

    assert len(banks) == DSA_ENTRIES
    assert read == allocated
    # On the two-buffer layout the mapper reaches exactly half of what was
    # allocated: the waste was never unreachable memory, it was memory no reader
    # ever touched.
    assert control_read * 2 == control_allocated


def test_the_mapper_reads_the_one_allocated_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same storage, the flattened sequence view, and one tensor per layer."""
    raw = _raw_fixture()
    model, _, caches, attention_names, _, _ = _drive_real(
        raw, monkeypatch, latent=True
    )
    records = {
        record["name"]: record
        for record in model.glm5next_layer_banks
        if record["family"] == "self_attn"
    }
    head_size = _latent_geometry(raw)["head_size"]
    name = attention_names[0]
    record = records[name]

    assert {len(caches[n]) for n in attention_names} == {1}
    assert record["latent_bank"].data_ptr() == caches[name][0].data_ptr()
    assert tuple(record["latent_bank"].shape) == (
        BLOCKS_AT_ONE_BUFFER,
        1,
        HYBRID_BLOCK_SIZE,
        head_size,
    )
    assert tuple(record["latent_cache"].shape) == (
        BLOCKS_AT_ONE_BUFFER * HYBRID_BLOCK_SIZE,
        1,
        head_size,
    )
    # A reader pointed at the value half that used to exist raises, so the second
    # buffer is gone rather than merely unused.
    with pytest.raises(IndexError):
        caches[name][1]


def test_a_layer_without_a_latent_cache_still_gets_two_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration is what decides; the layer name and the geometry are not."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    control_layers = _without_latent(layers)
    control_specs = _call(control_layers)
    _, attention_names = _split(control_specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    budget = one_buffer * BLOCKS_AT_ONE_BUFFER
    head_size = _latent_geometry(raw)["head_size"]

    control = _drive(
        _fixed_budget_config(control_specs, budget),
        _FakeModel(control_layers),
        monkeypatch,
    )
    name = attention_names[0]

    assert all(type(control_specs[n]) is FullAttentionSpec for n in attention_names)
    assert {control_specs[n].page_size_bytes for n in attention_names} == {two_buffer}
    assert {len(control[n]) for n in attention_names} == {2}
    expected = (
        BLOCKS_AT_ONE_BUFFER // 2,
        1,
        HYBRID_BLOCK_SIZE,
        head_size,
    )
    assert tuple(control[name][0].shape) == expected
    assert tuple(control[name][1].shape) == expected
    assert control[name][0].data_ptr() != control[name][1].data_ptr()

    # Declaring a latent cache on those same layers flips them to one buffer, so
    # the field is what decides rather than dead code.
    flipped_specs = _call(
        [replace(layer, **{LATENT_FIELD: True}) for layer in control_layers]
    )
    flipped = _drive(
        _fixed_budget_config(flipped_specs, budget),
        _FakeModel(control_layers),
        monkeypatch,
    )
    assert all(isinstance(flipped_specs[n], MLAAttentionSpec) for n in attention_names)
    assert {len(flipped[n]) for n in attention_names} == {1}


def test_page_unification_survives_the_halved_page() -> None:
    """One page across all 45 entries, and the downward refusal is still live."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, attention_names = _split(specs)
    one_buffer, _ = _vendor_pages(raw)
    padded = sorted(
        name for name, spec in specs.items() if spec.page_size_padded is not None
    )

    assert {spec.page_size_bytes for spec in specs.values()} == {one_buffer}
    assert padded == sorted(kda_names)
    assert len(attention_names) == DSA_ENTRIES
    # The halved page still exceeds the state it must hold, which is why the
    # padding above still runs upward.
    assert KDA_STATE_PAGE_BYTES < one_buffer

    # A recurrent state larger than the halved page must be refused, not narrowed.
    oversized = []
    for layer in layers:
        if layer.kda_recurrent_state_shape is None:
            oversized.append(layer)
            continue
        heads, rows, columns = layer.kda_recurrent_state_shape
        oversized.append(
            replace(layer, kda_recurrent_state_shape=(heads, rows, columns * 64))
        )
    with pytest.raises(ValueError, match="Cannot unify KV cache pages"):
        _call(oversized)
