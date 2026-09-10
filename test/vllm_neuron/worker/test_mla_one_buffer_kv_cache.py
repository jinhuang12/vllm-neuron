# SPDX-License-Identifier: Apache-2.0
"""Acceptance: the MLA latent cache is allocated as ONE buffer per layer.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_SSM_CONV_STATE_LAYOUT=SD \\
      python -m pytest test/vllm_neuron/worker/test_mla_one_buffer_kv_cache.py \\
      -q -rA -p no:randomly -p no:cacheprovider --timeout 600

MLA keeps one latent vector per token and has no value half. A plain
``FullAttentionSpec`` budgets two buffers per page -- upstream hardcodes the 2 in
``AttentionSpec.real_page_size_bytes`` and adds no override on that subclass -- so
reporting it for a latent cache buys a second buffer nothing reads and halves the
pages a fixed byte budget can hold. ``MLAAttentionSpec`` is upstream's own
one-vector page. Six items, ONE test each, no ``parametrize``:

* T01 -- the latent layers report the MLA page, and it is exactly half the
  two-buffer page. Both numbers are read off vendor spec objects.
* T02 -- one fixed byte budget yields twice the blocks it used to.
* T03 -- READ FRACTION through the model's own mapper is 1.0, against 0.5 on the
  two-buffer layout. The waste was never unreachable memory; it was memory no
  reader ever touched, so the mapper is the instrument and not the allocator.
* T04 -- the mapper's bank IS the allocated tensor, and there is no second index.
* T05 -- a layer that does not declare a latent cache still gets two buffers,
  and declaring one on that same layer flips it. The declaration is the
  discriminator, not the layer's name.
* T06 -- the page unification still succeeds at the halved page, and still
  refuses to narrow a recurrent state.

REFERENTS. Every expected number is read from something the world reports: the
two pages off vendor spec objects, the block counts off the returned buffers, the
read fraction off the mapper's own records, the recurrent-state page off the
recorded value the hybrid spec module already carries. This file mints no
tolerance and no comparison pair; every assertion is an integer, a class, a
shape, a pointer identity or a raised message.

VEHICLE, shared rather than rebuilt. The 45-layer fixture, its derived
``LayerSpec`` list and the unbound-method driver come from the landed hybrid
spec module's helpers, for the reason that module's own consumer states: a second
construction of the fake would be free to drift from the one construction the
campaign declares. The drivers below are this file's own, because they size a
config by a FIXED byte budget where the landed one sizes it per page.

WHAT THIS DOES NOT CERTIFY. No hardware behaviour and no serving path: this is a
CPU-mode, host-side allocation and mapping certificate.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from test.vllm_neuron.worker.test_get_kv_cache_spec_hybrid import (
    DECLARED_DSA_ENTRIES,
    DECLARED_KDA_ENTRIES,
    DECLARED_TOTAL_ENTRIES,
    RECORDED_KDA_STATE_PAGE_BYTES,
    REGISTERED_HYBRID_BLOCK_SIZE,
    _call,
    _fake_layers,
    _FakeModel,
    _raw_fixture,
)

#: The field on ``LayerSpec`` by which a layer declares a latent cache. Named
#: once so the whole file follows a rename of it.
LATENT_FIELD = "latent_kv"

#: Blocks the budget below buys at the one-buffer page. Two is enough to catch a
#: count stuck at 1; eight keeps the halved reading a whole number and the whole
#: 45-layer footprint under 50 MB of host memory.
BLOCKS_AT_ONE_BUFFER = 8


def _record(**readings: object) -> None:
    """Put a reading in the ``-q`` transcript."""
    for key, value in readings.items():
        warnings.warn(f"RECORDED {key}={value!r}", UserWarning, stacklevel=2)


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


class _StubInputBatch:
    """The allocator builds one; no item here measures it."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _latent_geometry(raw: dict) -> dict:
    """The latent cache's own geometry, off the fixture and the model."""
    text = raw["text_config"]
    return dict(
        block_size=REGISTERED_HYBRID_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=int(text["kv_lora_rank"]) + int(text["qk_rope_head_dim"]),
        dtype=torch.bfloat16,
    )


def _vendor_pages(raw: dict) -> tuple[int, int]:
    """``(one_buffer, two_buffer)`` pages, both REPORTED by vendor spec objects.

    Neither number is written down here. The pair is what the two candidate spec
    classes say about one identical geometry, which is the whole defect.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

    geometry = _latent_geometry(raw)
    return (
        MLAAttentionSpec(**geometry).page_size_bytes,
        FullAttentionSpec(**geometry).page_size_bytes,
    )


def _without_latent(layers: list) -> list:
    """The same layers with every latent declaration cleared.

    This BUILDS the absence rather than inheriting it, so a control cannot go
    quietly hollow if the producer starts setting the field somewhere new.
    """
    return [replace(layer, **{LATENT_FIELD: False}) for layer in layers]


def _split(specs: dict) -> tuple[list[str], list[str]]:
    """``(kda_names, attention_names)``, read off the returned spec CLASSES."""
    from vllm.v1.kv_cache_interface import MambaSpec

    kda = [name for name, spec in specs.items() if isinstance(spec, MambaSpec)]
    recurrent = set(kda)
    return kda, [name for name in specs if name not in recurrent]


def _fixed_budget_config(specs: dict, budget_bytes: int):
    """One equally sized buffer per layer, so the PAGE decides the block count.

    The landed config helper sizes each buffer as ``page x blocks``, which can
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
    """Call the UNBOUND allocator on CPU against ``model``; no runner is built.

    The two helpers this file's arms read are the REAL ones, bound onto the fake
    self, so the packed-K decision and the K allocation shape are exercised
    rather than stubbed.
    """
    module = _runner_module()
    runner = module.NeuronModelRunner
    monkeypatch.setattr(module, "InputBatch", _StubInputBatch)
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)

    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=REGISTERED_HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        neuron_config=SimpleNamespace(fp8_packed_kv=False),
        speculative_config=None,
        drafter=None,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_model_len=256,
        max_num_batched_tokens=256,
        vocab_size=128,
        is_pooling_model=False,
        model=model,
        _kv_cache_full_tensors={},
    )
    fake._kv_cache_is_fp8_packed = MethodType(runner._kv_cache_is_fp8_packed, fake)
    fake._k_cache_alloc_shape = runner._k_cache_alloc_shape

    # A model with no mapper of its own gets a collector, so the arms that do not
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
    """Allocate against the REAL model's own spec, then let its mapper read it.

    Spec and mapper come from one ``get_kv_spec`` call chain, so no arm compares
    a buffer built from one geometry against a reader expecting another.
    """
    model = _real_model(raw)
    layers = list(model.get_kv_spec().layers)
    if not latent:
        layers = _without_latent(layers)
        model.get_kv_spec = lambda: SimpleNamespace(layers=layers)
    specs = _call(layers)
    _, attention_names = _split(specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    budget = one_buffer * BLOCKS_AT_ONE_BUFFER
    caches = _drive(_fixed_budget_config(specs, budget), model, monkeypatch)
    return model, specs, caches, attention_names, budget, (one_buffer, two_buffer)


def _bytes_of(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


# T01 -- the latent layers report the MLA page, half the two-buffer page.
def test_t01_latent_layers_report_the_one_buffer_page() -> None:
    """11 latent entries on the MLA spec class, at exactly half the old page."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec

    raw = _raw_fixture()
    specs = _call(_fake_layers(raw))
    kda_names, attention_names = _split(specs)
    one_buffer, two_buffer = _vendor_pages(raw)
    pages = {specs[name].page_size_bytes for name in attention_names}
    classes = sorted({type(specs[name]).__name__ for name in attention_names})

    _record(
        t01_one_buffer_page=one_buffer,
        t01_two_buffer_page=two_buffer,
        t01_attention_classes=classes,
        t01_attention_pages=sorted(pages),
        t01_attention_entries=len(attention_names),
        t01_kda_entries=len(kda_names),
    )
    assert len(specs) == DECLARED_TOTAL_ENTRIES
    assert (len(kda_names), len(attention_names)) == (
        DECLARED_KDA_ENTRIES,
        DECLARED_DSA_ENTRIES,
    )
    assert all(isinstance(specs[n], MLAAttentionSpec) for n in attention_names)
    assert all(isinstance(specs[n], MambaSpec) for n in kda_names)
    assert pages == {one_buffer}

    # MUST-FAIL ARM: the pre-change class on the SAME geometry reports twice as
    # much, so a change that did nothing cannot satisfy the line above.
    assert two_buffer == 2 * one_buffer
    assert pages != {two_buffer}


# T02 -- one fixed byte budget buys twice the blocks.
def test_t02_a_fixed_budget_buys_twice_the_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same bytes hold 8 blocks at the one-buffer page and 4 at the old one."""
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

    _record(
        t02_budget_bytes_per_layer=budget,
        t02_blocks=sorted(blocks),
        t02_control_blocks=sorted(control_blocks),
    )
    assert blocks == {BLOCKS_AT_ONE_BUFFER}
    assert budget // one_buffer == BLOCKS_AT_ONE_BUFFER

    # MUST-FAIL ARM: the two-buffer page over the identical budget is asserted to
    # give exactly half, so the item reads a count that MOVED.
    assert control_blocks == {BLOCKS_AT_ONE_BUFFER // 2}
    assert budget // two_buffer == BLOCKS_AT_ONE_BUFFER // 2


# T03 -- read fraction through the model's own mapper.
def test_t03_every_allocated_latent_byte_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapper reaches all of the latent allocation, against half of the old."""
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

    _record(
        t03_latent_layers=len(banks),
        t03_bytes_read=read,
        t03_bytes_allocated=allocated,
        t03_control_bytes_read=control_read,
        t03_control_bytes_allocated=control_allocated,
    )
    assert len(banks) == DECLARED_DSA_ENTRIES
    assert read == allocated

    # MUST-FAIL ARM: on the two-buffer layout the mapper reaches exactly half of
    # what was allocated -- the waste, stated as a number the transcript carries.
    assert control_read * 2 == control_allocated


# T04 -- the bank IS the allocated tensor, and no second index exists.
def test_t04_the_mapper_reads_the_one_allocated_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same storage, the declared sequence view, and one tensor per layer."""
    raw = _raw_fixture()
    model, specs, caches, attention_names, _, _ = _drive_real(
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

    _record(
        t04_buffers_per_latent_layer=sorted({len(caches[n]) for n in attention_names}),
        t04_bank_shape=tuple(record["latent_bank"].shape),
        t04_sequence_view_shape=tuple(record["latent_cache"].shape),
        t04_slots=record["slots"],
    )
    assert {len(caches[n]) for n in attention_names} == {1}
    assert record["latent_bank"].data_ptr() == caches[name][0].data_ptr()
    assert tuple(record["latent_bank"].shape) == (
        BLOCKS_AT_ONE_BUFFER,
        1,
        REGISTERED_HYBRID_BLOCK_SIZE,
        head_size,
    )
    assert tuple(record["latent_cache"].shape) == (
        BLOCKS_AT_ONE_BUFFER * REGISTERED_HYBRID_BLOCK_SIZE,
        1,
        head_size,
    )

    # MUST-FAIL ARM: a reader pointed at the half that used to exist raises, which
    # is the positive proof the dead buffer is gone rather than merely unused.
    with pytest.raises(IndexError):
        caches[name][1]


# T05 -- a layer that declares no latent cache is untouched.
def test_t05_a_non_latent_layer_still_gets_two_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration is the discriminator; the name and the geometry are not."""
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
    control_classes = sorted({type(control_specs[n]).__name__ for n in attention_names})

    _record(
        t05_control_classes=control_classes,
        t05_control_pages=sorted(
            {control_specs[n].page_size_bytes for n in attention_names}
        ),
        t05_control_buffers_per_layer=sorted(
            {len(control[n]) for n in attention_names}
        ),
        t05_control_k_shape=tuple(control[name][0].shape),
        t05_control_v_shape=tuple(control[name][1].shape),
    )
    assert all(type(control_specs[n]) is FullAttentionSpec for n in attention_names)
    assert {control_specs[n].page_size_bytes for n in attention_names} == {two_buffer}
    assert {len(control[n]) for n in attention_names} == {2}
    expected = (
        BLOCKS_AT_ONE_BUFFER // 2,
        1,
        REGISTERED_HYBRID_BLOCK_SIZE,
        head_size,
    )
    assert tuple(control[name][0].shape) == expected
    assert tuple(control[name][1].shape) == expected
    assert control[name][0].data_ptr() != control[name][1].data_ptr()

    # MUST-FAIL ARM: declaring a latent cache on those same layers flips them to
    # one buffer, so the field is proved to be what decides and not dead code.
    flipped_specs = _call(
        [replace(layer, **{LATENT_FIELD: True}) for layer in control_layers]
    )
    flipped = _drive(
        _fixed_budget_config(flipped_specs, budget),
        _FakeModel(control_layers),
        monkeypatch,
    )
    _record(
        t05_flipped_classes=sorted(
            {type(flipped_specs[n]).__name__ for n in attention_names}
        ),
        t05_flipped_buffers_per_layer=sorted(
            {len(flipped[n]) for n in attention_names}
        ),
    )
    assert all(isinstance(flipped_specs[n], MLAAttentionSpec) for n in attention_names)
    assert {len(flipped[n]) for n in attention_names} == {1}


# T06 -- unification still succeeds, and still refuses to narrow a state.
def test_t06_page_unification_survives_the_halved_page() -> None:
    """One page across all 45 entries, and the downward refusal is still live."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, attention_names = _split(specs)
    one_buffer, _ = _vendor_pages(raw)
    padded = sorted(
        name for name, spec in specs.items() if spec.page_size_padded is not None
    )

    _record(
        t06_distinct_pages=sorted({spec.page_size_bytes for spec in specs.values()}),
        t06_padded_names=len(padded),
        t06_recurrent_state_page=RECORDED_KDA_STATE_PAGE_BYTES,
        t06_attention_page=one_buffer,
    )
    assert {spec.page_size_bytes for spec in specs.values()} == {one_buffer}
    assert padded == sorted(kda_names)
    assert len(attention_names) == DECLARED_DSA_ENTRIES
    # The halved page still exceeds the state it must hold, which is why the
    # padding above still runs upward.
    assert RECORDED_KDA_STATE_PAGE_BYTES < one_buffer

    # MUST-FAIL ARM: a recurrent state larger than the halved page must be
    # REFUSED, not narrowed. Without this the item could not tell a live
    # unification from one that silently stopped checking.
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
