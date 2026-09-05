# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-017`` acceptance -- WP2: runner ``initialize_kv_cache``, hybrid.

THE DECLARED ACCEPTANCE, the plan block's command, verbatim:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      VLLM_SSM_CONV_STATE_LAYOUT=SD python -m pytest \\
      test/vllm_neuron/worker/test_initialize_kv_cache_hybrid.py -q --timeout 60 \\
      -p no:cacheprovider

Four counted conjuncts, ONE item each, no ``parametrize`` (section 6 rule 6).
Every parent reading below was MEASURED at the parent commit before a source
line was written (``../increments/probe-017-parent-readings.py`` at ``069863ee``),
so each arm's discrimination is a transcript and not a code reading:

* C01 -- allocation produces **45** entries FOR THE ELECTED KDA CLASS rather
  than raising. PARENT: ``NotImplementedError("Unsupported Attention spec type:
  ...MambaSpec...")`` raised at ``neuron_model_runner.py:8600``, **0** KDA
  buffers allocated -- the count is not reachable at all at the parent.
* C02 -- the **34** KDA entries allocate BOTH state buffers, the element counts
  being PRODUCTS of ``inc-glm53f-015``'s two landed shape fields' extents.
  PARENT: no conv buffer is allocated at all, the parent allocating from a
  single-tensor attention spec class.
* C03 -- the **11** DSA entries allocate ``head_size == 512``, measured against
  the MODEL's reported value. PARENT: **green, and declared green** -- this is a
  REGRESSION guard on the DSA half and certifies nothing about this increment.
* C04 -- total allocated bytes equals the sum of the PER-ENTRY PAGES THE SPEC
  OBJECTS REPORT, discrepancy **0 B**. PARENT: on 34 of 45 entries the sum
  carries no state page at all (the same raise as C01), a shortfall of
  **2,306,560 B** at one block per entry.

REFERENTS, so that no term here is referent-free. Every expected value is
measured against something the world reports and never against a number this
file chooses: the page against ``page_size_bytes`` READ OFF the spec objects
``get_kv_cache_spec`` constructs; the element counts against the two
``LayerSpec`` shape FIELDS; the DSA head size against the model's own reported
value. The one external literal is ``DECISIONS.md`` section 6's
recorded KDA state page, CITED and neither restated nor re-derived (P9).

VEHICLE, and why it is shared rather than rebuilt. The 45 ``LayerSpec`` objects,
the fake model and the spec dict come from the LANDED ``inc-glm53f-016``
acceptance module's own helpers. Rebuilding them here would create a SECOND
construction of the fake, free to drift from the one ``inc-glm53f-016`` declares
as ``inc-glm53f-038``'s specification; importing them keeps one construction,
and with it the property that every KDA field value is DERIVED by calling the
vendor authorities (``MambaStateShapeCalculator.kda_state_shape`` and the
SEPARATE ``MambaStateDtypeCalculator.kda_state_dtype``) at the registered
geometry rather than hand-written.

TWO STAGES, as the block's Tests bullet declares: a tiny layer count scaled from
the fixture's own schedule with REAL CPU tensors (where both state buffers are
written and read back), then a re-assertion at 45 layers with allocation
intercepted by a COUNTING instrument. That counting wrapper delegates to the
real allocator: the whole 45-layer footprint is ~10 MB at two blocks per entry,
so nothing needs substituting, and counting is what C04 needs from it.

ORIENTATION. The command pins the conv layout for DETERMINISM of the derived
fixture. The resolved layout is RECORDED, not asserted: every count here is a
byte total or a product of extents, hence transposition-invariant, and
``inc-glm53f-015``'s conjunct 3 remains the campaign's only orientation guard.
This file asserts no conv EXTENT.

WHAT THIS DOES NOT CERTIFY. Not the producer of the field values (that is
``inc-glm53f-038``, at M3), not the model-side consumption of the returned state
buffers, and no hardware behaviour: this is a CPU-mode, host-side allocation
certificate.
"""

from __future__ import annotations

import math
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

#: Two blocks per entry at the full stack: enough that a ``num_blocks`` stuck at
#: 1 cannot pass as the computed value, and ~10 MB in total.
NUM_BLOCKS_FULL = 2

#: The tiny stage -- the first eight layers of the fixture's OWN schedule, which
#: carries both families there (six KDA, two DSA), at three blocks per entry.
TINY_LAYER_COUNT = 8
NUM_BLOCKS_TINY = 3

#: PARENT readings, measured by ``probe-017-parent-readings.py`` at ``069863ee``.
PARENT_KDA_BUFFERS_ALLOCATED = 0
PARENT_RAISE_MESSAGE_FRAGMENT = "Unsupported Attention spec type"
PARENT_DSA_ALLOCATED_HEAD_SIZE = 512


def _record(**readings: object) -> None:
    """Put a reading in the ``-q`` transcript (``-075``'s convention)."""
    for key, value in readings.items():
        warnings.warn(f"RECORDED {key}={value!r}", UserWarning, stacklevel=2)


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


class _StubInputBatch:
    """``initialize_kv_cache`` builds one; no conjunct here measures it."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _drive(kv_cache_config, layers, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Call the UNBOUND ``initialize_kv_cache`` on CPU; no runner constructed.

    The two helpers on this increment's declared surface are the REAL ones,
    bound onto the fake self, so the FP8-packed decision and the K allocation
    shape are exercised rather than stubbed.
    """
    module = _runner_module()
    runner = module.NeuronModelRunner
    monkeypatch.setattr(module, "InputBatch", _StubInputBatch)
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)

    bound: list[dict] = []
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
        model=_FakeModel(layers),
        _kv_cache_full_tensors={},
    )
    fake.model.bind_kv_cache = bound.append
    fake._kv_cache_is_fp8_packed = MethodType(runner._kv_cache_is_fp8_packed, fake)
    fake._k_cache_alloc_shape = runner._k_cache_alloc_shape

    caches = runner.initialize_kv_cache(fake, kv_cache_config)
    # The dict handed to the model is the dict returned, so nothing below reads
    # a structure the model never sees.
    assert bound and bound[0] is caches
    return caches


def _config(specs: dict, num_blocks: int):
    """One ``KVCacheTensor`` per layer, each sized to that layer's OWN page.

    Groups are split on the SPEC CLASSES rather than on layer names.
    """
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
    """``(kda_names, dsa_names)``, read off the returned spec CLASSES."""
    from vllm.v1.kv_cache_interface import MambaSpec

    kda = [name for name, spec in specs.items() if isinstance(spec, MambaSpec)]
    dsa = [name for name, spec in specs.items() if not isinstance(spec, MambaSpec)]
    return kda, dsa


def _counting_zeros(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Intercept the raw-buffer allocation and COUNT it, then delegate.

    The block's Tests bullet asks the 45-layer stage to run with allocation
    "mocked to counting"; this is that mock. It delegates to the real allocator
    because the full footprint is ~10 MB, so the buffers stay real while every
    request is recorded.
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
    """Bytes the RETURNED buffers span -- what the allocator handed back."""
    return sum(
        buffer.numel() * buffer.element_size()
        for buffers in caches.values()
        for buffer in buffers
    )


def _model_reported_head_size(raw: dict) -> int:
    """``kv_lora_rank + qk_rope_head_dim`` off the fixture -- ``-013``'s surface."""
    text = raw["text_config"]
    return int(text["kv_lora_rank"]) + int(text["qk_rope_head_dim"])


# C01 -- 45 entries FOR THE ELECTED CLASS rather than a raise.
def test_initialize_kv_cache_c01_forty_five_entries_for_the_elected_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """45 allocated entries, the 34 KDA ones through the elected class's branch."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, dsa_names = _split(specs)
    _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    kda_shapes = [tuple(buffer.shape) for buffer in caches[kda_names[0]]]
    _record(
        c01_entries_allocated=len(caches),
        c01_kda_entries=len(kda_names),
        c01_dsa_entries=len(dsa_names),
        c01_kda_buffers_per_entry=sorted({len(caches[n]) for n in kda_names}),
        c01_dsa_buffers_per_entry=sorted({len(caches[n]) for n in dsa_names}),
        c01_kda_buffer_shapes=kda_shapes,
    )
    assert len(caches) == DECLARED_TOTAL_ENTRIES
    assert (len(kda_names), len(dsa_names)) == (
        DECLARED_KDA_ENTRIES,
        DECLARED_DSA_ENTRIES,
    )
    # Reachable at all, which the parent was not: it raised on the first KDA
    # entry and allocated 0 KDA buffers.
    assert all(len(caches[name]) == 2 for name in kda_names)
    assert sum(len(caches[n]) for n in kda_names) != PARENT_KDA_BUFFERS_ALLOCATED

    # CONTROL 1, the block's declared one -- the PRE-ELECTION class on the same
    # path returns the parent's ATTENTION allocation, so the arm reads WHICH
    # BRANCH ran and not how many entries came back.
    pre_election = {
        name: (
            FullAttentionSpec(
                block_size=REGISTERED_HYBRID_BLOCK_SIZE,
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
    control = _drive(_config(pre_election, NUM_BLOCKS_FULL), layers, monkeypatch)
    control_shapes = {tuple(control[name][0].shape) for name in kda_names}
    _record(
        c01_control_pre_election_entries=len(control),
        c01_control_kda_buffer_shapes=sorted(control_shapes),
        c01_control_kda_buffer_dims=sorted({len(s) for s in control_shapes}),
    )
    assert len(control) == DECLARED_TOTAL_ENTRIES
    # 4-D attention buffers, not this increment's two state buffers.
    assert {len(shape) for shape in control_shapes} == {4}
    assert control_shapes != {kda_shapes[0]}

    # CONTROL 2 -- the PARENT READING reproduced against THIS code: make the
    # elected class unmatchable at the branch and the same call raises the
    # parent's own NotImplementedError from the same else-arm.
    class _Unmatched:
        pass

    monkeypatch.setattr(_runner_module(), "MambaSpec", _Unmatched)
    with pytest.raises(NotImplementedError) as raised:
        _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)
    _record(c01_control_parent_raise=str(raised.value))
    assert PARENT_RAISE_MESSAGE_FRAGMENT in str(raised.value)


# C02 -- the 34 KDA entries allocate BOTH state buffers, counts from -015's fields.
def test_initialize_kv_cache_c02_both_state_buffers_with_declared_element_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two buffers per KDA entry; element counts are products of the two fields."""
    from vllm.v1.kv_cache_interface import MambaSpec

    raw = _raw_fixture()

    # ---- STAGE 1: tiny layer count, REAL CPU tensors. -----------------------
    tiny_layers = _fake_layers(raw)[:TINY_LAYER_COUNT]
    tiny_specs = _call(tiny_layers)
    tiny_kda, tiny_dsa = _split(tiny_specs)
    tiny = _drive(_config(tiny_specs, NUM_BLOCKS_TINY), tiny_layers, monkeypatch)
    by_name = {layer.name: layer for layer in tiny_layers}
    _record(
        c02_tiny_entries=len(tiny),
        c02_tiny_kda=len(tiny_kda),
        c02_tiny_dsa=len(tiny_dsa),
        c02_tiny_num_blocks=NUM_BLOCKS_TINY,
    )
    assert len(tiny) == TINY_LAYER_COUNT
    assert tiny_kda and tiny_dsa

    for name in tiny_kda:
        layer = by_name[name]
        # The referents: -015's two landed SHAPE fields, whose values were
        # derived from the shape authority. Counts are PRODUCTS of extents, so
        # nothing here reads an orientation.
        conv_elements = math.prod(layer.kda_conv_state_shape)
        recurrent_elements = math.prod(layer.kda_recurrent_state_shape)
        assert len(tiny[name]) == 2
        conv_buffer, recurrent_buffer = tiny[name]
        assert conv_buffer.numel() == NUM_BLOCKS_TINY * conv_elements
        assert recurrent_buffer.numel() == NUM_BLOCKS_TINY * recurrent_elements
        assert conv_buffer.dtype is layer.kda_conv_state_dtype
        assert recurrent_buffer.dtype is layer.kda_recurrent_state_dtype
        assert conv_buffer.shape[0] == NUM_BLOCKS_TINY
        assert recurrent_buffer.shape[0] == NUM_BLOCKS_TINY

    # The two states must not overlap INSIDE the page: write one, read the other
    # back. A wrong storage offset shows here as a corrupted read.
    conv_buffer, recurrent_buffer = tiny[tiny_kda[0]]
    conv_buffer.fill_(1)
    recurrent_buffer.fill_(2)
    _record(
        c02_conv_values_after_recurrent_write=sorted(
            {float(v) for v in conv_buffer.flatten().tolist()}
        ),
        c02_recurrent_values_after_write=sorted(
            {float(v) for v in recurrent_buffer.flatten().tolist()}
        ),
    )
    assert torch.all(conv_buffer == 1)
    assert torch.all(recurrent_buffer == 2)
    # Distinct blocks are distinct storage, so a collapsed block stride shows.
    for block in range(NUM_BLOCKS_TINY):
        conv_buffer[block].fill_(block + 3)
    per_block = [float(conv_buffer[b].flatten()[0]) for b in range(NUM_BLOCKS_TINY)]
    _record(c02_conv_first_element_per_block=per_block)
    assert per_block == [float(b + 3) for b in range(NUM_BLOCKS_TINY)]

    # ---- STAGE 2: re-asserted at 45 layers, allocation counted. -------------
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, _ = _split(specs)
    requested = _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)
    full_by_name = {layer.name: layer for layer in layers}
    two_buffer_entries = [name for name in kda_names if len(caches[name]) == 2]
    element_counts = {
        (
            caches[n][0].numel() // NUM_BLOCKS_FULL,
            caches[n][1].numel() // NUM_BLOCKS_FULL,
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
    _record(
        c02_full_kda_entries_with_two_buffers=len(two_buffer_entries),
        c02_full_per_block_element_counts=sorted(element_counts),
        c02_full_expected_element_counts=sorted(expected_counts),
        c02_counting_mock_calls=len(requested),
    )
    assert len(two_buffer_entries) == DECLARED_KDA_ENTRIES
    assert element_counts == expected_counts
    # The counting instrument saw every raw buffer, so its scope is measured.
    assert len(requested) == DECLARED_TOTAL_ENTRIES

    # CONTROL (D1.5) -- drop the conv carrier and the allocated buffer count for
    # those entries falls, so the arm reads BOTH buffers and not only the larger.
    # It is built at the ALLOCATOR's boundary because the landed
    # get_kv_cache_spec REFUSES a partial geometry (measured just below), so a
    # conv field zeroed on the fixture never reaches this method at all.
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
    control_counts = sorted({len(control[name]) for name in kda_names})
    _record(c02_control_buffers_per_kda_entry=control_counts)
    assert control_counts == [1]

    # The disclosure above, measured rather than asserted in prose.
    partial = [
        replace(layer, kda_conv_state_shape=None)
        if layer.name in set(kda_names)
        else layer
        for layer in layers
    ]
    with pytest.raises(ValueError) as refused:
        _call(partial)
    _record(c02_partial_geometry_refused=str(refused.value)[:80])


# C03 -- the 11 DSA entries allocate head_size == the model's reported value.
def test_initialize_kv_cache_c03_dsa_entries_allocate_the_reported_head_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REGRESSION guard on the DSA half: parent-green, and declared so."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, dsa_names = _split(specs)
    _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    by_name = {layer.name: layer for layer in layers}
    reported = {by_name[name].head_size for name in dsa_names}
    allocated = {int(caches[name][1].shape[-1]) for name in dsa_names}
    k_last = {int(caches[name][0].shape[-1]) for name in dsa_names}
    config_value = _model_reported_head_size(raw)
    _record(
        c03_dsa_entries=len(dsa_names),
        c03_model_reported_head_size=sorted(reported),
        c03_allocated_head_size=sorted(allocated),
        c03_allocated_k_last_dim=sorted(k_last),
        c03_config_kv_lora_rank_plus_rope=config_value,
        c03_parent_reading=PARENT_DSA_ALLOCATED_HEAD_SIZE,
    )
    assert len(dsa_names) == DECLARED_DSA_ENTRIES
    # A RELATION, not a constant: the allocation matches what the model reports,
    # which matches the configured kv_lora_rank (+ rope) it is derived from.
    # Re-registering the geometry moves all three together.
    assert len(reported) == 1
    assert allocated == reported
    assert k_last == reported
    assert reported == {config_value}
    # The KDA work moved neither half's buffer count.
    assert all(len(caches[name]) == 2 for name in dsa_names)
    assert all(len(caches[name]) == 2 for name in kda_names)


# C04 -- total allocated bytes == sum of the spec-reported pages, 0 B apart.
def test_initialize_kv_cache_c04_total_bytes_reconcile_with_zero_discrepancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allocation total against the pages the SPEC OBJECTS report."""
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, dsa_names = _split(specs)
    requested = _counting_zeros(monkeypatch)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    # The RAW referent: page_size_bytes off the spec objects get_kv_cache_spec
    # constructs, times the blocks. -086 pads it, so this is the padded span.
    expected_bytes = sum(
        spec.page_size_bytes * NUM_BLOCKS_FULL for spec in specs.values()
    )
    allocated = _allocated_bytes(caches)
    kda_pages = sorted({specs[n].page_size_bytes for n in kda_names})
    dsa_pages = sorted({specs[n].page_size_bytes for n in dsa_names})
    _record(
        c04_num_blocks=NUM_BLOCKS_FULL,
        c04_allocated_bytes=allocated,
        c04_sum_of_spec_reported_pages=expected_bytes,
        c04_discrepancy_bytes=allocated - expected_bytes,
        c04_kda_page_size_bytes=kda_pages,
        c04_dsa_page_size_bytes=dsa_pages,
        c04_recorded_kda_page=RECORDED_KDA_STATE_PAGE_BYTES,
        c04_counting_mock_total_bytes=sum(requested),
        c04_counting_mock_calls=len(requested),
    )
    assert len(caches) == DECLARED_TOTAL_ENTRIES
    assert allocated - _addressable_bytes(specs, NUM_BLOCKS_FULL) == 0

    # The KDA page's own referent stays DECISIONS section 6's recorded page (P9).
    # -086 pads page_size_bytes, so the state's own geometry answers this now.
    assert _kda_natural_pages(specs, kda_names) == [RECORDED_KDA_STATE_PAGE_BYTES]

    # Per entry too, so a compensating pair of errors cannot net to zero.
    per_entry = {
        name: sum(b.numel() * b.element_size() for b in buffers)
        - _addressable_page_bytes(specs[name]) * NUM_BLOCKS_FULL
        for name, buffers in caches.items()
    }
    _record(
        c04_entries_off_by_any_bytes=sorted(
            name for name, delta in per_entry.items() if delta != 0
        )
    )
    assert set(per_entry.values()) == {0}

    # UNMOVED BY -086. Both sides read page_size_bytes: this file's config sizes
    # each raw tensor from that page, and the counting mock records what was
    # asked for, so padding raises the two together. The buffers the allocator
    # RETURNED span less, which is why the addressable zero above re-pinned.
    assert sum(requested) == expected_bytes

    # READINGS inherited from inc-glm53f-016's construction, RECORDED here and
    # adding no criterion: (a) which entries now carry page_size_padded, (b) both
    # carriers are present, so the vendor's non-strict pairing cannot have
    # truncated the sum this arm reconciles against.
    padded = {name: spec.page_size_padded for name, spec in specs.items()}
    arities = {(len(specs[n].shapes), len(specs[n].dtypes)) for n in kda_names}
    _record(
        c04_reading_a_non_none=sorted(n for n, v in padded.items() if v is not None),
        c04_reading_b_arities=sorted(arities),
        c04_parent_shortfall_bytes_at_one_block=sum(
            specs[n].page_size_bytes for n in kda_names
        ),
        c04_dsa_entries=len(dsa_names),
    )
    assert _non_none(padded) == _padded_page_expected_for_kda(specs)
    assert arities == {(2, 2)}

    # No conv EXTENT is asserted anywhere above, so the layout term is RECORDED
    # and -015's conjunct 3 stays the campaign's orientation guard.
    _record(c04_resolved_conv_state_layout=get_conv_state_layout())

    # ------------------------------------------------------------------
    # `inc-glm53f-086`: the readings the pad makes measurable, and the two
    # D1.5 controls for the zeros this increment re-pinned. All of it sits
    # BELOW every line this file is pinned from, so the re-pins above moved
    # nothing.
    # ------------------------------------------------------------------
    addressable = _addressable_bytes(specs, NUM_BLOCKS_FULL)
    pad_bytes = sum(requested) - allocated
    storage_span = sum(
        max(buffer.untyped_storage().nbytes() for buffer in buffers)
        for buffers in caches.values()
    )
    _record(
        c04_addressable_bytes=addressable,
        c04_pad_bytes_allocated_unaddressable=pad_bytes,
        c04_kda_natural_page_bytes=_kda_natural_pages(specs, kda_names),
        c04_kda_natural_sum_at_one_block=sum(
            _addressable_page_bytes(specs[name]) for name in kda_names
        ),
        c04_summed_max_storage_span_per_entry=storage_span,
    )

    # D1.5 CONTROL, re-pinned side. The addressable zero is only worth reading
    # if the WRONG referent moves it, so the same allocated total is taken
    # against the PADDED span, and it has to come out non-zero. How far it
    # misses by is RECORDED above and asserted nowhere.
    assert allocated - expected_bytes != 0

    # D1.5 CONTROL, raw side. The requested-bytes equality above is a zero that
    # a config sized off any other page would move, so this re-drives the same
    # allocator with the KDA tensors at TWICE the padded page and reads how far
    # the total travels. A natural-page config is NOT usable as this control: it
    # fails the runner's divisibility guard and never reaches the equality.
    doubled = _counting_zeros(monkeypatch)
    _drive(_config_doubled_kda(specs, NUM_BLOCKS_FULL), layers, monkeypatch)
    control_delta = len(kda_names) * NUM_BLOCKS_FULL * MEASURED_PADDED_PAGE_BYTES
    _record(
        c04_control_doubled_requested_bytes=sum(doubled),
        c04_control_doubled_minus_expected=sum(doubled) - expected_bytes,
        c04_control_expected_delta=control_delta,
    )
    assert sum(doubled) - expected_bytes == control_delta


# ===========================================================================
# `inc-glm53f-086` HELPERS, placed BELOW the tests deliberately. Three places
# pin the reading block at the tail of C04 by line number, and a name defined
# down here still resolves inside a test body, because that body runs after
# this module is imported. So the four re-pins above cost zero line movement.
# Two short helpers are repeated from `test_get_kv_cache_spec_hybrid.py`
# rather than imported, because a new name in the import block at the top of
# this file would move every line below it.
# ===========================================================================

#: The unified page every KDA entry now reports. MEASURED at round 1, read from
#: `probe-086-r1-landed-diagnostic.out` (`KDA_page_size_padded_DISTINCT`).
MEASURED_PADDED_PAGE_BYTES = 262_144


def _addressable_page_bytes(spec) -> int:
    """Bytes of one page that an allocated buffer can actually reach.

    A recurrent state occupies only its own geometry, and ``-086`` pads the page
    it REPORTS up to the attention page. The allocation arm packs both states at
    the front of the page and makes the block stride the whole page, so a
    returned view spans the geometry and never the pad. An attention page has no
    pad, so ``page_size_bytes`` is already the addressable answer there.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    if not isinstance(spec, MambaSpec):
        return spec.page_size_bytes
    total = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        elements = 1
        for extent in shape:
            elements *= extent
        total += elements * dtype.itemsize
    return total


def _addressable_bytes(specs: dict, num_blocks: int) -> int:
    """Every entry's addressable page, times the blocks per entry."""
    return sum(_addressable_page_bytes(spec) * num_blocks for spec in specs.values())


def _kda_natural_pages(specs: dict, kda_names: list) -> list:
    """The DISTINCT addressable page across the KDA entries, sorted."""
    return sorted({_addressable_page_bytes(specs[name]) for name in kda_names})


def _non_none(mapping: dict) -> dict:
    """The entries whose value is set, so an empty result reads as ``{}``."""
    return {name: value for name, value in mapping.items() if value is not None}


def _padded_page_expected_for_kda(specs: dict) -> dict:
    """Every KDA name mapped to the unified page, and no other name present."""
    from vllm.v1.kv_cache_interface import MambaSpec

    return {
        name: MEASURED_PADDED_PAGE_BYTES
        for name, spec in specs.items()
        if isinstance(spec, MambaSpec)
    }


def _config_doubled_kda(specs: dict, num_blocks: int):
    """``_config``, with every KDA raw tensor sized at TWICE the padded page."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
    )

    tensors = []
    for name, spec in specs.items():
        factor = 2 if isinstance(spec, MambaSpec) else 1
        tensors.append(
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks * factor, shared_by=[name]
            )
        )
    groups = [
        KVCacheGroupSpec(layer_names=names, kv_cache_spec=specs[names[0]])
        for names in _split(specs)
        if names
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
    )
