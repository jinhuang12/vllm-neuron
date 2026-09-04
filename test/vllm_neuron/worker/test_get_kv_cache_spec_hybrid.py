# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-016`` acceptance -- WP2: runner ``get_kv_cache_spec``, hybrid stack.

THE DECLARED ACCEPTANCE, the plan block's command, verbatim:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      VLLM_SSM_CONV_STATE_LAYOUT=SD python -m pytest \\
      test/vllm_neuron/worker/test_get_kv_cache_spec_hybrid.py -q --timeout 60 \\
      -p no:cacheprovider

Four counted conjuncts, ONE item each, no ``parametrize`` (section 6 rule 6).
Each carries the block's measured PARENT reading, so the discrimination is
checkable rather than hoped: C01 split **34 MambaSpec / 11 FullAttentionSpec / 0
other** (parent **0/45/0**, control name-blind); C02 **0 of 45** entries whose
returned dtype carrier differs from what the MODEL reports for that state
(parent **34**, control reverts this increment's own per-entry assignment to the
global, **0 -> 34**); C03 the four ``inc-glm53f-015`` fields ARE READ, **34**
engaged / **0** engaged (parent **0/0**); C04 the KDA page reconciles to the
recorded state page, discrepancy **0 B**, READ off the instrument and RECORDED
(parent **65,536 B**), with readings **(a)** ``page_size_padded is None`` on
every constructed object and **(b)** ``len(shapes) == len(dtypes) == 2`` on all
34.

SCOPE. This certifies the runner's TRANSLATION -- ``LayerSpec`` in, spec-dict out
-- and nothing about the producer of the values. The vehicle is a fake model
exposing only ``get_kv_spec()``, and that fake IS the specification
``inc-glm53f-038`` must satisfy; real field values are M3's.

NOT SELF-REFERENTIAL. Every KDA field value is DERIVED by calling the vendor
authorities at the registered geometry, never hand-written: shapes from
``MambaStateShapeCalculator.kda_state_shape``, dtypes from
``MambaStateDtypeCalculator.kda_state_dtype`` -- **two separate classes**,
conflating them raises ``AttributeError``. The roster, family schedule and
per-layer geometry come from the digest-pinned 45-layer fixture through the
landed ``inc-glm53f-013`` skeleton. This is ``-015`` conjunct 3's pattern.

ORIENTATION. The command pins the conv layout for DETERMINISM of the derived
fake. The resolved layout is RECORDED, not asserted: every count here is a
product of extents and so transposition-invariant, and ``-015``'s conjunct 3
remains the campaign's only orientation guard.

WHAT C02'S ZERO DOES NOT COVER. Its DSA half agrees because the registered cache
dtype ``"auto"`` resolves to the model's own dtype. The attention branch keeps
the GLOBAL KV cache dtype deliberately -- that dtype describes a key/value cache
and is the knob an fp8-KV election would move, which is a comparator question,
not this file's. Only the recurrent-state branch reads per-state dtypes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.kv_cache import LayerSpec

#: The landed 45-layer instrument, same digest ``test_kv_spec.py:99`` pins.
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "glm5_next"
    / "fixtures"
    / "config.json"
)
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

#: ``DECISIONS.md`` section 6: registered TP, the user-decided block
#: size, and the recorded KDA state page. All CITED, none re-derived (P9).
REGISTERED_TP_WORLD_SIZE = 64
REGISTERED_HYBRID_BLOCK_SIZE = 128
RECORDED_KDA_STATE_PAGE_BYTES = 67_840

DECLARED_TOTAL_ENTRIES = 45
DECLARED_KDA_ENTRIES = 34
DECLARED_DSA_ENTRIES = 11

#: The block's measured PARENT readings, named here so an arm that stops
#: discriminating is visible in this file and not only in the plan.
PARENT_KDA_SPEC_ENTRIES = 0
PARENT_DTYPE_MISMATCHES = 34
PARENT_ENGAGED_ENTRIES = 0


def _record(**readings: object) -> None:
    """Put a reading in the ``-q`` transcript (``-075``'s convention)."""
    for key, value in readings.items():
        warnings.warn(f"RECORDED {key}={value!r}", UserWarning, stacklevel=2)


def _authority_state_shapes() -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """``(conv, recurrent, resolved_layout)`` from vLLM's SHAPE calculator."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        get_conv_state_layout,
    )

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    linear_attn = Glm5NextTextConfig().linear_attn_config
    conv, recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=REGISTERED_TP_WORLD_SIZE,
        num_heads=linear_attn["num_heads"],
        head_dim=linear_attn["head_dim"],
        conv_kernel_size=linear_attn["short_conv_kernel_size"],
    )
    return tuple(conv), tuple(recurrent), get_conv_state_layout()


def _authority_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    """``(conv_dtype, recurrent_dtype)`` from vLLM's DTYPE calculator.

    A SEPARATE class from the shape calculator above.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    # bf16 model dtype is section 6's registered KV precondition; "auto" is the
    # cache-dtype form that follows it.
    return MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")


def _raw_fixture() -> dict:
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256
    return json.loads(FIXTURE_PATH.read_text())


def _fake_layers(
    raw: dict, *, populate_kda: bool = True, name_blind: bool = False
) -> list[LayerSpec]:
    """The fake model's 45 layers, derived end to end.

    ``populate_kda=False`` CLEARS the four fields to ``None`` (C03's other half) --
    it clears rather than inherits, because since ``inc-glm53f-038a`` the real
    model's own spec carries real values on the 34 linear-attention layers;
    ``name_blind=True`` strips every family suffix (C01's control) -- which is
    why the family is read off the fixture's own schedule and never off a name.
    """
    from vllm_neuron.model.glm5_next.config import KDA_LAYER_TYPE
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    conv_shape, recurrent_shape, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    kda = {
        index
        for index, family in enumerate(raw["text_config"]["layer_types"])
        if family == KDA_LAYER_TYPE
    }
    landed = Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))

    layers: list[LayerSpec] = []
    for index, layer in enumerate(landed.get_kv_spec().layers):
        name = f"layers.{index}" if name_blind else layer.name
        if index in kda and populate_kda:
            layers.append(
                replace(
                    layer,
                    name=name,
                    kda_conv_state_shape=conv_shape,
                    kda_recurrent_state_shape=recurrent_shape,
                    kda_conv_state_dtype=conv_dtype,
                    kda_recurrent_state_dtype=recurrent_dtype,
                )
            )
        else:
            # CLEARED EXPLICITLY, never inherited. `inc-glm53f-038a` filled these
            # four fields on the real model's own spec, so a bare
            # `replace(layer, name=name)` started PRESERVING real values on the 34
            # linear-attention layers -- and C03's "fields are unset" control below
            # quietly became a second populated case, which is the regression
            # `B36-F2` found. A control has to BUILD the absence it claims.
            #
            # No other caller is affected: every other call takes the default
            # `populate_kda=True`, where the 34 KDA layers go through the branch
            # above and the 11 MLA layers arrive here already carrying `None`, so
            # clearing them is a no-op. Measured, not assumed -- the C01/C02/C04
            # readings are unchanged by this edit.
            layers.append(
                replace(
                    layer,
                    name=name,
                    kda_conv_state_shape=None,
                    kda_recurrent_state_shape=None,
                    kda_conv_state_dtype=None,
                    kda_recurrent_state_dtype=None,
                )
            )
    return layers


class _FakeModel:
    """Exposes ONLY ``get_kv_spec()``, as the block's Tests bullet requires."""

    def __init__(self, layers: list[LayerSpec]) -> None:
        self._spec = SimpleNamespace(layers=layers)

    def get_kv_spec(self):
        return self._spec


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


def _call(layers: list[LayerSpec]) -> dict:
    """Drive the UNBOUND method, so no real runner is constructed."""
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=REGISTERED_HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        speculative_config=None,
        model=_FakeModel(layers),
    )
    return _runner_module().NeuronModelRunner.get_kv_cache_spec(fake_self)


def _dtype_mismatches(specs: dict, layers: list[LayerSpec]) -> list[str]:
    """Entries whose returned dtype carrier differs from the model's.

    The elected recurrent-state class has no ``dtype`` field -- its carrier is
    the ``dtypes`` TUPLE; the pin's attention class carries a single ``dtype``.
    Both normalise to a tuple, compared element for element in the construction
    bullet's declared order (position 0 conv, position 1 recurrent), so an ARITY
    difference counts as a difference and a truncated tuple cannot pass as a
    match.
    """
    mismatched: list[str] = []
    for layer in layers:
        spec = specs[layer.name]
        returned = tuple(spec.dtypes) if hasattr(spec, "dtypes") else (spec.dtype,)
        if layer.kda_recurrent_state_shape is not None:
            reported = (layer.kda_conv_state_dtype, layer.kda_recurrent_state_dtype)
        else:
            reported = (layer.dtype,)
        if len(returned) != len(reported) or any(
            got is not want for got, want in zip(returned, reported, strict=True)
        ):
            mismatched.append(layer.name)
    return mismatched


# C01 -- the split is a property of the RETURNED DICT: 34 / 11 / 0.
def test_get_kv_cache_spec_c01_split_is_thirty_four_mamba_eleven_attention() -> None:
    """34 recurrent-state specs, 11 attention specs, 0 other classes."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    specs = _call(_fake_layers(raw))

    assert len(specs) == DECLARED_TOTAL_ENTRIES
    kda = sum(1 for spec in specs.values() if isinstance(spec, MambaSpec))
    dsa = sum(1 for spec in specs.values() if isinstance(spec, FullAttentionSpec))
    other = len(specs) - kda - dsa
    _record(
        c01_class_census=sorted({type(s).__name__ for s in specs.values()}),
        c01_split=(kda, dsa, other),
    )
    assert (kda, dsa, other) == (DECLARED_KDA_ENTRIES, DECLARED_DSA_ENTRIES, 0)
    # The parent read 0 / 45 / 0, so the pre-election behaviour cannot satisfy it.
    assert kda != PARENT_KDA_SPEC_ENTRIES

    # CONTROL -- NAME-BLIND: every family suffix stripped, split unchanged,
    # because the branch selects on a LayerSpec FIELD. At the parent the same
    # control leaves all 45 entries attention specs.
    blind_layers = _fake_layers(raw, name_blind=True)
    assert not any(layer.name.endswith((".linear_attn", ".self_attn")) for layer in blind_layers)
    blind = _call(blind_layers)
    blind_kda = sum(1 for spec in blind.values() if isinstance(spec, MambaSpec))
    _record(c01_name_blind_split=(blind_kda, len(blind) - blind_kda))
    assert (blind_kda, len(blind) - blind_kda) == (
        DECLARED_KDA_ENTRIES,
        DECLARED_DSA_ENTRIES,
    )


# C02 -- per-entry dtype comes from the model: 0 differing of 45.
def test_get_kv_cache_spec_c02_zero_of_forty_five_dtypes_differ_from_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every entry's dtype carrier equals what the model reports for that state."""
    from vllm.v1.kv_cache_interface import MambaSpec

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    mismatched = _dtype_mismatches(specs, layers)
    _record(
        c02_mismatch_count=len(mismatched),
        c02_denominator=len(specs),
        c02_kda_carrier=[str(d) for d in specs[layers[0].name].dtypes],
    )
    assert len(specs) == DECLARED_TOTAL_ENTRIES
    assert mismatched == []

    # The recurrent half is per-STATE, not one dtype repeated: the two authority
    # dtypes differ, so a single-dtype assignment could not have passed above.
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    assert conv_dtype is not recurrent_dtype

    # CONTROL (D1.5) -- revert THIS increment's per-entry assignment to the
    # global, on this same fake, by intercepting the construction the branch
    # performs. The mismatch count must move 0 -> 34 (the parent's own reading)
    # while C01 still holds, so what moves is this conjunct's own counted value.
    from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype

    global_dtype = kv_cache_dtype_str_to_dtype(
        "auto", SimpleNamespace(dtype=torch.bfloat16)
    )

    def _reverted_to_global(**kwargs):
        kwargs["dtypes"] = tuple(global_dtype for _ in kwargs["shapes"])
        return MambaSpec(**kwargs)

    monkeypatch.setattr(_runner_module(), "MambaSpec", _reverted_to_global)
    reverted = _call(layers)
    reverted_kda = sum(1 for spec in reverted.values() if isinstance(spec, MambaSpec))
    reverted_mismatches = _dtype_mismatches(reverted, layers)
    _record(
        c02_control_mismatch_count=len(reverted_mismatches),
        c02_control_kda_entries=reverted_kda,
    )
    assert len(reverted_mismatches) == PARENT_DTYPE_MISMATCHES
    assert reverted_kda == DECLARED_KDA_ENTRIES


# C03 -- the four -015 fields are READ: 34 engaged / 0 engaged.
def test_get_kv_cache_spec_c03_the_four_state_fields_are_read() -> None:
    """A counted differential on the four landed fields: 34 engaged, 0 engaged."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    populated = _call(_fake_layers(raw, populate_kda=True))
    engaged = sum(1 for spec in populated.values() if isinstance(spec, MambaSpec))

    # Same fake, four fields None: the branch must not fire and the dict must
    # return the parent's shape -- 45 attention specs.
    emptied_layers = _fake_layers(raw, populate_kda=False)
    assert all(
        (
            layer.kda_conv_state_shape,
            layer.kda_recurrent_state_shape,
            layer.kda_conv_state_dtype,
            layer.kda_recurrent_state_dtype,
        )
        == (None, None, None, None)
        for layer in emptied_layers
    )
    emptied = _call(emptied_layers)
    not_engaged = sum(1 for spec in emptied.values() if isinstance(spec, MambaSpec))

    # WHY THE CONTROL HAS TO CLEAR RATHER THAN INHERIT, read off the real model.
    # This number was 0 before `inc-glm53f-038a` and is 34 after it. The arm used
    # to inherit these fields and assume they were absent, which is exactly the
    # regression `B36-F2` found; recording the number here means a future change
    # to it is visible in the transcript instead of turning the control hollow.
    from vllm_neuron.model.glm5_next.model_fp8 import (
        Glm5NextForConditionalGeneration,
    )

    parent_populated = sum(
        1
        for layer in Glm5NextForConditionalGeneration.from_configs(
            copy.deepcopy(raw)
        ).get_kv_spec().layers
        if layer.kda_conv_state_shape is not None
    )
    _record(c03_parent_layers_carrying_kda_fields=parent_populated)
    assert parent_populated == DECLARED_KDA_ENTRIES, (
        f"the real spec carries KDA fields on {parent_populated} layers, so the "
        f"control below must clear them explicitly rather than inherit"
    )

    _record(c03_engaged=engaged, c03_not_engaged=not_engaged)
    assert (engaged, not_engaged) == (DECLARED_KDA_ENTRIES, PARENT_ENGAGED_ENTRIES)
    assert len(emptied) == DECLARED_TOTAL_ENTRIES
    assert all(isinstance(spec, FullAttentionSpec) for spec in emptied.values())


# C04 -- the page reconciles to the recorded page, discrepancy 0 B.
def test_get_kv_cache_spec_c04_kda_page_reconciles_with_zero_discrepancy() -> None:
    """The elected class's own page, read off the instrument, hits the record."""
    from vllm.v1.kv_cache_interface import MambaSpec

    specs = _call(_fake_layers(_raw_fixture()))
    _, _, resolved_layout = _authority_state_shapes()
    # RECORDED, not asserted: the counts are transposition-invariant and -015's
    # conjunct 3 is the campaign's only orientation guard.
    _record(c04_resolved_conv_state_layout=resolved_layout)

    kda_specs = [spec for spec in specs.values() if isinstance(spec, MambaSpec)]
    assert len(kda_specs) == DECLARED_KDA_ENTRIES

    # READ off the instrument, then RECORDED -- D1.3's preferred form.
    pages = {spec.page_size_bytes for spec in kda_specs}
    _record(
        c04_kda_page_size_bytes=sorted(pages),
        c04_recorded_page=RECORDED_KDA_STATE_PAGE_BYTES,
        c04_shapes=[tuple(shape) for shape in kda_specs[0].shapes],
        c04_dtypes=[str(dtype) for dtype in kda_specs[0].dtypes],
    )
    assert len(pages) == 1
    assert sorted(_natural_state_pages(kda_specs)) == [RECORDED_KDA_STATE_PAGE_BYTES]

    # READING (a) -- page_size_padded now carries the unified page on the 34 KDA
    # entries and stays None on the 11 attention ones (`inc-glm53f-086`).
    padded = {name: spec.page_size_padded for name, spec in specs.items()}
    _record(
        c04_reading_a_non_none=sorted(n for n, v in padded.items() if v is not None)
    )
    assert len(padded) == DECLARED_TOTAL_ENTRIES
    assert _non_none(padded) == _padded_page_expected_for_kda(specs)

    # READING (b) -- both carriers length 2 on all 34, so the vendor's
    # strict-less pairing cannot truncate the sum in silence.
    arities = {(len(spec.shapes), len(spec.dtypes)) for spec in kda_specs}
    _record(c04_reading_b_arities=sorted(arities))
    assert arities == {(2, 2)}

    # -086's READINGS, recorded and adding no criterion: the two pages side by
    # side, so the natural-page assert above cannot be read as a tautology.
    _record(
        c086_kda_natural_page_bytes=sorted(_natural_state_pages(kda_specs)),
        c086_kda_page_size_bytes=sorted(pages),
        c086_padded_field_entries_set=len(_non_none(padded)),
    )

    # D1.5 CONTROL for the natural-page zero above: the page the spec REPORTS is
    # a different number now, so that zero discriminates and is no tautology.
    assert sorted(pages) != sorted(_natural_state_pages(kda_specs))


# ===========================================================================
# `inc-glm53f-086` HELPERS. They sit BELOW the tests on purpose. Every pin
# into this file cites a line above C04, and a name defined down here still
# resolves inside a test body, because that body runs after the module is
# imported. So the two re-pins above cost zero line movement.
# ===========================================================================

#: The unified page every KDA entry now reports. MEASURED at round 1, read from
#: `probe-086-r1-landed-diagnostic.out` (`KDA_page_size_padded_DISTINCT`).
MEASURED_PADDED_PAGE_BYTES = 262_144


def _natural_state_pages(kda_specs: list) -> set:
    """The bytes each recurrent state's OWN geometry occupies.

    ``page_size_bytes`` stopped answering this question at ``-086``, which pads
    it up to the attention page, so the state's own size is summed from the
    shapes and dtypes the spec carries. The product is spelt out rather than
    imported from ``math``, because a new import at the top of this file would
    move every line below it and four places pin lines in it.
    """
    pages = set()
    for spec in kda_specs:
        total = 0
        for shape, dtype in zip(spec.shapes, spec.dtypes):
            elements = 1
            for extent in shape:
                elements *= extent
            total += elements * dtype.itemsize
        pages.add(total)
    return pages


def _non_none(mapping: dict) -> dict:
    """The entries whose value is set, so an empty result reads as ``{}``."""
    return {name: value for name, value in mapping.items() if value is not None}


def _padded_page_expected_for_kda(specs: dict) -> dict:
    """Every KDA name mapped to the unified page, and no other name present.

    ``MambaSpec`` is read from its OWN module and never off the runner module,
    whose name C02 above replaces with a factory function.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    return {
        name: MEASURED_PADDED_PAGE_BYTES
        for name, spec in specs.items()
        if isinstance(spec, MambaSpec)
    }
