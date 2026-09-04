# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-013`` acceptance -- WP1: model skeleton and ``KVSpec``.

THE DECLARED ACCEPTANCE (increment plan revision 12, L3589), verbatim:

    "``get_kv_spec()`` returns a ``KVSpec`` whose ``layers`` list has
    **exactly 45** ``LayerSpec`` entries, of which **11** carry MLA/DSA
    geometry (``head_size == 512`` from ``kv_lora_rank``) and **34** carry KDA
    geometry; **0** entries have ``dtype is None``."

Four counted conjuncts, measured below as C01 / C02 / C03 / C04. The plan
declares no exit code for this block and no collected-item count, so neither
is claimed here: the exit code is the default and the item count is reported
as measured.

WHY EVERY IMPORT OF THE MODULE UNDER TEST IS DEFERRED
-----------------------------------------------------
``test_factory.py:318-319`` is a **landed passing assertion** that
``vllm_neuron.model.glm5_next.model_fp8`` is *not* in ``sys.modules`` -- it is
what proves ``glm5_next/factory.py:340``'s lazy import stays lazy. pytest imports **every
collected test module during collection, before it runs any test**, so a
module-level ``import model_fp8`` in this file would put the module in
``sys.modules`` before ``test_factory.py``'s test body ever runs and would
break that landed test -- and it would do so regardless of how the filenames
sort. Alphabetical ordering protects the *execution* order, not the *import*
order. So this module imports the implementation only inside test bodies and
fixtures, via :func:`_impl`, and the filename stays ``test_kv_spec.py`` so it
also executes after ``test_factory.py``.

WHAT C03 DOES AND DOES NOT CERTIFY -- STATED, NOT GLOSSED
---------------------------------------------------------
C03 names no field and no value where C02 names both, so it is adjudicable
only by **complement**: the 34 are the entries that are not the MLA/DSA 11,
and by C01's total the count is arithmetic. So the 34 entries are certified as
*not MLA/DSA, present, and carrying a non-``None`` dtype*.

They are still **not** certified as describing a correct KDA cache. That
conclusion is unchanged; only its ground has moved. When this file landed the
ground was that the pin's ``LayerSpec`` had no vocabulary for a recurrent state
at all -- **history now**, true at ``-013`` and ended by ``inc-glm53f-015``,
which appended the four KDA state fields. The ground today is that the
vocabulary exists but **nothing in this file populates or certifies its
contents**: filling those fields per layer is ``-016``'s and ``-017``'s work,
so the coverage limit survives the widening instead of evaporating with it.
C03's **34** does not move -- the widening is additive with ``None`` defaults,
so no entry's geometry split, ``head_size`` or ``dtype`` changes.

:func:`test_kv_spec_the_pin_dataclass_is_not_widened` is **superseded** by that
increment and now measures the opposite of its own name: the pin's six survive
as an in-order PREFIX at arity ten, with ``-015``'s four declared fields as the
tail. The name is kept byte-unchanged because it is the item id this file's
landed acceptance collects.

PARAMETER NAMES ARE DERIVED, NOT CHOSEN
---------------------------------------
Per the lead ruling recorded at
``artifacts/campaigns/glm-5.3-flash-port/increments/evidence-013.md`` L212 --
the original ``approvals/lead-ruling-013-param-name-authority.md`` was deleted
in the 2026-08-31 residue purge -- the landed weight
map's param-name side is the authority for this skeleton's parameter attribute
paths. :func:`test_kv_spec_declared_parameter_names_match_the_landed_weight_map`
asserts that derivation as an exact set equality against
``build_weight_mappings``, with a mutation arm proving the comparison can
fail. That is not an acceptance conjunct -- C01-C04 are unchanged -- it is the
ruling's derivation made mechanical.

``inc-glm53f-082`` is the THIRD writer here (after ``-013``'s creation and
``-015``'s pin-widening guard), declared in the plan's ``§9`` row for this file.
It adds four counted items -- one per declared parameter-name family -- and
repairs one attribute access onto the stable ``.attention`` property. It owns no
KV-geometry census item, and ``-013``'s four counted values (45 / 11 / 34 / 0)
do not move.

FALSIFIABILITY
--------------
Every counted reading carries an arm that would fail if the reading were
vacuous: the 45/11/34 counts are re-derived from three independent bases and
proved to follow the schedule rather than a constant; C04's counted zero is
paired with a live positive control on the counter and with the ``None``
branch of the dtype resolver it depends on; and the zero-allocation claim is
proved live by allocating one parameter inside the same patched window.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextTextConfig,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import build_weight_mappings
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig

# ---------------------------------------------------------------------------
# Declared values, and the pins that keep them honest.
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

# Same digest ``test_config.py:50`` pins, so a silent fixture edit cannot move
# a declared value here either.
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

#: C01.
DECLARED_TOTAL_ENTRIES = 45
#: C02.
DECLARED_MLA_ENTRIES = 11
#: C03.
DECLARED_KDA_ENTRIES = 34
#: C02's value, ``kv_lora_rank`` with a zero rotary slice.
DECLARED_MLA_HEAD_SIZE = 512
#: C04.
DECLARED_NONE_DTYPE_ENTRIES = 0

#: BASE 3 for the schedule, as ``test_config.py:54`` enumerates it from the
#: intake record. Repeated as a literal rather than imported so the two files
#: are independent bases and not one base read twice.
INTAKE_RECORDED_DSA_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

#: The landed split ``test_config.py:57`` pins, as ``(num_kda, num_dsa)``.
EXPECTED_LAYER_SPLIT = (34, 11)

#: KDA geometry from ``linear_attn_config`` (``glm5_next/config.py:165-171``).
DECLARED_KDA_HEAD_SIZE = 128
DECLARED_KDA_NUM_HEADS = 64

#: The pin's ``LayerSpec`` field set, in order (``model/kv_cache.py:16-21``).
PIN_LAYER_SPEC_FIELDS = (
    "name",
    "num_kv_heads",
    "head_size",
    "dtype",
    "sliding_window_size",
    "chunk_size",
)

#: The six fused KDA names ``inc-glm53f-078`` retired, listed so ``-082``'s
#: KDA item can count SURVIVALS of them rather than assert a bare absence.
#: These are the ONLY literal names ``-082`` types on an expectation side; every
#: count below is derived from the landed map inside the test body.
RETIRED_KDA_FUSED_NAMES = (
    "in_proj_qkvz_weight",
    "in_proj_ba_weight",
    "out_proj_weight",
    "conv1d_weight",
    "norm_weight",
    "conv1d_bias",
)

#: The provisional DSA indexer name ``-078`` retired for ``wq_b_weight``.
RETIRED_DSA_INDEXER_NAME = "wq_weight"

#: ``-013``'s two layer-level names, read here only to ISOLATE the mHC leaves
#: from the map's layer-level set. Their count is not a ``-082`` reading.
LANDED_LAYER_LEVEL_LAYERNORMS = (
    "input_layernorm_weight",
    "post_attention_layernorm_weight",
)

IMPL_MODULE = "vllm_neuron.model.glm5_next.model_fp8"

_RESULTS_PATH = Path(
    os.environ.get("VLLM_NEURON_INC013_RESULTS_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc013_predicates.json"
)
_RESULTS: dict[str, Any] = {}
_RESULTS_PATH.write_text("{}\n")  # truncate stale values from an earlier run


def _record(**values: Any) -> None:
    _RESULTS.update(values)
    _RESULTS_PATH.write_text(
        json.dumps(_RESULTS, indent=2, sort_keys=True, default=str) + "\n"
    )


# ---------------------------------------------------------------------------
# The deferred import (see the module docstring) and the fixtures.
# ---------------------------------------------------------------------------


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _raw() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def raw() -> dict:
    return _raw()


@pytest.fixture(scope="module")
def model(raw: dict):
    """The skeleton built from the landed 45-layer fixture.

    Built through the classmethod ``glm5_next/factory.py:342`` actually calls, so the
    acceptance exercises the pinned construction signature rather than a
    convenience path.
    """
    return _impl().Glm5NextForConditionalGeneration.from_configs(
        copy.deepcopy(raw),
        text_neuron_config=None,
        vision_neuron_config=None,
    )


@pytest.fixture(scope="module")
def spec(model) -> KVSpec:
    return model.get_kv_spec()


def _mla_entries(spec: KVSpec) -> list[LayerSpec]:
    return [
        layer for layer in spec.layers if layer.head_size == DECLARED_MLA_HEAD_SIZE
    ]


def _kda_entries(spec: KVSpec) -> list[LayerSpec]:
    """The COMPLEMENT of the MLA/DSA set -- the reading C03 compels."""
    return [
        layer for layer in spec.layers if layer.head_size != DECLARED_MLA_HEAD_SIZE
    ]


def _map_leaves(names: set[str], prefix: str) -> set[str]:
    """Leaf names in ``names`` sitting DIRECTLY under ``prefix``, no deeper.

    ``inc-glm53f-082``'s helper. The "no deeper" rule is what separates an
    attention module's own leaves from its ``indexer`` submodule's, so the two
    families can be counted independently out of one map.
    """
    leaves = set()
    for name in names:
        if not name.startswith(prefix):
            continue
        leaf = name[len(prefix) :]
        if "." in leaf:
            continue
        leaves.add(leaf)
    return leaves


def _layer_indices(model, layer_type: str) -> list[int]:
    """Indices of one family, read off the built layers' own ``layer_type``."""
    return [
        index
        for index, layer in enumerate(model.model.layers)
        if layer.layer_type == layer_type
    ]


# ---------------------------------------------------------------------------
# The instrument is the instrument it claims to be.
# ---------------------------------------------------------------------------


def test_kv_spec_fixture_is_the_pinned_forty_five_layer_config(raw: dict) -> None:
    """The 45-layer instrument, pinned by digest before anything counts it."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256

    text = raw["text_config"]
    assert text["num_hidden_layers"] == DECLARED_TOTAL_ENTRIES
    assert len(text["layer_types"]) == DECLARED_TOTAL_ENTRIES
    assert text["kv_lora_rank"] == DECLARED_MLA_HEAD_SIZE
    assert text["qk_rope_head_dim"] == 0
    assert text["linear_attn_config"]["head_dim"] == DECLARED_KDA_HEAD_SIZE
    _record(report_fixture_sha256=digest)


def test_kv_spec_the_pin_dataclass_is_not_widened() -> None:
    """SUPERSEDED by ``inc-glm53f-015``: the pin's SIX survive as a PREFIX.

    This guard's landed docstring named its own retirer -- *"it proves this
    increment did not widen ``LayerSpec``, which is ``inc-glm53f-015``'s
    declared surface at M1"* -- and that increment has now landed, so it
    retires the guard inside its own changeset rather than leave a landed test
    red. What the guard exists to catch is KEPT and gains a count: a rename, a
    reorder or a removal of the pin's six still fails the prefix assertion.
    The other half of the old reading -- *"there is no recurrent-state field
    for a KDA entry to fill"* -- is what ``-015`` makes false BY DESIGN, so it
    is INVERTED into two positive counted statements (arity ten, and the four
    declared KDA state fields as the tail in order) rather than deleted. The
    6-argument construction below is ``-015``'s own "0 signature breaks"
    conjunct, preserved here as landed bytes.

    The function NAME is deliberately byte-unchanged: it is the item id this
    file's landed ``-013`` acceptance command collects, and renaming it would
    silently change that collected set to buy a more accurate label.
    """
    from dataclasses import fields

    names = tuple(f.name for f in fields(LayerSpec))
    assert names[:6] == PIN_LAYER_SPEC_FIELDS
    assert len(names) == 10
    assert names[6:] == (
        "kda_conv_state_shape",
        "kda_recurrent_state_shape",
        "kda_conv_state_dtype",
        "kda_recurrent_state_dtype",
    )
    assert tuple(f.name for f in fields(KVSpec)) == ("layers",)

    # The pin's exact 6-argument positional form still constructs.
    probe = LayerSpec("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)
    assert probe.sliding_window_size is None
    assert probe.chunk_size is None


# ---------------------------------------------------------------------------
# C01 -- exactly 45 LayerSpec entries.
# ---------------------------------------------------------------------------


def test_kv_spec_c01_the_spec_has_exactly_forty_five_layer_entries(
    spec: KVSpec, model
) -> None:
    """C01, plus a second base: the entry count equals the stack depth."""
    assert isinstance(spec, KVSpec)
    assert len(spec.layers) == DECLARED_TOTAL_ENTRIES
    assert all(isinstance(layer, LayerSpec) for layer in spec.layers)

    # BASE 2 -- the module tree's own layer count, not the declared literal.
    assert len(model.model.layers) == DECLARED_TOTAL_ENTRIES
    # BASE 3 -- the config's declared depth.
    assert model.text_config.num_hidden_layers == DECLARED_TOTAL_ENTRIES
    _record(report_c01_entry_count=len(spec.layers))


# ---------------------------------------------------------------------------
# C02 -- 11 entries carry MLA/DSA geometry, head_size == 512.
# ---------------------------------------------------------------------------


def test_kv_spec_c02_eleven_entries_carry_the_mla_latent_geometry(
    spec: KVSpec, model
) -> None:
    """C02, with the positions checked against three independent bases."""
    mla = _mla_entries(spec)
    assert len(mla) == DECLARED_MLA_ENTRIES

    positions = [
        index
        for index, layer in enumerate(spec.layers)
        if layer.head_size == DECLARED_MLA_HEAD_SIZE
    ]
    # BASE 1 -- the fixture's own schedule, read through the config accessor.
    assert positions == model.text_config.dsa_layer_indices
    # BASE 2 -- the indices enumerated in the intake record.
    assert positions == INTAKE_RECORDED_DSA_INDICES
    # BASE 3 -- the 3:1 interleave rule, arithmetically.
    assert positions == [
        index for index in range(DECLARED_TOTAL_ENTRIES) if index % 4 == 3
    ]

    for layer in mla:
        assert layer.name.endswith(".self_attn")
        assert layer.head_size == DECLARED_MLA_HEAD_SIZE
        assert layer.num_kv_heads == 1, "the MLA latent is one replicated KV head"
        assert layer.chunk_size is None
    _record(report_c02_mla_entry_count=len(mla), report_c02_positions=positions)


def test_kv_spec_c02_the_head_size_is_the_latent_width_not_a_literal(model) -> None:
    """512 is ``kv_lora_rank + qk_rope_head_dim``, and the rope slice is 0.

    Non-vacuity for C02's value: the number is derived from two config fields
    whose sum is asserted, so a hard-coded 512 in the resolver would not
    survive the mutation arm below.
    """
    impl = _impl()
    text_config = model.text_config
    assert text_config.kv_lora_rank == DECLARED_MLA_HEAD_SIZE
    assert text_config.qk_rope_head_dim == 0
    assert text_config.mla_use_nope is True
    assert impl._resolve_mla_head_size(text_config) == DECLARED_MLA_HEAD_SIZE

    # MUTATION ARM: a config with a rotary slice must widen the latent.
    with_rope = Glm5NextTextConfig(kv_lora_rank=512, qk_rope_head_dim=64)
    assert impl._resolve_mla_head_size(with_rope) == 576


# ---------------------------------------------------------------------------
# C03 -- 34 entries carry KDA geometry, by complement.
# ---------------------------------------------------------------------------


def test_kv_spec_c03_thirty_four_entries_carry_the_kda_geometry(
    spec: KVSpec, model
) -> None:
    """C03 by complement, and the complement is exhaustive.

    The partition is unambiguous because ``128 != 512``: no KDA entry can be
    mistaken for an MLA/DSA one, and the two counts sum to C01's total.
    """
    kda = _kda_entries(spec)
    assert len(kda) == DECLARED_KDA_ENTRIES
    assert len(kda) + len(_mla_entries(spec)) == DECLARED_TOTAL_ENTRIES

    positions = [
        index
        for index, layer in enumerate(spec.layers)
        if layer.head_size != DECLARED_MLA_HEAD_SIZE
    ]
    assert positions == model.text_config.kda_layer_indices

    for layer in kda:
        assert layer.name.endswith(".linear_attn")
        assert layer.head_size == DECLARED_KDA_HEAD_SIZE
        assert layer.head_size != DECLARED_MLA_HEAD_SIZE
        assert layer.num_kv_heads == DECLARED_KDA_NUM_HEADS
    _record(report_c03_kda_entry_count=len(kda))


def test_kv_spec_the_split_reproduces_the_landed_config_pins(spec: KVSpec) -> None:
    """The pair ``(34, 11)`` that ``test_config.py:57`` already pins.

    Asserted as ONE pair, never as two independent equalities, for the same
    reason ``test_config.py:9-14`` gives.
    """
    measured = (len(_kda_entries(spec)), len(_mla_entries(spec)))
    assert measured == EXPECTED_LAYER_SPLIT
    assert sum(measured) == DECLARED_TOTAL_ENTRIES
    _record(report_layer_split=list(measured))


def test_kv_spec_the_counts_follow_the_schedule_and_not_a_constant(raw: dict) -> None:
    """MUTATION ARM for C01-C03: move one layer, and the counts must move.

    Flipping layer 0 from KDA to DSA keeps the partition a valid exhaustive
    one over 45 layers but makes the pair ``(33, 12)``. A ``get_kv_spec`` that
    returned the declared numbers as constants would fail here.
    """
    mutated = copy.deepcopy(raw)
    mutated["text_config"]["layer_types"][0] = DSA_LAYER_TYPE

    model = _impl().Glm5NextForConditionalGeneration.from_configs(mutated)
    spec = model.get_kv_spec()

    assert len(spec.layers) == DECLARED_TOTAL_ENTRIES
    assert len(_mla_entries(spec)) == DECLARED_MLA_ENTRIES + 1
    assert len(_kda_entries(spec)) == DECLARED_KDA_ENTRIES - 1
    assert spec.layers[0].name.endswith(".self_attn")
    assert spec.layers[0].head_size == DECLARED_MLA_HEAD_SIZE


def test_kv_spec_every_entry_name_is_unique_and_family_tagged(spec: KVSpec) -> None:
    """45 distinct names, each carrying its family's module path.

    ``inc-glm53f-016`` (M1) has to split the runner's spec dict **34 KDA / 11
    DSA** by *"the model's layer names"*, so the names must distinguish the
    families rather than merely being unique.
    """
    names = [layer.name for layer in spec.layers]
    assert len(set(names)) == DECLARED_TOTAL_ENTRIES
    assert names == [
        f"layers.{index}."
        + ("self_attn" if index in INTAKE_RECORDED_DSA_INDICES else "linear_attn")
        for index in range(DECLARED_TOTAL_ENTRIES)
    ]
    assert sum(1 for name in names if name.endswith(".linear_attn")) == (
        DECLARED_KDA_ENTRIES
    )
    assert sum(1 for name in names if name.endswith(".self_attn")) == (
        DECLARED_MLA_ENTRIES
    )
    _record(report_entry_names_sample=names[:5])


# ---------------------------------------------------------------------------
# C04 -- 0 entries have ``dtype is None``.
# ---------------------------------------------------------------------------


def test_kv_spec_c04_no_entry_has_a_none_dtype(spec: KVSpec, model) -> None:
    """C04, the counted zero, plus what every dtype actually is."""
    none_dtypes = [layer for layer in spec.layers if layer.dtype is None]
    assert len(none_dtypes) == DECLARED_NONE_DTYPE_ENTRIES

    for layer in spec.layers:
        assert isinstance(layer.dtype, torch.dtype)
        assert layer.dtype is model.text_config.torch_dtype
    assert model.text_config.torch_dtype is torch.bfloat16
    _record(
        report_c04_none_dtype_count=len(none_dtypes),
        report_c04_dtypes=sorted({str(layer.dtype) for layer in spec.layers}),
    )


def test_kv_spec_c04_the_none_dtype_counter_is_proved_live(spec: KVSpec) -> None:
    """POSITIVE CONTROL for C04: without this the zero above proves nothing.

    The same predicate, over a population deliberately driven non-zero. This
    is an instrument-liveness arm on the counter, not a change to C04 -- the
    criterion and its expected value are untouched.
    """
    counted = [layer for layer in spec.layers if layer.dtype is None]
    assert counted == []

    salted = [*spec.layers, LayerSpec("probe", 1, 512, None)]
    assert len([layer for layer in salted if layer.dtype is None]) == 1


def test_kv_spec_c04_the_kda_state_dtype_resolver_has_a_live_none_branch() -> None:
    """C04's zero rests on a resolver with a real ``None`` branch.

    ``NeuronConfig.kda_state_dtype`` is declared as a dtype **name** with
    *"None = follow the model's own dtype"* (``neuron_config.py:181-184``), so
    the KDA entries' dtype comes from a resolution with three outcomes. All
    three are exercised, and the ``None`` input is the one that would put a
    ``None`` dtype into a ``LayerSpec`` if the fallback were missing.
    """
    impl = _impl()

    # None override -> the model's own dtype, never None.
    unset = Glm5NextTextConfig(neuron_config=NeuronConfig())
    assert unset.neuron_config.kda_state_dtype is None
    assert impl._resolve_kda_state_dtype(unset) is torch.bfloat16

    # No neuron_config at all -> same fallback.
    bare = Glm5NextTextConfig()
    assert bare.neuron_config is None
    assert impl._resolve_kda_state_dtype(bare) is torch.bfloat16

    # A named override is honoured and coerced off the string.
    named = Glm5NextTextConfig(neuron_config=NeuronConfig(kda_state_dtype="float32"))
    assert impl._resolve_kda_state_dtype(named) is torch.float32

    # A name that is not a dtype is raised, not passed through as a str.
    with pytest.raises(ValueError, match="does not name a torch dtype"):
        impl._resolve_kda_state_dtype(
            Glm5NextTextConfig(neuron_config=NeuronConfig(kda_state_dtype="not_a_dtype"))
        )


def test_kv_spec_the_kda_state_dtype_override_reaches_the_spec(raw: dict) -> None:
    """The resolver is wired: an override moves the 34 KDA entries only."""
    model = _impl().Glm5NextForConditionalGeneration.from_configs(
        copy.deepcopy(raw),
        text_neuron_config=NeuronConfig(kda_state_dtype="float32"),
    )
    spec = model.get_kv_spec()

    kda = _kda_entries(spec)
    mla = _mla_entries(spec)
    assert len(kda) == DECLARED_KDA_ENTRIES
    assert all(layer.dtype is torch.float32 for layer in kda)
    assert all(layer.dtype is torch.bfloat16 for layer in mla)
    assert [layer for layer in spec.layers if layer.dtype is None] == []


# ---------------------------------------------------------------------------
# The skeleton is a skeleton: no allocation, and every compute site a stub.
# ---------------------------------------------------------------------------


def test_kv_spec_building_the_tree_allocates_no_parameters(
    raw: dict, monkeypatch
) -> None:
    """45 layers of 4096 hidden with 288 routed experts, allocating nothing.

    This is what makes the acceptance runnable in CPU mode at the real depth
    instead of narrowed to a mini config, and it is measured rather than
    asserted: the same counter is proved live inside the same patched window
    by allocating one parameter on purpose. Shape borrowed from
    ``test_factory.py:255-278``.
    """
    impl = _impl()
    created: list[type] = []
    original_new = torch.nn.Parameter.__new__

    def counting_new(cls, *args, **kwargs):
        created.append(cls)
        return original_new(cls, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Parameter, "__new__", staticmethod(counting_new))

    model = impl.Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))
    spec = model.get_kv_spec()

    assert len(spec.layers) == DECLARED_TOTAL_ENTRIES
    assert created == [], f"building the skeleton allocated {len(created)} Parameter(s)"

    # POSITIVE CONTROL, same window.
    torch.nn.Parameter(torch.zeros(1))
    assert len(created) == 1
    _record(report_parameters_allocated=0)


def test_kv_spec_declared_parameters_are_reserved_and_unmaterialised(model) -> None:
    """A declared name resolves to ``None`` and is skipped by torch's walks.

    Both halves matter: the attribute path exists (so a later increment fills
    a name that is already there), and ``named_parameters()`` / ``state_dict``
    stay empty (so nothing here pretends to be a loadable weight).
    """
    layer_zero = model.model.layers[0]
    assert layer_zero.input_layernorm_weight is None
    assert "input_layernorm_weight" in layer_zero._parameters

    assert list(model.named_parameters()) == []
    assert model.state_dict() == {}


def test_kv_spec_every_compute_site_is_a_stub(model) -> None:
    """Forward raises everywhere -- the substrate declaration's own ground.

    This increment is declared NON-KERNEL-CLASS because every compute site is
    a stub, so "a stub computes nothing" is a property worth measuring rather
    than promising: a later increment that quietly implemented a torch
    forward here would change that declaration and this test would fail.
    """
    impl = _impl()
    # Counted, not asserted against a literal: two retirements landed here for
    # ``inc-glm53f-038a`` and the number this census still guards belongs in the
    # transcript, where a later retirement that quietly emptied the walk would
    # show up as a smaller reading rather than as a still-green test.
    asserted = 0
    for model_level in (model, model.model):
        with pytest.raises(NotImplementedError):
            model_level.forward()
        asserted += 1

    # NOT ``(0, 3)`` any more: ``layers[0]`` is retired below, so only the DSA
    # layer is walked at the layer level.
    modules = [model.model.layers[3]]
    # ``.attention`` rather than a family attribute name: ``inc-glm53f-082``
    # moved the KDA module onto the map's ``self_attn`` path, and the property
    # is the access that survives such a move. ``layers[0].attention`` itself is
    # retired below; the ``.mlp`` beside it is not.
    modules += [model.model.layers[0].mlp]
    modules += [
        model.model.layers[3].self_attn,
        model.model.layers[3].self_attn.indexer,
        model.model.layers[3].mlp,
        model.model.layers[3].mlp.experts,
        model.model.layers[3].mlp.shared_experts,
    ]
    for module in modules:
        with pytest.raises(NotImplementedError):
            module.forward()
        asserted += 1

    assert len(modules) == 7, f"module arms drifted to {len(modules)}"
    _record(stub_forwards_asserted=asserted)
    assert asserted == 9

    # THE RETIREMENT IS MEASURED, NOT ANNOUNCED. The two retired forwards must
    # really be implemented, or this retirement would be hiding a stub instead of
    # handing one over. The property checked is the one being handed over and
    # nothing more: calling it no longer says NotImplementedError. Deliberately
    # NOT a match on the new signature -- a later increment that gives
    # ``hidden_states`` a default would then redden this census for a reason that
    # has nothing to do with a quiet stub, and a working forward must be allowed
    # to simply succeed.
    #
    # THE SECOND CLAUSE IS NARROW, and `B40 N7` is why. It used to read
    # ``except Exception``, which also swallowed a forward that failed for an
    # unrelated reason -- a shape bug or a typo then read as a retirement. The
    # class is not guessed: both retired forwards were measured through this
    # file's own fixture to raise ``TypeError`` for a missing ``hidden_states``
    # (``probe-R13-retired-forward-classes.out``), which is the outcome the
    # paragraph above already describes. Success is still allowed, because no
    # clause catches it, and anything else now propagates instead of passing.
    retired = [model.model.layers[0], model.model.layers[0].attention]
    still_stubbed = []
    for module in retired:
        try:
            module.forward()
        except NotImplementedError:
            still_stubbed.append(type(module).__name__)
        except TypeError:
            pass  # implemented; it merely wants its arguments (`B40 N7`)
    _record(retired_forwards_still_stubbed=still_stubbed)
    assert still_stubbed == [], (
        f"retired as handed over to inc-glm53f-038a, but still a stub: "
        f"{still_stubbed}"
    )
    assert len(retired) == 2

    # NO reserved-name arm is left in this census, and every retirement was a
    # declared handover rather than a loss of coverage. `Glm5NextQuantConfig()`
    # was retired by `inc-glm53f-023` and `Glm5NextHyperConnection().forward()`
    # by `inc-glm53f-030`, each the DECLARED lander of that D14 section -- this
    # census is a tripwire against a QUIET implementation, and a declared lander
    # is the opposite of quiet. `-030` implements the mHC wiring its section
    # reserved, so the constructor now takes the config the layer is sized from
    # and `forward` computes instead of raising.
    #
    # THIRD AND FOURTH RETIREMENT, `inc-glm53f-038a`, same form and same reason.
    # `layers[0].forward()` (`Glm5NextKDALayer`) and `layers[0].attention.forward()`
    # (`Glm5NextKDAAttention`) are gone from the walk above. `-038a` is the
    # DECLARED lander of both, and both now take `hidden_states` positionally plus
    # three keyword-only carriers -- so a zero-argument call raises `TypeError`,
    # which `pytest.raises(NotImplementedError)` does not catch. Keeping the arms
    # would have made this census fail on the very implementation it exists to
    # announce. Retiring them costs no coverage that matters here: the census
    # guards against a QUIET forward, and `-038a` declared both.
    #
    # WHAT STILL HOLDS EACH RETIRED MODULE IN THIS FILE, so neither disappears
    # from the file's reach along with its arm:
    #   * `layers[0]` -- `test_kv_spec_the_tree_carries_every_d14_section_name`
    #     asserts it is a `Glm5NextKDALayer`, and five other arms read it.
    #   * `layers[0].attention` -- the `.attention` property access is exercised
    #     on a KDA layer at line 870 of this file, which is the access
    #     `inc-glm53f-082`'s move made load-bearing.
    #
    # `-013`'s four declared counts (45 / 11 / 34 / 0) are untouched, as is every
    # other arm. The two model-level arms and the seven remaining module arms
    # stand, so this census still asserts NINE stub forwards.


def test_kv_spec_the_tree_carries_every_d14_section_name(model) -> None:
    """D14's table names the sections later increments scope against.

    ``-013`` is the creator of a coordinated merge point, so every name in
    that table must exist when this increment lands, or eleven later
    increments have no declared scope to write into.
    """
    impl = _impl()
    for name in (
        "Glm5NextQuantConfig",
        "Glm5NextMoEBlock",
        "Glm5NextHyperConnection",
        "Glm5NextKDALayer",
        "Glm5NextMLAAttention",
        "Glm5NextDSALayer",
    ):
        assert hasattr(impl, name), f"D14 section name {name} is missing"

    assert isinstance(model.model.layers[0], impl.Glm5NextKDALayer)
    assert isinstance(model.model.layers[3], impl.Glm5NextDSALayer)
    assert isinstance(model.model.layers[3].self_attn, impl.Glm5NextMLAAttention)
    assert isinstance(model.model.layers[3].mlp, impl.Glm5NextMoEBlock)
    assert isinstance(model.model.layers[0].mlp, impl.Glm5NextDenseMLP)


# ---------------------------------------------------------------------------
# The lead ruling's derivation: the landed weight map is the authority for
# this skeleton's parameter attribute paths.
# ---------------------------------------------------------------------------


def test_kv_spec_declared_parameter_names_match_the_landed_weight_map(model) -> None:
    """Exact set equality against ``build_weight_mappings``'s param-name side.

    Per the lead ruling recorded at
    ``artifacts/campaigns/glm-5.3-flash-port/increments/evidence-013.md`` L212
    (the original ``approvals/lead-ruling-013-param-name-authority.md`` was
    deleted in the 2026-08-31 residue purge), the landed,
    passing map is the authority and this skeleton derives from it. Neither
    side is renamed here: this asserts the derivation, so a divergence is a
    red test at CPU-mode rung 1 instead of a defect that surfaces only when a
    real checkpoint is loaded on hardware.

    NOT an acceptance conjunct -- C01-C04 are untouched.
    """
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())

    assert declared == mapped, (
        f"skeleton-only names: {sorted(declared - mapped)}; "
        f"map-only names: {sorted(mapped - declared)}"
    )
    # No name is declared twice under the same path.
    assert len(model.declared_parameter_names()) == len(declared)
    _record(
        report_declared_parameter_count=len(declared),
        report_mapped_parameter_count=len(mapped),
    )


def test_kv_spec_the_map_comparison_is_proved_live(model) -> None:
    """MUTATION ARM: the set equality above must be able to fail.

    Without this, a comparison of two empty sets would pass and certify
    nothing.
    """
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())
    assert declared, "the skeleton declared no parameter names at all"
    assert mapped, "the landed map produced no parameter names at all"

    dropped = sorted(declared)[0]
    assert (declared - {dropped}) != mapped


def test_kv_spec_the_quantised_flag_does_not_move_the_param_name_side(model) -> None:
    """AMENDED by ``inc-glm53f-085``: the flag ADDS exactly the 44 scale names.

    THE NAME IS KEPT BYTE-UNCHANGED because it is the item id this file's landed
    acceptance collects -- the same reason
    :func:`test_kv_spec_the_pin_dataclass_is_not_widened` keeps a name that now
    measures the opposite of itself.

    WHY IT CHANGED, measured rather than to reach green. The landed assertion
    ``quantised == plain`` held only because both checkpoint keys of a scaled
    projection were bound to ONE weight parameter, so 44 DSA weight parameters
    carried a two-key list and the parameter side could not move with the flag.
    The loader's default refuses more than one slice, so as landed the scale
    reached no arithmetic -- that binding is the defect ``inc-glm53f-085``
    repairs, and once each scale key has its own parameter the flag necessarily
    ADDS exactly the 44 scale-parameter names.

    CERTIFYING COMPONENT (D1.4): ``_add_dsa_attention``'s
    ``DSA_SCALED_PROJECTIONS`` loop and ``_quantised``.
    """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
        DSA_SCALED_PROJECTIONS,
        FP8_SCALE_SUFFIX,
    )

    quantised = set(build_weight_mappings(model.text_config, quantised=True))
    plain = set(build_weight_mappings(model.text_config, quantised=False))
    dsa_indices = _layer_indices(model, DSA_LAYER_TYPE)

    # (i) The flag only ever ADDS parameter names.
    assert plain <= quantised, f"names only in plain: {sorted(plain - quantised)}"

    # (ii) And what it adds is exactly the scale names, derived here.
    expected = {
        f"model.layers.{index}.self_attn.{leaf}_{FP8_SCALE_SUFFIX}"
        for index in dsa_indices
        for leaf in DSA_SCALED_PROJECTIONS
    }
    assert quantised - plain == expected, (
        f"symmetric difference: {sorted((quantised - plain) ^ expected)}"
    )
    assert len(expected) == 4 * DECLARED_MLA_ENTRIES == 44

    # (iii) A counted zero, with its control on the mutation-arm form above:
    #       dropping one shared name from the superset makes it read 1.
    assert len(plain - quantised) == 0
    mutated = quantised - {sorted(plain)[0]}
    assert len(plain - mutated) == 1, "the zero above is vacuous"

    # (iv) The landed second assertion, kept.
    assert set(model.declared_parameter_names()) == quantised

    # (v) Each scaled leaf's WEIGHT parameter maps to a SCALAR target, so the
    #     scale key no longer shares it. On the base tree all 44 were lists.
    mappings = build_weight_mappings(model.text_config, quantised=True)
    weight_names = [
        f"model.layers.{index}.self_attn.{leaf}_weight"
        for index in dsa_indices
        for leaf in DSA_SCALED_PROJECTIONS
    ]
    assert len(weight_names) == 44
    list_valued = [n for n in weight_names if isinstance(mappings[n], list)]
    assert not list_valued, (
        f"{len(list_valued)} scaled-leaf weight parameters still carry a "
        f"multi-key list: {sorted(list_valued)[:4]}"
    )

    _record(
        c085_quantised_only_param_names=len(quantised - plain),
        c085_plain_only_param_names=len(plain - quantised),
        c085_dsa_layers=len(dsa_indices),
        c085_dsa_weight_list_valued_count=len(list_valued),
    )


def test_kv_spec_the_tied_head_condition_mirrors_the_map(raw: dict) -> None:
    """``lm_head_weight`` is declared iff the map declares it.

    The map omits the key when ``tie_word_embeddings`` is set
    (``weight_loaders_fp8.py:382-383``); the skeleton must omit the parameter
    on the same condition or the two sides disagree on that one name.
    """
    tied = copy.deepcopy(raw)
    tied["tie_word_embeddings"] = True

    model = _impl().Glm5NextForConditionalGeneration.from_configs(tied)
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())

    assert model.text_config.tie_word_embeddings is True
    assert "lm_head_weight" not in declared
    assert declared == mapped


# ---------------------------------------------------------------------------
# ``inc-glm53f-082``: one counted item per declared parameter-name family.
#
# The set equality above is the BINDING reading and it is family-blind -- it
# passes or fails on the whole map at once. These four say WHICH family moved,
# so a later regression names itself instead of surfacing as one opaque diff.
# Every denominator is derived from the landed map inside the body: no count is
# typed, and there is no ``parametrize`` (plan ``§6`` rule 6 -- one item per
# counted conjunct).
# ---------------------------------------------------------------------------


def test_kv_spec_kda_attention_declares_the_maps_fifteen_leaf_names(model) -> None:
    """The KDA half's declared names ARE the map's, per module and per path.

    ``inc-glm53f-078`` measured fifteen ``self_attn.*`` leaves off the published
    checkpoint index where the skeleton had eight fused ones. This reads the
    leaf set out of the map's own keys, so the expectation cannot drift from the
    authority it is quoting.
    """
    mapped = set(build_weight_mappings(model.text_config))
    declared = set(model.declared_parameter_names())
    kda_indices = _layer_indices(model, KDA_LAYER_TYPE)
    assert kda_indices, "the fixture produced no linear-attention layer"

    leaves = _map_leaves(mapped, f"model.layers.{kda_indices[0]}.self_attn.")
    for index in kda_indices:
        assert (
            _map_leaves(mapped, f"model.layers.{index}.self_attn.") == leaves
        ), f"the map is not uniform across the KDA family at layer {index}"

    module = model.model.layers[kda_indices[0]].attention
    assert set(module.declared_param_names) == leaves, (
        f"module-only: {sorted(set(module.declared_param_names) - leaves)}; "
        f"map-only: {sorted(leaves - set(module.declared_param_names))}"
    )
    # The two bare state tensors have no ``.weight`` leaf and the map keeps
    # them, so their presence is part of the family's shape, not an accident.
    assert {"A_log", "dt_bias"} <= leaves

    present = {
        name
        for index in kda_indices
        for name in declared
        if name.startswith(f"model.layers.{index}.self_attn.")
    }
    assert len(present) == len(leaves) * len(kda_indices)

    # SURVIVALS, scoped to the family's own module paths: ``norm_weight`` is
    # also the legitimate ``model.norm_weight``, so an unscoped scan would
    # false-fire on a name this block must leave alone.
    survivals = sorted(
        name
        for name in declared
        if name.rsplit(".", 1)[0].endswith(".self_attn")
        and name.rsplit(".", 1)[-1] in RETIRED_KDA_FUSED_NAMES
    )
    assert survivals == []

    _record(
        report_kda_attention_leaf_count=len(leaves),
        report_kda_attention_path_count=len(present),
        report_kda_layer_count=len(kda_indices),
        report_kda_retired_name_survivals=len(survivals),
    )


def test_kv_spec_dsa_indexer_declares_the_maps_seven_leaf_names(model) -> None:
    """The sparse indexer's declared names ARE the map's seven.

    Three of the four names the skeleton carried were right; ``wq_weight`` named
    a tensor the checkpoint does not carry, and three more were missing.
    """
    mapped = set(build_weight_mappings(model.text_config))
    declared = set(model.declared_parameter_names())
    dsa_indices = _layer_indices(model, DSA_LAYER_TYPE)
    assert dsa_indices, "the fixture produced no sparse-attention layer"

    prefix = f"model.layers.{dsa_indices[0]}.self_attn.indexer."
    leaves = _map_leaves(mapped, prefix)
    for index in dsa_indices:
        assert (
            _map_leaves(mapped, f"model.layers.{index}.self_attn.indexer.") == leaves
        ), f"the map is not uniform across the indexer family at layer {index}"

    module = model.model.layers[dsa_indices[0]].attention.indexer
    assert set(module.declared_param_names) == leaves, (
        f"module-only: {sorted(set(module.declared_param_names) - leaves)}; "
        f"map-only: {sorted(leaves - set(module.declared_param_names))}"
    )

    present = {
        name
        for index in dsa_indices
        for name in declared
        if name.startswith(f"model.layers.{index}.self_attn.indexer.")
    }
    assert len(present) == len(leaves) * len(dsa_indices)

    survivals = sorted(
        name
        for name in declared
        if name.endswith(f".indexer.{RETIRED_DSA_INDEXER_NAME}")
    )
    assert survivals == []

    _record(
        report_dsa_indexer_leaf_count=len(leaves),
        report_dsa_indexer_path_count=len(present),
        report_dsa_layer_count=len(dsa_indices),
        report_dsa_indexer_retired_name_survivals=len(survivals),
    )


def test_kv_spec_both_layer_classes_declare_the_maps_mhc_names_flat(model) -> None:
    """The mHC weights sit FLAT ON THE LAYER, on both layer classes.

    The map emits them from an unconditional ``_add_mhc``, so every layer of the
    stack carries them and neither layer class declared any of them before. The
    "flat" half is measured, not assumed: a name of this set found under any
    submodule attribute fails here.
    """
    mapped = set(build_weight_mappings(model.text_config))
    declared = set(model.declared_parameter_names())
    layers = list(model.model.layers)
    assert layers

    layer_paths = {f"model.layers.{index}" for index in range(len(layers))}
    first = _map_leaves(mapped, "model.layers.0.")
    for index, layer in enumerate(layers):
        at_layer = _map_leaves(mapped, f"model.layers.{index}.")
        assert at_layer == first, (
            f"layer {index} layer-level leaves differ: {sorted(at_layer)}"
        )
        assert set(layer.declared_param_names) == at_layer, (
            f"layer {index} module-only: "
            f"{sorted(set(layer.declared_param_names) - at_layer)}; map-only: "
            f"{sorted(at_layer - set(layer.declared_param_names))}"
        )

    mhc_leaves = first - set(LANDED_LAYER_LEVEL_LAYERNORMS)
    assert mhc_leaves, "the map emits no layer-level name beyond the layernorms"

    misplaced = sorted(
        name
        for name in declared
        if name.rsplit(".", 1)[-1] in mhc_leaves
        and name.rsplit(".", 1)[0] not in layer_paths
    )
    assert misplaced == []

    present = {
        name for name in declared if name.rsplit(".", 1)[-1] in mhc_leaves
    }
    assert len(present) == len(mhc_leaves) * len(layers)

    # BOTH classes, so a single-family regression cannot hide behind the other.
    class_names = {type(layer).__name__ for layer in layers}
    assert len(class_names) == 2, sorted(class_names)
    for layer in layers:
        assert mhc_leaves <= set(layer.declared_param_names)

    _record(
        report_mhc_leaf_count=len(mhc_leaves),
        report_mhc_path_count=len(present),
        report_mhc_layer_classes=sorted(class_names),
        report_mhc_misplaced_paths=len(misplaced),
    )


def test_kv_spec_mla_attention_names_are_unchanged_and_match_the_map(model) -> None:
    """A MEASURED NEGATIVE: the MLA names already agreed and still do.

    ``inc-glm53f-082`` writes no line in this class. The item exists so "the
    fourth family needed nothing" is a reading off the instrument rather than a
    claim in a record.
    """
    mapped = set(build_weight_mappings(model.text_config))
    declared = set(model.declared_parameter_names())
    dsa_indices = _layer_indices(model, DSA_LAYER_TYPE)
    assert dsa_indices

    leaves = _map_leaves(mapped, f"model.layers.{dsa_indices[0]}.self_attn.")
    module = model.model.layers[dsa_indices[0]].attention
    assert set(module.declared_param_names) == leaves, (
        f"module-only: {sorted(set(module.declared_param_names) - leaves)}; "
        f"map-only: {sorted(leaves - set(module.declared_param_names))}"
    )
    # No name declared twice on the module.
    assert len(module.declared_param_names) == len(leaves)

    present = {
        name
        for index in dsa_indices
        for name in _map_leaves(declared, f"model.layers.{index}.self_attn.")
    }
    assert present == leaves
    total = sum(
        len(_map_leaves(declared, f"model.layers.{index}.self_attn."))
        for index in dsa_indices
    )
    assert total == len(leaves) * len(dsa_indices)

    _record(
        report_mla_declared_leaf_count=len(leaves),
        report_mla_declared_path_count=total,
    )


# ---------------------------------------------------------------------------
# Reporting -- the readings the evidence record quotes.
#
# NO P4 GUARD LIVES HERE, DELIBERATELY. An in-test screen for the run-wide
# NxDI-import prohibition would have to carry the forbidden module prefix as a
# string literal, which is itself a textual hit against the mechanical scan
# that ``record_changeset`` runs over added lines -- a self-inflicted false
# positive on a screen this increment does not own the shape of. Writing the
# literal in split form to evade that scan would be worse: never soften a
# run-wide guard to make one's own diff read clean. The prohibition is
# enforced where it belongs, over the branch diff at ``record_changeset`` and
# again at implementation review. This module tree imports torch, the local
# config, ``kv_cache`` and ``neuron_config``, and nothing else.
# ---------------------------------------------------------------------------


def test_kv_spec_reports_the_measured_readings(spec: KVSpec, model) -> None:
    """Write the measured values out; pytest swallows stdout on a pass."""
    _record(
        report_total_entries=len(spec.layers),
        report_mla_entries=len(_mla_entries(spec)),
        report_kda_entries=len(_kda_entries(spec)),
        report_none_dtype_entries=len(
            [layer for layer in spec.layers if layer.dtype is None]
        ),
        report_world_size=model.world_size,
        report_kda_head_size=_kda_entries(spec)[0].head_size,
        report_kda_num_kv_heads=_kda_entries(spec)[0].num_kv_heads,
        report_mla_head_size=_mla_entries(spec)[0].head_size,
        report_mla_num_kv_heads=_mla_entries(spec)[0].num_kv_heads,
        report_chunk_sizes=sorted(
            {str(layer.chunk_size) for layer in spec.layers}
        ),
        report_sliding_windows=sorted(
            {str(layer.sliding_window_size) for layer in spec.layers}
        ),
        report_impl_module=IMPL_MODULE,
        report_results_path=str(_RESULTS_PATH),
    )
    assert _RESULTS_PATH.exists()
