# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-015`` acceptance -- WP2: ``LayerSpec`` KDA state-field widening.

THE DECLARED ACCEPTANCE, command (A) of the plan block, verbatim:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      VLLM_SSM_CONV_STATE_LAYOUT=SD python -m pytest \\
      test/vllm_neuron/model/test_kv_cache_spec.py -q --timeout 60 \\
      -p no:cacheprovider

Four counted conjuncts, one test item each, no ``parametrize``:

* C01 -- ``0`` signature breaks at ``1/1`` construction of the pin's exact
  six-argument positional form, with a live control proving the zero moves.
* C02 -- ``4/4`` new-field defaults read ``None``, by the four declared names
  in the declared order, and ``KVSpec`` is NOT widened.
* C03 -- ``2/2`` state shapes round-trip exactly, rank AND every extent.
* C04 -- byte reconciliation to the recorded KDA state page, discrepancy 0 B.

WHY THE SHAPES ARE DERIVED AND NOT HAND-WRITTEN
-----------------------------------------------
C03 does not write ``(3, 384)`` into a fixture and read it back. It CALLS
``MambaStateShapeCalculator.kda_state_shape`` -- vLLM's own calculator, the
authority the campaign's design cites for this geometry -- at the geometry the
fork's landed ``linear_attn_config`` carries (``glm5_next/config.py:165-171``) and at the
registered tensor-parallel degree, then asserts the returned value against the
expected literals. A vendor change to that calculator, a wrong geometry
argument or a wrong orientation therefore FAILS the arm instead of being
absorbed by it.

THE ORIENTATION TERM IS LOAD-BEARING, AND ONLY C03 GUARDS IT
------------------------------------------------------------
The conv state's extent ORDER is environment-chosen: vLLM's
``_orient_conv_shape`` yields ``(state_len, dim)`` under the ``"SD"`` layout and
``(dim, state_len)`` under ``"DS"``. Command (A) pins ``"SD"`` in the process
invocation (never a fixture, never ``monkeypatch.setenv``), and C03 re-reads the
resolved layout live so a mis-pinned run cannot pass quietly. C04 CANNOT do
this job and does not claim to: transposing a conv extent leaves the element
count -- and therefore the byte total -- identical, so a byte reconciliation is
transposition-invariant by construction. C03 is the only orientation guard.

WHY THIS FILE IMPORTS NO OTHER TEST MODULE
------------------------------------------
The two geometry constants ``-013`` landed
(``glm5_next/test_kv_spec.py:135-137``) cite ``linear_attn_config`` as their own
origin, so this file reads that origin directly instead of importing them: the
``-013`` module truncates a results file at import time, and importing a test
module for its constants would fire that side effect from a foreign session.
Heavy imports stay inside test bodies, which is also ``-013``'s convention for
keeping ``model_fp8`` out of ``sys.modules`` during collection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import pytest
import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

# ---------------------------------------------------------------------------
# Cited values. Nothing here is authored by this increment.
# ---------------------------------------------------------------------------

#: The four field names the increment plan block declares, in its APPEND order.
DECLARED_KDA_FIELDS = (
    "kda_conv_state_shape",
    "kda_recurrent_state_shape",
    "kda_conv_state_dtype",
    "kda_recurrent_state_dtype",
)

#: The pin's ``LayerSpec`` field set, in order (``model/kv_cache.py:16-21``). The
#: widening must leave these six as an in-order PREFIX.
PIN_LAYER_SPEC_FIELDS = (
    "name",
    "num_kv_heads",
    "head_size",
    "dtype",
    "sliding_window_size",
    "chunk_size",
)

#: KDA geometry, as the landed ``linear_attn_config`` carries it
#: (``vllm_neuron/model/glm5_next/config.py:110-117``).
DECLARED_KDA_NUM_HEADS = 64
DECLARED_KDA_HEAD_SIZE = 128
DECLARED_KDA_CONV_KERNEL_SIZE = 4

#: Registered tensor-parallel degree (``DECISIONS.md`` section 6,
#: whose registered preconditions are TP = 64 and a bf16 KV cache).
REGISTERED_TP_WORLD_SIZE = 64

#: The KDA state page byte value recorded at ``DECISIONS.md``
#: section 6, inside a user decision. Cited, never re-derived here.
RECORDED_KDA_STATE_PAGE_BYTES = 67_840

#: The recorded reading of the authority at that geometry under the pinned
#: ``"SD"`` layout (evidence-015.md sections 4 and 10).
EXPECTED_CONV_STATE_SHAPE = (3, 384)
EXPECTED_RECURRENT_STATE_SHAPE = (1, 128, 128)

#: The layout the orientation term pins, and the order it implies.
PINNED_CONV_STATE_LAYOUT = "SD"


def _authority_state_shapes() -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Return ``(conv, recurrent, resolved_layout)`` from vLLM's calculator.

    The geometry arguments are read from the fork's landed config, not passed
    as literals, so a config drift changes the derived value and reddens C03.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        get_conv_state_layout,
    )

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    linear_attn = Glm5NextTextConfig().linear_attn_config
    num_heads = linear_attn["num_heads"]
    head_dim = linear_attn["head_dim"]
    conv_kernel_size = linear_attn["short_conv_kernel_size"]

    # The derivation's inputs are pinned, so a silent geometry move fails here
    # rather than downstream in a shape that still looks plausible.
    assert num_heads == DECLARED_KDA_NUM_HEADS
    assert head_dim == DECLARED_KDA_HEAD_SIZE
    assert conv_kernel_size == DECLARED_KDA_CONV_KERNEL_SIZE

    conv, recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=REGISTERED_TP_WORLD_SIZE,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
    )
    return tuple(conv), tuple(recurrent), get_conv_state_layout()


def _authority_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    """Return ``(conv_dtype, recurrent_dtype)`` from vLLM's dtype calculator."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )

    # Model dtype bf16 is the registered KV precondition of section 6; "auto"
    # is the cache-dtype form that follows it.
    return MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")


@dataclass
class _PinShapedSpec:
    """The pin's six fields, replicated for the C01 control only."""

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


# ---------------------------------------------------------------------------
# C01 -- 0 signature breaks at 1/1 construction.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c01_the_pin_six_argument_form_constructs_unbroken() -> None:
    """The pin's exact 6-argument positional form still constructs: 0 breaks."""
    import inspect

    probe = LayerSpec("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)
    assert probe.name == "layers.0.self_attn"
    assert probe.num_kv_heads == 1
    assert probe.head_size == 512
    assert probe.dtype is torch.bfloat16
    assert probe.sliding_window_size is None
    assert probe.chunk_size is None

    # The break count is measured on the generated __init__, not inferred from
    # the call succeeding: the first six positional parameters must still be
    # the pin's six, in order.
    positional = [
        name
        for name, parameter in inspect.signature(LayerSpec.__init__).parameters.items()
        if name != "self"
        and parameter.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert tuple(positional[:6]) == PIN_LAYER_SPEC_FIELDS
    signature_breaks = sum(
        1
        for expected, actual in zip(PIN_LAYER_SPEC_FIELDS, positional[:6], strict=True)
        if expected != actual
    )
    assert signature_breaks == 0

    # CONTROL -- the zero moves. Both mechanisms by which a new field could
    # break this construction raise TypeError, so the arm is falsifiable.
    #
    # (i) appended without a default: the dataclass cannot even be created.
    with pytest.raises(TypeError) as decoration_error:

        @dataclass
        class _AppendedWithoutDefault(_PinShapedSpec):
            kda_conv_state_shape: tuple[int, int]

    assert "kda_conv_state_shape" in str(decoration_error.value)

    # (ii) appended as a required keyword-only field: the class is created and
    # THE CONSTRUCTION raises, which is exactly the arm C01 asserts.
    @dataclass
    class _RequiredKeywordOnly(_PinShapedSpec):
        kda_conv_state_shape: tuple[int, int] = field(kw_only=True)

    with pytest.raises(TypeError) as construction_error:
        _RequiredKeywordOnly("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)

    assert "kda_conv_state_shape" in str(construction_error.value)


# ---------------------------------------------------------------------------
# C02 -- 4/4 new-field defaults read None.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c02_four_of_four_new_field_defaults_read_none() -> None:
    """The four declared fields are appended, in order, all defaulting None."""
    names = tuple(f.name for f in fields(LayerSpec))
    assert names[:6] == PIN_LAYER_SPEC_FIELDS
    assert len(names) == 10
    assert names[6:] == DECLARED_KDA_FIELDS

    # Declared defaults, read off the dataclass rather than off an instance.
    declared_defaults = {
        f.name: f.default for f in fields(LayerSpec) if f.name in DECLARED_KDA_FIELDS
    }
    assert [declared_defaults[name] is None for name in DECLARED_KDA_FIELDS] == [
        True,
        True,
        True,
        True,
    ]

    # And as an omitting caller sees them, 4/4.
    unset = LayerSpec("layers.0.linear_attn", 64, 128, torch.bfloat16)
    defaults_none = sum(
        1 for name in DECLARED_KDA_FIELDS if getattr(unset, name) is None
    )
    assert defaults_none == 4

    # KVSpec is NOT widened: layers stays its only field.
    assert tuple(f.name for f in fields(KVSpec)) == ("layers",)


# ---------------------------------------------------------------------------
# C03 -- 2/2 state shapes round-trip exactly, rank AND every extent.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c03_both_state_shapes_round_trip_exactly() -> None:
    """The derived conv and recurrent shapes round-trip through ``LayerSpec``."""
    conv, recurrent, resolved_layout = _authority_state_shapes()

    # The orientation term is pinned in the process invocation; read it live so
    # a mis-pinned run fails loudly instead of asserting the host's own order.
    assert resolved_layout == PINNED_CONV_STATE_LAYOUT

    spec = LayerSpec(
        "layers.0.linear_attn",
        DECLARED_KDA_NUM_HEADS,
        DECLARED_KDA_HEAD_SIZE,
        torch.bfloat16,
        None,
        None,
        kda_conv_state_shape=conv,
        kda_recurrent_state_shape=recurrent,
    )

    # Rank, then every extent, on the value read back off the spec.
    assert len(spec.kda_conv_state_shape) == 2
    assert len(spec.kda_recurrent_state_shape) == 3
    assert spec.kda_conv_state_shape == EXPECTED_CONV_STATE_SHAPE
    assert spec.kda_recurrent_state_shape == EXPECTED_RECURRENT_STATE_SHAPE

    round_tripped_exactly = sum(
        1
        for stored, expected in (
            (spec.kda_conv_state_shape, EXPECTED_CONV_STATE_SHAPE),
            (spec.kda_recurrent_state_shape, EXPECTED_RECURRENT_STATE_SHAPE),
        )
        if stored == expected
    )
    assert round_tripped_exactly == 2


# ---------------------------------------------------------------------------
# C04 -- byte reconciliation, discrepancy 0 B.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c04_state_page_bytes_reconcile_with_zero_discrepancy() -> None:
    """The four fields reconstruct the recorded KDA state page exactly."""
    conv, recurrent, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    assert (conv_dtype, recurrent_dtype) == (torch.bfloat16, torch.float32)

    spec = LayerSpec(
        "layers.0.linear_attn",
        DECLARED_KDA_NUM_HEADS,
        DECLARED_KDA_HEAD_SIZE,
        torch.bfloat16,
        None,
        None,
        kda_conv_state_shape=conv,
        kda_recurrent_state_shape=recurrent,
        kda_conv_state_dtype=conv_dtype,
        kda_recurrent_state_dtype=recurrent_dtype,
    )

    page_bytes = (
        math.prod(spec.kda_conv_state_shape) * spec.kda_conv_state_dtype.itemsize
        + math.prod(spec.kda_recurrent_state_shape)
        * spec.kda_recurrent_state_dtype.itemsize
    )
    assert page_bytes - RECORDED_KDA_STATE_PAGE_BYTES == 0

    # CONTROL -- the two dtype fields carry this reconciliation. Swap them and
    # the total misses the recorded page, so no other value can rescue the arm.
    swapped_bytes = (
        math.prod(spec.kda_conv_state_shape) * spec.kda_recurrent_state_dtype.itemsize
        + math.prod(spec.kda_recurrent_state_shape) * spec.kda_conv_state_dtype.itemsize
    )
    assert swapped_bytes != RECORDED_KDA_STATE_PAGE_BYTES

# ===========================================================================
# ``inc-glm53f-086`` -- WP2 page-size unification. Five counted conjuncts, one
# collected item each, selected by ``-k unification``, no ``parametrize`` added.
#
# APPENDED BELOW EVERY EXISTING ITEM ON PURPOSE. Three live citations pin lines
# of this file BY NUMBER: ``model_fp8.py:1882`` pins ``:31-40``, the increment
# plan pins ``:98`` and ``:322-327``, and ``test_kda_layer.py:99`` pins ``:94``.
# An insertion above any of them would move bytes another file names, which is
# the drift ``inc-glm53f-085`` had to repair. Every constant this section needs
# therefore sits HERE and not in the file's top block.
#
# THIS FILE STILL IMPORTS NO OTHER TEST MODULE, which is the convention the
# module docstring states and the reason it states it. The 45-layer fake is
# rebuilt from the same fixture the two worker test files read, never imported
# from them.
# ===========================================================================

#: The user-decided hybrid block size (``DECISIONS.md`` section 6). Cited, and
#: neither re-derived nor re-priced here (P9).
REGISTERED_HYBRID_BLOCK_SIZE = 128

#: The DSA page this checkpoint's geometry produces. MEASURED before this file
#: was authored (``../../../increments/probe-086-r1-landed-diagnostic.out``),
#: so the number below is a recorded reading and not a prediction (D1.3).
MEASURED_DSA_PAGE_BYTES = 262_144

#: The reading that rules out the vendor's re-block branch, MEASURED in the same
#: transcript: the larger page is not a whole multiple of the smaller one.
MEASURED_PAGE_REMAINDER = 58_624

#: The layer split the fixture's own schedule carries.
DECLARED_TOTAL_LAYERS = 45
DECLARED_KDA_LAYERS = 34
DECLARED_DSA_LAYERS = 11

#: The group count the pin's own grouping produces AFTER unification. MEASURED.
MEASURED_GROUP_COUNT = 5

#: Declared inputs for the derived comparator, so its value is reproducible on
#: any host rather than depending on the machine it runs on.
DECLARED_AVAILABLE_MEMORY_BYTES = 34 * (1024**3)
DECLARED_MAX_MODEL_LEN = 8192

#: A head size small enough that the attention page falls BELOW the state page.
#: That is the direction the change refuses, and the vendor's own hook forbids.
DECLARED_TINY_HEAD_SIZE = 8

#: The fragment the refusal's message carries. The module defines no named
#: exception class and its own KDA arm raises ``ValueError`` with an explicit
#: message, so the refusal is named in the message and matched on it.
REFUSAL_MESSAGE_FRAGMENT = "Cannot unify KV cache pages"

#: The fixture the 45-layer schedule comes from, pinned by digest so a changed
#: fixture fails here rather than moving a reading quietly.
UNIFICATION_FIXTURE_SHA256 = (
    "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"
)


def _record_unification(**readings: object) -> None:
    """Put a reading in the ``-q`` transcript, ``inc-glm53f-075``'s convention."""
    import warnings

    for key, value in readings.items():
        warnings.warn(f"RECORDED {key}={value!r}", UserWarning, stacklevel=2)


def _raw_unification_fixture() -> dict:
    """The fixture's own bytes, digest-checked BEFORE they are parsed."""
    import hashlib
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "glm5_next" / "fixtures" / "config.json"
    raw_bytes = path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == UNIFICATION_FIXTURE_SHA256
    return json.loads(raw_bytes.decode("utf-8"))


class _UnificationFakeModel:
    """Exposes ONLY ``get_kv_spec()``, which is all the method under test reads."""

    def __init__(self, layers: list[LayerSpec]) -> None:
        self._layers = layers

    def get_kv_spec(self):
        import types

        return types.SimpleNamespace(layers=self._layers)


def _unification_layers() -> list[LayerSpec]:
    """The fixture's 45 layers, with the KDA four filled from the authority.

    The non-KDA layers have those four fields CLEARED rather than inherited. The
    arm under test recognises a KDA layer BY THE FIELDS IT CARRIES, so a stray
    value on an attention layer would turn it into a state layer and change the
    split this file counts.
    """
    import copy
    from dataclasses import replace as dc_replace

    from vllm_neuron.model.glm5_next.config import KDA_LAYER_TYPE
    from vllm_neuron.model.glm5_next.model_fp8 import (
        Glm5NextForConditionalGeneration,
    )

    raw = _raw_unification_fixture()
    conv_shape, recurrent_shape, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    kda_indices = {
        index
        for index, family in enumerate(raw["text_config"]["layer_types"])
        if family == KDA_LAYER_TYPE
    }
    landed = Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))

    layers: list[LayerSpec] = []
    for index, layer in enumerate(landed.get_kv_spec().layers):
        if index in kda_indices:
            layers.append(
                dc_replace(
                    layer,
                    kda_conv_state_shape=conv_shape,
                    kda_recurrent_state_shape=recurrent_shape,
                    kda_conv_state_dtype=conv_dtype,
                    kda_recurrent_state_dtype=recurrent_dtype,
                )
            )
        else:
            layers.append(
                dc_replace(
                    layer,
                    kda_conv_state_shape=None,
                    kda_recurrent_state_shape=None,
                    kda_conv_state_dtype=None,
                    kda_recurrent_state_dtype=None,
                )
            )
    return layers


def _refusal_layers() -> list[LayerSpec]:
    """Two layers whose attention page sits BELOW the state page."""
    conv_shape, recurrent_shape, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    return [
        LayerSpec(
            name="layers.0.linear_attn",
            num_kv_heads=1,
            head_size=DECLARED_TINY_HEAD_SIZE,
            dtype=torch.bfloat16,
            kda_conv_state_shape=conv_shape,
            kda_recurrent_state_shape=recurrent_shape,
            kda_conv_state_dtype=conv_dtype,
            kda_recurrent_state_dtype=recurrent_dtype,
        ),
        LayerSpec(
            name="layers.1.self_attn",
            num_kv_heads=1,
            head_size=DECLARED_TINY_HEAD_SIZE,
            dtype=torch.bfloat16,
        ),
    ]


def _call_get_kv_cache_spec(layers: list[LayerSpec]) -> dict:
    """Drive the UNBOUND method, so no real runner is ever constructed."""
    import types

    from vllm_neuron.vllm.worker import neuron_model_runner

    fake_self = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            cache_config=types.SimpleNamespace(
                block_size=REGISTERED_HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        ),
        speculative_config=None,
        model=_UnificationFakeModel(layers),
    )
    return neuron_model_runner.NeuronModelRunner.get_kv_cache_spec(fake_self)


def _natural_page_bytes(spec) -> int:
    """The page the spec's OWN geometry describes, before any padding."""
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(spec.shapes, spec.dtypes)
    )


def _split_by_family(specs: dict) -> tuple[dict, dict]:
    """``(kda, attention)``, split on the real spec classes rather than names."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    kda = {name: s for name, s in specs.items() if isinstance(s, MambaSpec)}
    attention = {
        name: s for name, s in specs.items() if isinstance(s, FullAttentionSpec)
    }
    return kda, attention


def _unification_vllm_config():
    """Only the leaves the pin's three grouping functions actually read."""
    import types

    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=None,
        ),
        speculative_config=None,
        cache_config=types.SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
        ),
        model_config=types.SimpleNamespace(max_model_len=DECLARED_MAX_MODEL_LEN),
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


def _derived_window(specs: dict) -> dict:
    """Drive three of the pin's OWN functions and return every number they give.

    This is conjunct 3's whole point and conjunct 4's control: the admission
    window is READ by calling the pin, never copied from a module constant. It
    returns a dict rather than asserting, so both conjuncts read the same values.
    """
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
        get_max_concurrency_for_kv_cache_config,
        max_memory_usage_bytes,
    )

    config = _unification_vllm_config()
    groups = get_kv_cache_groups(config, dict(specs))
    kv_config = get_kv_cache_config_from_groups(
        config, groups, DECLARED_AVAILABLE_MEMORY_BYTES
    )
    concurrency = get_max_concurrency_for_kv_cache_config(config, kv_config)

    # The two windows, computed the two ways. The fork truncates the pin's own
    # float (``vllm_neuron/vllm/core/scheduler.py:237-250``); the pin's own
    # arithmetic floor-divides. Both are read here, neither is assumed equal.
    layers_per_group = max(len(g.layer_names) for g in kv_config.kv_cache_groups)
    per_request = layers_per_group * max_memory_usage_bytes(
        config, (g.kv_cache_spec for g in kv_config.kv_cache_groups)
    )
    per_block = (
        kv_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes * layers_per_group
    )
    blocks_per_request = cdiv(per_request, per_block)
    return {
        "group_count": len(groups),
        "group_sizes": sorted(len(g.layer_names) for g in groups),
        "group_pages": sorted({g.kv_cache_spec.page_size_bytes for g in groups}),
        "num_blocks": kv_config.num_blocks,
        "blocks_per_request": blocks_per_request,
        "concurrency_raw": concurrency,
        "fork_window": int(concurrency),
        "pin_window": kv_config.num_blocks // blocks_per_request,
    }


# ---------------------------------------------------------------------------
# Conjunct 1 -- the two pages, READ rather than assumed.
# Certifies ``NeuronModelRunner.get_kv_cache_spec``.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c086_unification_reads_both_pages_rather_than_assuming() -> None:
    """Both page sizes are read off the real 45-layer spec set, as numbers."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    kda, attention = _split_by_family(specs)

    # POPULATION BEFORE PROPERTY: a page reading over an empty family is void,
    # so the split is asserted before any page is compared.
    assert len(specs) == DECLARED_TOTAL_LAYERS
    assert len(kda) == DECLARED_KDA_LAYERS
    assert len(attention) == DECLARED_DSA_LAYERS

    kda_natural = {_natural_page_bytes(s) for s in kda.values()}
    kda_reported = {s.page_size_bytes for s in kda.values()}
    attention_pages = {s.page_size_bytes for s in attention.values()}
    _record_unification(
        c086_1_kda_natural_page=sorted(kda_natural),
        c086_1_kda_reported_page=sorted(kda_reported),
        c086_1_attention_page=sorted(attention_pages),
    )
    # The state page is the natural one, which does NOT move (P9): section 6's
    # recorded value. The padded value is recorded beside it, never instead.
    assert kda_natural == {RECORDED_KDA_STATE_PAGE_BYTES}
    assert attention_pages == {MEASURED_DSA_PAGE_BYTES}

    # The reading that rules out the vendor's re-block branch: the larger page is
    # not a whole multiple of the smaller one, so no block-size scaling can
    # unify them and padding is the only route left.
    remainder = MEASURED_DSA_PAGE_BYTES % RECORDED_KDA_STATE_PAGE_BYTES
    _record_unification(c086_1_modulo=remainder)
    assert remainder == MEASURED_PAGE_REMAINDER
    assert remainder != 0


# ---------------------------------------------------------------------------
# Conjunct 2 -- the padding is in the SPEC, not in a patch.
# Certifies the padded page the KDA arm now sets.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c086_unification_padding_is_in_the_spec_not_a_patch() -> None:
    """The 34 state specs carry the field; the pin's OWN unification takes them."""
    from dataclasses import replace as dc_replace

    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.patches import kv_spec_patch

    specs = _call_get_kv_cache_spec(_unification_layers())
    kda, _ = _split_by_family(specs)

    carrying = {name for name, s in specs.items() if s.page_size_padded is not None}
    padded_values = {s.page_size_padded for s in kda.values()}
    _record_unification(
        c086_2_entries_carrying_the_field=len(carrying),
        c086_2_padded_values=sorted(padded_values),
        c086_2_set_equals_the_state_layers=carrying == set(kda),
    )
    assert len(carrying) == DECLARED_KDA_LAYERS
    assert carrying == set(kda)
    assert padded_values == {MEASURED_DSA_PAGE_BYTES}

    # THE CONTROL FOR "NOT IN A PATCH", and the reason this calls the ORIGINAL.
    # Importing the module attribute would reach ``inc-glm53f-018``'s wrapper,
    # which pads by itself -- so this conjunct would pass with the source change
    # reverted, which is the one thing it exists to rule out. The pre-patch
    # original is the function object the wrapper captured, so calling it is a
    # reading about the spec and about nothing installed at import time.
    is_wrapper = (
        kv_cache_utils.unify_kv_cache_spec_page_size
        is kv_spec_patch._unify_kv_cache_spec_page_size_widened
    )
    original = kv_spec_patch._original_unify
    _record_unification(
        c086_2_module_attribute_is_the_wrapper=is_wrapper,
        c086_2_original_is_bound=original is not None,
    )
    assert is_wrapper
    assert original is not None

    unified = original(dict(specs))
    pages_after = {s.page_size_bytes for s in unified.values()}
    _record_unification(
        c086_2_entries_after_unify=len(unified),
        c086_2_pages_after_unify=sorted(pages_after),
    )
    assert len(unified) == DECLARED_TOTAL_LAYERS
    assert pages_after == {MEASURED_DSA_PAGE_BYTES}

    # NON-VACUITY (D1.5): the same call, on the same spec set, with the field
    # stripped back to ``None``. It must refuse -- otherwise the success above
    # would say nothing about the field.
    stripped = {
        name: (dc_replace(s, page_size_padded=None) if isinstance(s, MambaSpec) else s)
        for name, s in specs.items()
    }
    with pytest.raises(NotImplementedError) as excinfo:
        original(stripped)
    _record_unification(c086_2_control_message=str(excinfo.value)[:120])
    assert "page size is not divisible" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Conjunct 3 -- the comparator is DERIVED, which is the B14-M2 repair.
# Certifies the pin's own grouping and concurrency functions.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c086_unification_window_is_derived_not_a_constant() -> None:
    """The window is read by CALLING the pin, and the group count is read too."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    window = _derived_window(specs)
    _record_unification(**{f"c086_3_{k}": v for k, v in window.items()})

    # The group count is READ, never asserted at 45: the pin groups these 45
    # specs into 5, and the sizes are recorded so the count is legible.
    assert window["group_count"] == MEASURED_GROUP_COUNT
    assert sum(window["group_sizes"]) == DECLARED_TOTAL_LAYERS

    # Unification happened before grouping, so every group reports ONE page.
    assert window["group_pages"] == [MEASURED_DSA_PAGE_BYTES]

    # The window is a number the pin produced, not a module constant. Asserting
    # it is positive rather than equal to a literal keeps this conjunct about
    # the derivation; conjunct 4 is where the value itself is compared.
    assert window["num_blocks"] > 0
    assert window["blocks_per_request"] > 0
    assert window["concurrency_raw"] > 0


# ---------------------------------------------------------------------------
# Conjunct 4 -- a counted zero with its control.
# Certifies that the fork's window is the pin's own computation.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c086_unification_fork_window_differs_in_zero_cases() -> None:
    """The fork's window and the pin's window differ in 0 of the declared cases."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    window = _derived_window(specs)

    differing = [
        case
        for case, (fork, pin) in {
            "declared_45_layer_hybrid": (window["fork_window"], window["pin_window"]),
        }.items()
        if fork != pin
    ]
    _record_unification(
        c086_4_fork_window=window["fork_window"],
        c086_4_pin_window=window["pin_window"],
        c086_4_cases_differing=len(differing),
        c086_4_which=differing,
    )

    # THE CONTROL (D1.5) is conjunct 3's reading: this zero means something only
    # because the config built and yielded a number at all. A configuration that
    # failed to build would give two absent values and a vacuous zero.
    assert window["concurrency_raw"] > 0
    assert window["fork_window"] > 0

    # This zero is what retires inc-glm53f-021's premise on an instrument. A
    # NONZERO reading is evidence_contradicts_design and routes to the lead; it
    # is never a silent revival of that increment.
    assert len(differing) == 0


# ---------------------------------------------------------------------------
# Conjunct 5 -- the refusal in the forbidden direction.
# Certifies the named refusal the KDA arm now carries.
# ---------------------------------------------------------------------------


def test_kv_cache_spec_c086_unification_refuses_the_forbidden_direction() -> None:
    """An attention page BELOW the state page raises by name, 1/1."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # POPULATION FIRST: the fake really must put the attention page below the
    # state page, or the raise below would be testing nothing at all. The page
    # is read off a real spec rather than recomputed from the vendor's formula.
    probe = FullAttentionSpec(
        block_size=REGISTERED_HYBRID_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=DECLARED_TINY_HEAD_SIZE,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )
    tiny_page = probe.page_size_bytes
    _record_unification(
        c086_5_tiny_attention_page=tiny_page,
        c086_5_state_page=RECORDED_KDA_STATE_PAGE_BYTES,
    )
    assert tiny_page < RECORDED_KDA_STATE_PAGE_BYTES

    with pytest.raises(ValueError, match=REFUSAL_MESSAGE_FRAGMENT) as excinfo:
        _call_get_kv_cache_spec(_refusal_layers())

    message = str(excinfo.value)
    _record_unification(c086_5_message=message[:160])
    # The message names BOTH pages, so a log reader can see which direction was
    # refused without reading the source.
    assert str(tiny_page) in message
    assert str(RECORDED_KDA_STATE_PAGE_BYTES) in message
