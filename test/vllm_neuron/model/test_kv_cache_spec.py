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
fork's landed ``linear_attn_config`` carries (``config.py:110-117``) and at the
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

#: The pin's ``LayerSpec`` field set, in order (``kv_cache.py:16-21``). The
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

#: Registered tensor-parallel degree (``approvals/DECISIONS.md`` section 6,
#: whose registered preconditions are TP = 64 and a bf16 KV cache).
REGISTERED_TP_WORLD_SIZE = 64

#: The KDA state page byte value recorded at ``approvals/DECISIONS.md``
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
