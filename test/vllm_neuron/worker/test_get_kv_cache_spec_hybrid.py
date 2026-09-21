# SPDX-License-Identifier: Apache-2.0
"""``NeuronModelRunner.get_kv_cache_spec`` on a hybrid recurrent/latent stack.

The runner translates the model's ``LayerSpec`` list into vLLM's KV cache spec
dict. The model here is a fake exposing only ``get_kv_spec()``, driven through
the unbound method so no runner, device or weights are built. Every recurrent
state shape and dtype is derived by calling vLLM's own state calculators at the
45-layer fixture's geometry, so no field value is hand-written.

Run with ``VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2
VLLM_SSM_CONV_STATE_LAYOUT=SD``: the conv layout is pinned only so the derived
shapes are deterministic, and every count below is a product of extents, so no
assertion here depends on the conv orientation.

The attention branch keeps the global KV cache dtype; only the recurrent-state
branch reads the per-state dtypes.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_neuron.model.kv_cache import LayerSpec

#: The 45-layer GLM fixture, the same one the model tests read.
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "glm5_next"
    / "fixtures"
    / "config.json"
)

#: The stack these specs describe: tensor-parallel degree, KV block size, and
#: the bytes one recurrent state occupies at that degree.
TP_WORLD_SIZE = 64
HYBRID_BLOCK_SIZE = 128
KDA_STATE_PAGE_BYTES = 67_840

TOTAL_ENTRIES = 45
KDA_ENTRIES = 34
DSA_ENTRIES = 11

#: The page every KDA entry reports: the attention page its own smaller state
#: page is padded up to.
PADDED_PAGE_BYTES = 131_072


def _vendor_state_shapes() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(conv, recurrent)`` state shapes from vLLM's shape calculator."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    linear_attn = Glm5NextTextConfig().linear_attn_config
    conv, recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=TP_WORLD_SIZE,
        num_heads=linear_attn["num_heads"],
        head_dim=linear_attn["head_dim"],
        conv_kernel_size=linear_attn["short_conv_kernel_size"],
    )
    return tuple(conv), tuple(recurrent)


def _vendor_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    """``(conv_dtype, recurrent_dtype)`` from vLLM's dtype calculator."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    # bf16 model dtype with cache dtype "auto" is the precondition this stack
    # serves under.
    return MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")


def _raw_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _fake_layers(
    raw: dict, *, populate_kda: bool = True, name_blind: bool = False
) -> list[LayerSpec]:
    """The fake model's 45 layers, each field derived rather than written out.

    ``populate_kda=False`` clears the four recurrent-state fields to ``None``;
    ``name_blind=True`` strips every family suffix from the layer names.
    """
    from vllm_neuron.model.glm5_next.config import KDA_LAYER_TYPE
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    conv_shape, recurrent_shape = _vendor_state_shapes()
    conv_dtype, recurrent_dtype = _vendor_state_dtypes()
    kda = {
        index
        for index, family in enumerate(raw["text_config"]["layer_types"])
        if family == KDA_LAYER_TYPE
    }
    real_model = Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))

    layers: list[LayerSpec] = []
    for index, layer in enumerate(real_model.get_kv_spec().layers):
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
            # Cleared explicitly rather than inherited: the real model sets these
            # four fields on its 34 linear-attention layers, so a bare
            # `replace(layer, name=name)` would preserve real values here and the
            # "fields are unset" case would quietly become a populated one.
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
    """A model exposing only ``get_kv_spec()``."""

    def __init__(self, layers: list[LayerSpec]) -> None:
        self._spec = SimpleNamespace(layers=layers)

    def get_kv_spec(self):
        return self._spec


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


def _call(layers: list[LayerSpec]) -> dict:
    """Drive the unbound method, so no real runner is constructed."""
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        speculative_config=None,
        model=_FakeModel(layers),
    )
    return _runner_module().NeuronModelRunner.get_kv_cache_spec(fake_self)


def _dtype_mismatches(specs: dict, layers: list[LayerSpec]) -> list[str]:
    """Entries whose returned dtype carrier differs from the model's.

    The recurrent spec class has no ``dtype`` field -- its carrier is the
    ``dtypes`` tuple -- while the attention class carries a single ``dtype``.
    Both normalise to a tuple compared element for element in construction order
    (position 0 conv, position 1 recurrent), so an arity difference counts as a
    difference and a truncated tuple cannot pass as a match.
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
            got is not expected
            for got, expected in zip(returned, reported, strict=True)
        ):
            mismatched.append(layer.name)
    return mismatched


def _natural_state_pages(kda_specs: list) -> set:
    """The bytes each recurrent state's own geometry occupies.

    ``page_size_bytes`` answers with the padded page, so the state's own size is
    summed from the shapes and dtypes the spec carries.
    """
    return {
        sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(spec.shapes, spec.dtypes)
        )
        for spec in kda_specs
    }


def _non_none(mapping: dict) -> dict:
    return {name: value for name, value in mapping.items() if value is not None}


def _padded_page_expected_for_kda(specs: dict) -> dict:
    """Every KDA name mapped to the padded page, and no other name present."""
    from vllm.v1.kv_cache_interface import MambaSpec

    return {
        name: PADDED_PAGE_BYTES
        for name, spec in specs.items()
        if isinstance(spec, MambaSpec)
    }


def test_the_split_is_thirty_four_recurrent_and_eleven_attention_specs() -> None:
    """34 recurrent-state specs, 11 attention specs, no other class."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    specs = _call(_fake_layers(raw))

    assert len(specs) == TOTAL_ENTRIES
    kda = sum(1 for spec in specs.values() if isinstance(spec, MambaSpec))
    dsa = sum(1 for spec in specs.values() if isinstance(spec, FullAttentionSpec))
    other = len(specs) - kda - dsa
    assert (kda, dsa, other) == (KDA_ENTRIES, DSA_ENTRIES, 0)

    # The branch selects on a LayerSpec field, so stripping every family suffix
    # from the layer names must leave the split where it was.
    blind_layers = _fake_layers(raw, name_blind=True)
    assert not any(
        layer.name.endswith((".linear_attn", ".self_attn")) for layer in blind_layers
    )
    blind = _call(blind_layers)
    blind_kda = sum(1 for spec in blind.values() if isinstance(spec, MambaSpec))
    assert (blind_kda, len(blind) - blind_kda) == (KDA_ENTRIES, DSA_ENTRIES)


def test_every_entry_carries_the_dtype_the_model_reports_for_that_state() -> None:
    """No entry's dtype carrier differs from what the model reports."""
    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)

    assert len(specs) == TOTAL_ENTRIES
    assert _dtype_mismatches(specs, layers) == []

    # The recurrent half is per state, not one dtype repeated: the two vendor
    # dtypes differ, so a single-dtype assignment could not pass the line above.
    conv_dtype, recurrent_dtype = _vendor_state_dtypes()
    assert conv_dtype is not recurrent_dtype


def test_the_recurrent_state_fields_are_what_selects_the_branch() -> None:
    """With the four state fields cleared the recurrent branch does not fire."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    raw = _raw_fixture()
    populated = _call(_fake_layers(raw, populate_kda=True))
    engaged = sum(1 for spec in populated.values() if isinstance(spec, MambaSpec))

    # Same fake, four fields None: the branch must not fire, and the dict must
    # come back as 45 attention specs.
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

    from vllm_neuron.model.glm5_next.model_fp8 import (
        Glm5NextForConditionalGeneration,
    )

    real_layers_with_state = sum(
        1
        for layer in Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))
        .get_kv_spec()
        .layers
        if layer.kda_conv_state_shape is not None
    )
    assert real_layers_with_state == KDA_ENTRIES, (
        f"the real spec carries KDA fields on {real_layers_with_state} layers, so "
        f"the cleared case above must clear them rather than inherit them"
    )

    assert (engaged, not_engaged) == (KDA_ENTRIES, 0)
    assert len(emptied) == TOTAL_ENTRIES
    assert all(isinstance(spec, FullAttentionSpec) for spec in emptied.values())


def test_the_recurrent_entries_report_the_padded_attention_page() -> None:
    """One page across the 34 recurrent entries, above each state's own size."""
    from vllm.v1.kv_cache_interface import MambaSpec

    specs = _call(_fake_layers(_raw_fixture()))
    kda_specs = [spec for spec in specs.values() if isinstance(spec, MambaSpec)]
    assert len(kda_specs) == KDA_ENTRIES

    pages = {spec.page_size_bytes for spec in kda_specs}
    assert len(pages) == 1
    assert sorted(_natural_state_pages(kda_specs)) == [KDA_STATE_PAGE_BYTES]
    # The reported page is the padded one, not the state's own geometry.
    assert sorted(pages) != sorted(_natural_state_pages(kda_specs))

    # page_size_padded carries the unified page on the 34 recurrent entries and
    # stays None on the 11 attention ones.
    padded = {name: spec.page_size_padded for name, spec in specs.items()}
    assert len(padded) == TOTAL_ENTRIES
    assert _non_none(padded) == _padded_page_expected_for_kda(specs)

    # Both carriers are length two on every recurrent entry, so the vendor's
    # pairing of shapes with dtypes cannot truncate the page sum in silence.
    assert {(len(spec.shapes), len(spec.dtypes)) for spec in kda_specs} == {(2, 2)}
