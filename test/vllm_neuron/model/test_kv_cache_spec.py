# SPDX-License-Identifier: Apache-2.0
"""``LayerSpec``'s four KDA state fields, and the page unification they feed.

The six-argument construction form still works, the new fields default to
``None``, both state shapes round-trip, and the runner's spec set unifies the
KDA state page with the larger DSA attention page by padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import pytest
import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec


KDA_STATE_FIELDS = (
    "kda_conv_state_shape",
    "kda_recurrent_state_shape",
    "kda_conv_state_dtype",
    "kda_recurrent_state_dtype",
)

#: ``LayerSpec``'s field set before the KDA fields were added, in order. The
#: widening must leave these six as an in-order prefix, because callers pass them
#: positionally.
ORIGINAL_LAYER_SPEC_FIELDS = (
    "name",
    "num_kv_heads",
    "head_size",
    "dtype",
    "sliding_window_size",
    "chunk_size",
)

#: KDA geometry, as ``Glm5NextTextConfig.linear_attn_config`` carries it.
KDA_NUM_HEADS = 64
KDA_HEAD_SIZE = 128
KDA_CONV_KERNEL_SIZE = 4

#: The degree the hybrid stack runs at, which is what makes the per-rank state
#: shapes below one head wide.
TP_WORLD_SIZE = 64

#: The state shapes that geometry gives per rank under the ``"sd"`` conv layout.
#: The conv state holds ``conv_kernel_size - 1`` history steps over the three
#: short-convolved projections, each ``KDA_NUM_HEADS * KDA_HEAD_SIZE /
#: TP_WORLD_SIZE`` wide, so ``(3, 3 * 128)``. The recurrent state is one
#: ``head_dim x head_dim`` matrix per rank-local head, so ``(1, 128, 128)``.
EXPECTED_CONV_STATE_SHAPE = (3, 384)
EXPECTED_RECURRENT_STATE_SHAPE = (1, 128, 128)

#: Those two shapes at their own dtypes: 3 * 384 bfloat16 elements (2,304 B) plus
#: 128 * 128 float32 elements (65,536 B).
EXPECTED_KDA_STATE_PAGE_BYTES = 67_840

#: The conv-state layout the shapes above assume, and the order it implies.
EXPECTED_CONV_STATE_LAYOUT = "SD"


def _authority_state_shapes() -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Return ``(conv, recurrent, resolved_layout)`` from vLLM's calculator."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        get_conv_state_layout,
    )

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    linear_attn = Glm5NextTextConfig().linear_attn_config
    num_heads = linear_attn["num_heads"]
    head_dim = linear_attn["head_dim"]
    conv_kernel_size = linear_attn["short_conv_kernel_size"]

    # The derivation's inputs are checked, so a silent geometry move fails here
    # rather than downstream in a shape that still looks plausible.
    assert num_heads == KDA_NUM_HEADS
    assert head_dim == KDA_HEAD_SIZE
    assert conv_kernel_size == KDA_CONV_KERNEL_SIZE

    conv, recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=TP_WORLD_SIZE,
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

    # A bfloat16 model dtype is this stack's precondition for the hybrid cache,
    # and "auto" is the cache dtype that follows from it.
    return MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")


@dataclass
class _SixFieldSpec:
    """The six original fields, replicated for the control in the first test."""

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


def test_the_original_six_argument_form_still_constructs() -> None:
    """The six-argument positional form is unchanged by the widening."""
    import inspect

    probe = LayerSpec("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)
    assert probe.name == "layers.0.self_attn"
    assert probe.num_kv_heads == 1
    assert probe.head_size == 512
    assert probe.dtype is torch.bfloat16
    assert probe.sliding_window_size is None
    assert probe.chunk_size is None

    # Read off the generated __init__ rather than inferred from the call
    # succeeding: the first six positional parameters must still be those six,
    # in order.
    positional = [
        name
        for name, parameter in inspect.signature(LayerSpec.__init__).parameters.items()
        if name != "self"
        and parameter.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert tuple(positional[:6]) == ORIGINAL_LAYER_SPEC_FIELDS

    # Control: both mechanisms by which a new field could break this
    # construction raise TypeError, so the check above is falsifiable.
    #
    # (i) appended without a default: the dataclass cannot even be created.
    with pytest.raises(TypeError) as decoration_error:

        @dataclass
        class _AppendedWithoutDefault(_SixFieldSpec):
            kda_conv_state_shape: tuple[int, int]

    assert "kda_conv_state_shape" in str(decoration_error.value)

    # (ii) appended as a required keyword-only field: the class is created and
    # the construction is what raises.
    @dataclass
    class _RequiredKeywordOnly(_SixFieldSpec):
        kda_conv_state_shape: tuple[int, int] = field(kw_only=True)

    with pytest.raises(TypeError) as construction_error:
        _RequiredKeywordOnly("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)

    assert "kda_conv_state_shape" in str(construction_error.value)


def test_the_kda_state_fields_default_to_none() -> None:
    """The four KDA fields are appended, in order, all defaulting to ``None``."""
    names = tuple(f.name for f in fields(LayerSpec))
    assert names[:6] == ORIGINAL_LAYER_SPEC_FIELDS
    assert len(names) == 11
    assert names[6:10] == KDA_STATE_FIELDS
    assert names[10] == "latent_kv"
    latent_default = next(f.default for f in fields(LayerSpec) if f.name == "latent_kv")
    assert latent_default is False

    # Declared defaults, read off the dataclass rather than off an instance.
    declared_defaults = {
        f.name: f.default for f in fields(LayerSpec) if f.name in KDA_STATE_FIELDS
    }
    assert all(declared_defaults[name] is None for name in KDA_STATE_FIELDS)

    # And as an omitting caller sees them.
    unset = LayerSpec("layers.0.linear_attn", 64, 128, torch.bfloat16)
    assert all(getattr(unset, name) is None for name in KDA_STATE_FIELDS)

    # KVSpec is not widened: layers stays its only field.
    assert tuple(f.name for f in fields(KVSpec)) == ("layers",)


def test_both_state_shapes_round_trip_exactly() -> None:
    """The derived conv and recurrent shapes round-trip through ``LayerSpec``."""
    conv, recurrent, resolved_layout = _authority_state_shapes()

    # The layout is set in the process environment; read it live so a
    # misconfigured run fails loudly instead of asserting the host's own order.
    assert resolved_layout == EXPECTED_CONV_STATE_LAYOUT

    spec = LayerSpec(
        "layers.0.linear_attn",
        KDA_NUM_HEADS,
        KDA_HEAD_SIZE,
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


def test_state_page_bytes_reconcile() -> None:
    """The four fields reconstruct the KDA state page exactly."""
    conv, recurrent, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    assert (conv_dtype, recurrent_dtype) == (torch.bfloat16, torch.float32)

    spec = LayerSpec(
        "layers.0.linear_attn",
        KDA_NUM_HEADS,
        KDA_HEAD_SIZE,
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
    assert page_bytes == EXPECTED_KDA_STATE_PAGE_BYTES

    # Control: the two dtype fields are what carry this total. Swap them and it
    # misses the page, so no other value can rescue the check above.
    swapped_bytes = (
        math.prod(spec.kda_conv_state_shape) * spec.kda_recurrent_state_dtype.itemsize
        + math.prod(spec.kda_recurrent_state_shape) * spec.kda_conv_state_dtype.itemsize
    )
    assert swapped_bytes != EXPECTED_KDA_STATE_PAGE_BYTES


HYBRID_BLOCK_SIZE = 128

#: The DSA attention page at this geometry: one 512-wide latent entry per token,
#: bfloat16, over a 128-token block.
EXPECTED_DSA_PAGE_BYTES = 131_072

#: 131,072 - 67,840. Stated rather than derived, because the assertion below
#: computes the same modulo from the same two operands and would otherwise
#: compare a value against itself.
EXPECTED_PAGE_REMAINDER = 63_232

#: The layer split the fixture checkpoint's own schedule carries.
TOTAL_LAYERS = 45
KDA_LAYERS = 34
DSA_LAYERS = 11

#: How many groups vLLM's own grouping produces from those 45 specs.
EXPECTED_GROUP_COUNT = 5

#: Fixed inputs for the derived window, so its value is reproducible on any host
#: rather than depending on the machine it runs on.
AVAILABLE_MEMORY_BYTES = 34 * (1024**3)
MAX_MODEL_LEN = 8192

#: A head size small enough that the attention page falls below the state page.
#: That is the direction unification refuses.
TINY_HEAD_SIZE = 8

#: The fragment the refusal's message carries. The module defines no named
#: exception class and its KDA arm raises ``ValueError`` with an explicit
#: message, so the refusal is named in the message and matched on it.
REFUSAL_MESSAGE_FRAGMENT = "Cannot unify KV cache pages"


def _raw_unification_fixture() -> dict:
    """The 45-layer schedule, parsed from the fixture checkpoint's own config."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "glm5_next" / "fixtures" / "config.json"
    return json.loads(path.read_bytes().decode("utf-8"))


class _UnificationFakeModel:
    """Exposes only ``get_kv_spec()``, which is all the method under test reads."""

    def __init__(self, layers: list[LayerSpec]) -> None:
        self._layers = layers

    def get_kv_spec(self):
        import types

        return types.SimpleNamespace(layers=self._layers)


def _unification_layers() -> list[LayerSpec]:
    """The fixture's 45 layers, with the four KDA fields filled in."""
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
    model = Glm5NextForConditionalGeneration.from_configs(copy.deepcopy(raw))

    layers: list[LayerSpec] = []
    for index, layer in enumerate(model.get_kv_spec().layers):
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
    """Two layers whose attention page sits below the state page."""
    conv_shape, recurrent_shape, _ = _authority_state_shapes()
    conv_dtype, recurrent_dtype = _authority_state_dtypes()
    return [
        LayerSpec(
            name="layers.0.linear_attn",
            num_kv_heads=1,
            head_size=TINY_HEAD_SIZE,
            dtype=torch.bfloat16,
            kda_conv_state_shape=conv_shape,
            kda_recurrent_state_shape=recurrent_shape,
            kda_conv_state_dtype=conv_dtype,
            kda_recurrent_state_dtype=recurrent_dtype,
        ),
        LayerSpec(
            name="layers.1.self_attn",
            num_kv_heads=1,
            head_size=TINY_HEAD_SIZE,
            dtype=torch.bfloat16,
        ),
    ]


def _call_get_kv_cache_spec(layers: list[LayerSpec]) -> dict:
    """Drive the unbound method, so no real runner is ever constructed."""
    import types

    from vllm_neuron.vllm.worker import neuron_model_runner

    fake_self = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            cache_config=types.SimpleNamespace(
                block_size=HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        ),
        speculative_config=None,
        model=_UnificationFakeModel(layers),
    )
    return neuron_model_runner.NeuronModelRunner.get_kv_cache_spec(fake_self)


def _natural_page_bytes(spec) -> int:
    """The page the spec's own geometry describes, before any padding."""
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
    """Only the leaves vLLM's three grouping functions actually read."""
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
        model_config=types.SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


def _derived_window(specs: dict) -> dict:
    """Drive three of vLLM's own functions and return every number they give."""
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
        config, groups, AVAILABLE_MEMORY_BYTES
    )
    concurrency = get_max_concurrency_for_kv_cache_config(config, kv_config)

    # The same window computed two ways. The scheduler truncates vLLM's float;
    # vLLM's own arithmetic floor-divides. Both are read here, neither is
    # assumed equal to the other.
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
        "truncated_window": int(concurrency),
        "floor_divided_window": kv_config.num_blocks // blocks_per_request,
    }


def test_unification_reads_both_page_sizes() -> None:
    """Both page sizes come off the real 45-layer spec set, as numbers."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    kda, attention = _split_by_family(specs)

    # Population before property: a page reading over an empty family is void,
    # so the split is asserted before any page is compared.
    assert len(specs) == TOTAL_LAYERS
    assert len(kda) == KDA_LAYERS
    assert len(attention) == DSA_LAYERS

    kda_natural = {_natural_page_bytes(s) for s in kda.values()}
    attention_pages = {s.page_size_bytes for s in attention.values()}
    assert kda_natural == {EXPECTED_KDA_STATE_PAGE_BYTES}
    assert attention_pages == {EXPECTED_DSA_PAGE_BYTES}

    # The reading that rules out re-blocking: the larger page is not a whole
    # multiple of the smaller one, so no block-size scaling can unify them and
    # padding is the only route left.
    remainder = EXPECTED_DSA_PAGE_BYTES % EXPECTED_KDA_STATE_PAGE_BYTES
    assert remainder == EXPECTED_PAGE_REMAINDER
    assert remainder != 0


def test_unification_padding_is_carried_by_the_spec() -> None:
    """The 34 state specs carry ``page_size_padded``, and unification takes it."""
    from dataclasses import replace as dc_replace

    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.patches import kv_spec_patch

    specs = _call_get_kv_cache_spec(_unification_layers())
    kda, _ = _split_by_family(specs)

    carrying = {name for name, s in specs.items() if s.page_size_padded is not None}
    padded_values = {s.page_size_padded for s in kda.values()}
    assert len(carrying) == KDA_LAYERS
    assert carrying == set(kda)
    assert padded_values == {EXPECTED_DSA_PAGE_BYTES}

    # Why this calls the pre-patch original rather than the module attribute:
    # the attribute is the patch's wrapper, which pads by itself, so the check
    # below would pass with the spec change reverted -- the one thing it exists
    # to rule out. The original is the function object the wrapper captured, so
    # calling it reads the spec and nothing installed at import time.
    is_wrapper = (
        kv_cache_utils.unify_kv_cache_spec_page_size
        is kv_spec_patch._unify_kv_cache_spec_page_size_widened
    )
    original = kv_spec_patch._original_unify
    assert is_wrapper
    assert original is not None

    unified = original(dict(specs))
    pages_after = {s.page_size_bytes for s in unified.values()}
    assert len(unified) == TOTAL_LAYERS
    assert pages_after == {EXPECTED_DSA_PAGE_BYTES}

    # Control: the same call, on the same spec set, with the field stripped back
    # to None. It must refuse -- otherwise the success above would say nothing
    # about the field.
    stripped = {
        name: (dc_replace(s, page_size_padded=None) if isinstance(s, MambaSpec) else s)
        for name, s in specs.items()
    }
    with pytest.raises(NotImplementedError) as excinfo:
        original(stripped)
    assert "page size is not divisible" in str(excinfo.value)


def test_unification_window_is_derived_from_the_spec_set() -> None:
    """The window is read by calling vLLM, and so is the group count."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    window = _derived_window(specs)

    assert window["group_count"] == EXPECTED_GROUP_COUNT
    assert sum(window["group_sizes"]) == TOTAL_LAYERS

    # Unification happened before grouping, so every group reports one page.
    assert window["group_pages"] == [EXPECTED_DSA_PAGE_BYTES]

    # The window is a number vLLM produced, not a module constant. Asserting it
    # is positive rather than equal to a literal keeps this check about the
    # derivation; the next test compares the two values against each other.
    assert window["num_blocks"] > 0
    assert window["blocks_per_request"] > 0
    assert window["concurrency_raw"] > 0


def test_scheduler_window_matches_upstream_arithmetic() -> None:
    """Truncating the float and floor-dividing give the same window here."""
    specs = _call_get_kv_cache_spec(_unification_layers())
    window = _derived_window(specs)

    # Both values have to exist before their equality means anything: a
    # configuration that failed to build would compare two absent numbers.
    assert window["concurrency_raw"] > 0
    assert window["truncated_window"] > 0

    assert window["truncated_window"] == window["floor_divided_window"]


def test_unification_refuses_an_attention_page_below_the_state_page() -> None:
    """The direction that cannot be padded raises, and names both pages."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # Population first: the fake really must put the attention page below the
    # state page, or the raise below would be testing nothing at all. The page
    # is read off a real spec rather than recomputed from vLLM's formula.
    probe = FullAttentionSpec(
        block_size=HYBRID_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=TINY_HEAD_SIZE,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )
    tiny_page = probe.page_size_bytes
    assert tiny_page < EXPECTED_KDA_STATE_PAGE_BYTES

    with pytest.raises(ValueError, match=REFUSAL_MESSAGE_FRAGMENT) as excinfo:
        _call_get_kv_cache_spec(_refusal_layers())

    # The message names both pages, so a log reader can see which direction was
    # refused without reading the source.
    message = str(excinfo.value)
    assert str(tiny_page) in message
    assert str(EXPECTED_KDA_STATE_PAGE_BYTES) in message
