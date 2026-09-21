# SPDX-License-Identifier: Apache-2.0
"""The model skeleton and its ``KVSpec``.

One ``LayerSpec`` per layer of the hybrid stack: the MLA half reports latent
geometry and the KDA half reports its two state shapes, from one uniform read.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from pathlib import Path

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

#: The stack depth.
DECLARED_TOTAL_ENTRIES = 45
#: The MLA head size.
DECLARED_MLA_ENTRIES = 11
#: The KDA head size.
DECLARED_KDA_ENTRIES = 34
#: The MLA head size is ``kv_lora_rank`` with a zero rotary slice.
DECLARED_MLA_HEAD_SIZE = 512
#: The count of entries with no dtype.
DECLARED_NONE_DTYPE_ENTRIES = 0

#: base 3 for the schedule, as ``test_config.py`` enumerates it from the
#: intake record. Repeated as a literal rather than imported so the two files
#: are independent bases and not one base read twice.
INTAKE_RECORDED_DSA_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

#: The layer split ``test_config.py`` pins, as ``(num_kda, num_dsa)``.
EXPECTED_LAYER_SPLIT = (34, 11)

#: KDA geometry from ``linear_attn_config`` (``glm5_next/config.py``).
KDA_HEAD_SIZE = 128
KDA_NUM_HEADS = 64

#: The pin's ``LayerSpec`` field set, in order (``model/kv_cache.py``).
ORIGINAL_LAYER_SPEC_FIELDS = (
    "name",
    "num_kv_heads",
    "head_size",
    "dtype",
    "sliding_window_size",
    "chunk_size",
)

#: The six fused KDA names a later reading retired, listed so this file's
#: KDA test can count survivals of them rather than assert a bare absence.
#: These are the only literal names the family-map change types on an expectation side; every
#: count below is derived from the map inside the test body.
RETIRED_KDA_FUSED_NAMES = (
    "in_proj_qkvz_weight",
    "in_proj_ba_weight",
    "out_proj_weight",
    "conv1d_weight",
    "norm_weight",
    "conv1d_bias",
)

#: The provisional DSA indexer name a later reading retired for ``wq_b_weight``.
RETIRED_DSA_INDEXER_NAME = "wq_weight"

#: The two layer-level names, read here only to isolate the mHC leaves
#: from the map's layer-level set. Their count is not a family-map reading.
LAYER_LEVEL_LAYERNORMS = (
    "input_layernorm_weight",
    "post_attention_layernorm_weight",
)


# ---------------------------------------------------------------------------
# The deferred import (see the module docstring) and the fixtures.
# ---------------------------------------------------------------------------


def _impl():
    """Import the implementation module inside a test body, never at import."""
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
    """The skeleton built from the 45-layer fixture."""
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
    """The complement of the MLA/DSA set."""
    return [
        layer for layer in spec.layers if layer.head_size != DECLARED_MLA_HEAD_SIZE
    ]


def _map_leaves(names: set[str], prefix: str) -> set[str]:
    """Leaf names in ``names`` sitting directly under ``prefix``, no deeper. """
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
    """The fixture declares 45 layers and the geometry this file counts on."""
    text = raw["text_config"]
    assert text["num_hidden_layers"] == DECLARED_TOTAL_ENTRIES
    assert len(text["layer_types"]) == DECLARED_TOTAL_ENTRIES
    assert text["kv_lora_rank"] == DECLARED_MLA_HEAD_SIZE
    assert text["qk_rope_head_dim"] == 0
    assert text["linear_attn_config"]["head_dim"] == KDA_HEAD_SIZE


def test_kv_spec_the_pin_dataclass_is_not_widened() -> None:
    """The six original fields survive as a prefix of the widened dataclass."""
    from dataclasses import fields

    names = tuple(f.name for f in fields(LayerSpec))
    assert names[:6] == ORIGINAL_LAYER_SPEC_FIELDS
    assert len(names) == 11
    assert names[6:] == (
        "kda_conv_state_shape",
        "kda_recurrent_state_shape",
        "kda_conv_state_dtype",
        "kda_recurrent_state_dtype",
        "latent_kv",
    )
    assert next(f.default for f in fields(LayerSpec) if f.name == "latent_kv") is False
    assert tuple(f.name for f in fields(KVSpec)) == ("layers",)

    # The pin's exact 6-argument positional form still constructs.
    probe = LayerSpec("layers.0.self_attn", 1, 512, torch.bfloat16, None, None)
    assert probe.sliding_window_size is None
    assert probe.chunk_size is None


# ---------------------------------------------------------------------------
# Exactly 45 LayerSpec entries.
# ---------------------------------------------------------------------------


def test_the_spec_has_exactly_forty_five_layer_entries(
    spec: KVSpec, model
) -> None:
    """The entry count equals the stack depth, on two independent bases."""
    assert isinstance(spec, KVSpec)
    assert len(spec.layers) == DECLARED_TOTAL_ENTRIES
    assert all(isinstance(layer, LayerSpec) for layer in spec.layers)

    # Base 2 -- the module tree's own layer count, not the declared literal.
    assert len(model.model.layers) == DECLARED_TOTAL_ENTRIES
    # Base 3 -- the config's declared depth.
    assert model.text_config.num_hidden_layers == DECLARED_TOTAL_ENTRIES


# ---------------------------------------------------------------------------
# 11 entries carry MLA/DSA geometry, head_size == 512.
# ---------------------------------------------------------------------------


def test_eleven_entries_carry_the_mla_latent_geometry(
    spec: KVSpec, model
) -> None:
    """The MLA entries, with the positions checked against three independent bases."""
    mla = _mla_entries(spec)
    assert len(mla) == DECLARED_MLA_ENTRIES

    positions = [
        index
        for index, layer in enumerate(spec.layers)
        if layer.head_size == DECLARED_MLA_HEAD_SIZE
    ]
    # Base 1 -- the fixture's own schedule, read through the config accessor.
    assert positions == model.text_config.dsa_layer_indices
    # Base 2 -- the indices enumerated in the intake record.
    assert positions == INTAKE_RECORDED_DSA_INDICES
    # Base 3 -- the 3:1 interleave rule, arithmetically.
    assert positions == [
        index for index in range(DECLARED_TOTAL_ENTRIES) if index % 4 == 3
    ]

    for layer in mla:
        assert layer.name.endswith(".self_attn")
        assert layer.head_size == DECLARED_MLA_HEAD_SIZE
        assert layer.num_kv_heads == 1, "the MLA latent is one replicated KV head"
        assert layer.chunk_size is None


def test_the_head_size_is_the_latent_width_not_a_literal(model) -> None:
    """512 is ``kv_lora_rank + qk_rope_head_dim``, and the rope slice is 0. """
    impl = _impl()
    text_config = model.text_config
    assert text_config.kv_lora_rank == DECLARED_MLA_HEAD_SIZE
    assert text_config.qk_rope_head_dim == 0
    assert text_config.mla_use_nope is True
    assert impl._resolve_mla_head_size(text_config) == DECLARED_MLA_HEAD_SIZE

    # Mutation arm: a config with a rotary slice must widen the latent.
    with_rope = Glm5NextTextConfig(kv_lora_rank=512, qk_rope_head_dim=64)
    assert impl._resolve_mla_head_size(with_rope) == 576


# ---------------------------------------------------------------------------
# 34 entries carry KDA geometry, by complement.
# ---------------------------------------------------------------------------


def test_thirty_four_entries_carry_the_kda_geometry(
    spec: KVSpec, model
) -> None:
    """The KDA entries by complement, and the complement is exhaustive."""
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
        assert layer.head_size == KDA_HEAD_SIZE
        assert layer.head_size != DECLARED_MLA_HEAD_SIZE
        assert layer.num_kv_heads == KDA_NUM_HEADS


def test_kv_spec_the_split_reproduces_the_config_pins(spec: KVSpec) -> None:
    """The pair ``(34, 11)`` that ``test_config.py`` already pins. """
    measured = (len(_kda_entries(spec)), len(_mla_entries(spec)))
    assert measured == EXPECTED_LAYER_SPLIT
    assert sum(measured) == DECLARED_TOTAL_ENTRIES


def test_kv_spec_the_counts_follow_the_schedule_and_not_a_constant(raw: dict) -> None:
    """Move one layer, and the three counts above must move."""
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
    """45 distinct names, each carrying its family's module path. """
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


# ---------------------------------------------------------------------------
# 0 entries have ``dtype is None``.
# ---------------------------------------------------------------------------


def test_no_entry_has_a_none_dtype(spec: KVSpec, model) -> None:
    """The counted zero, plus what every dtype actually is."""
    none_dtypes = [layer for layer in spec.layers if layer.dtype is None]
    assert len(none_dtypes) == DECLARED_NONE_DTYPE_ENTRIES

    for layer in spec.layers:
        assert isinstance(layer.dtype, torch.dtype)
        assert layer.dtype is model.text_config.torch_dtype
    assert model.text_config.torch_dtype is torch.bfloat16


def test_the_kda_state_dtype_resolver_has_a_live_none_branch() -> None:
    """That zero rests on a resolver with a real ``None`` branch."""
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
    """45 layers of 4096 hidden with 288 routed experts, allocating nothing. """
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

    # Positive control, same window.
    torch.nn.Parameter(torch.zeros(1))
    assert len(created) == 1


def test_kv_spec_declared_parameters_are_reserved_and_unmaterialised(model) -> None:
    """A declared name resolves to ``None`` and is skipped by torch's walks. """
    layer_zero = model.model.layers[0]
    assert layer_zero.input_layernorm_weight is None
    assert "input_layernorm_weight" in layer_zero._parameters

    assert list(model.named_parameters()) == []
    assert model.state_dict() == {}


#: The sentence every stub in ``model_fp8.py`` used to raise with. It is still the
#: marker this census scans for and it is exact rather than a pattern: while stubs
#: existed, each spelled it identically, so a count of it is a reading of how many
#: stubs the module still has. That count is now zero.
STUB_SENTENCE = "is a stub created by"


def _stub_markers(source: str) -> tuple[list[int], list[int]]:
    """The lines of ``source`` that raise ``NotImplementedError`` or carry the sentence as code."""
    tree = ast.parse(textwrap.dedent(source))
    raises = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(raised, ast.Name) and raised.id == "NotImplementedError":
            raises.append(node.lineno)
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    sentences = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and STUB_SENTENCE in node.value
    ]
    return sorted(raises), sorted(sentences)


def test_kv_spec_every_compute_site_is_a_stub(model) -> None:
    """Every compute site computes -- the census this node id names has inverted. """
    impl = _impl()
    walked = [
        model,
        model.model,
        model.model.layers[0].mlp,
        model.model.layers[3].self_attn,
        model.model.layers[3].mlp,
        model.model.layers[3].mlp.experts,
        model.model.layers[3].mlp.shared_experts,
        model.model.layers[0],
        model.model.layers[0].attention,
        model.model.layers[3],
        model.model.layers[3].self_attn.indexer,
    ]
    assert len(walked) == 11, f"the walk drifted to {len(walked)} modules"

    seven = (
        "Glm5NextRoutedExperts",
        "Glm5NextSharedExperts",
        "Glm5NextMoEBlock",
        "Glm5NextDenseMLP",
        "Glm5NextMLAAttention",
        "Glm5NextModel",
        "Glm5NextForConditionalGeneration",
    )
    reached = {type(module).__name__ for module in walked}
    missing = sorted(set(seven) - reached)
    assert not missing, (
        f"this fixture's tree instantiates none of {missing}, so the census cannot "
        f"read the forwards that were implemented"
    )

    # ---- the reading, in the same source-reading form: an implemented forward carries
    # no stub sentence and raises no ``NotImplementedError`` anywhere in its body.
    # Read off the source rather than by calling, for the reason already recorded:
    # this fixture's model is an unmaterialised skeleton whose parameters are all
    # ``None``, so a real call cannot reach the numerics, and a ``TypeError`` from a
    # missing argument is equally true of a working forward and of a stub whose
    # signature demands arguments.
    read: list[str] = []
    for module in walked:
        cls = type(module)
        name = f"{cls.__name__}.forward"
        # The instrument is pointed at the right function, checked before its answer
        # is used: an inherited or wrapped ``forward`` would have another qualified
        # name, and reading someone else's source would make every absence below a
        # statement about the wrong function.
        assert cls.forward.__qualname__ == name, (
            f"{cls.__name__}'s forward is defined as "
            f"{cls.forward.__qualname__}, so this census would read another "
            f"class's source"
        )
        raises, sentences = _stub_markers(inspect.getsource(cls.forward))
        assert not sentences, (
            f"{name} still carries the stub sentence as code, at line(s) "
            f"{sentences} of its own source, so a compute site that has "
            f"declared is still a stub"
        )
        assert not raises, (
            f"{name} still raises NotImplementedError, at line(s) {raises} of its "
            f"own source"
        )
        read.append(name)

    # ---- the population, so the walk and the module are counted against one
    # another rather than each on its own. While stubs existed this count had to
    # equal the number of arms; now it has to be zero, and a sentence left anywhere
    # in the module is a stub no arm above reaches.
    # Read as code, by the same predicate the per-forward readings use, and for the
    # same reason: over a whole module the chance of prose naming the sentence is
    # larger, not smaller, and this count is the one that speaks for every line the
    # walk above does not reach.
    _, module_sentence_lines = _stub_markers(inspect.getsource(impl))
    sentences = len(module_sentence_lines)
    assert sentences == 0, (
        f"model_fp8.py carries {sentences} stub sentences as code, at line(s) "
        f"{module_sentence_lines}, while this census asserts every compute site is "
        f"implemented; every sentence owes an arm, so a leftover is a stub nothing "
        f"above guards"
    )

    # ---- the positive control, Re-grounded on a real forward. An absence is not a
    # measurement until the instrument is shown to find the thing when it is there,
    # and this census used to ground that on a forward it asserted was a stub -- a
    # ground that disappeared with the last stub. So the control now takes a real
    # forward's own source, splices the sentence into a copy of it, and requires the
    # same scan to fire: same instrument, same text, one planted marker.
    probe = model.model.layers[3].mlp
    probe_source = inspect.getsource(type(probe).forward)
    assert len(probe_source.splitlines()) > 20, (
        f"{type(probe).__name__}.forward reads as "
        f"{len(probe_source.splitlines())} lines; a near-empty read would make "
        f"every absence above vacuous"
    )
    # The planted marker is the form a stub really had, not the sentence in a comment.
    # A comment was what this control used to plant, and the readings above no longer
    # see one -- so the old control would now arm nothing. What every stub in this
    # module actually was is a raise whose message carries the sentence, so that is
    # what a copy of a real forward gets, and both readings have to fire on it.
    assert _stub_markers(probe_source) == ([], []), (
        "the control's own forward already reads as a stub, so a planted marker "
        "would prove nothing"
    )
    planted = textwrap.dedent(probe_source) + (
        "    raise NotImplementedError(\n"
        f'        "{type(probe).__name__}.forward {STUB_SENTENCE} "\n'
        '        "a control, to arm the two readings above"\n'
        "    )\n"
    )
    planted_raises, planted_sentences = _stub_markers(planted)
    assert planted_raises and planted_sentences, (
        f"a forward carrying a real stub raise reads as raises={planted_raises} "
        f"sentences={planted_sentences}; a reading that cannot find the thing when "
        f"it IS there makes every absence above vacuous"
    )
    commented = textwrap.dedent(probe_source) + f"    # {STUB_SENTENCE} in a comment\n"
    assert _stub_markers(commented) == ([], []), (
        "a comment naming the stub sentence still reads as a stub, so the readings "
        "above can be failed by a file that explains itself"
    )


def test_kv_spec_declared_parameter_names_match_the_weight_map(model) -> None:
    """Exact set equality against ``build_weight_mappings``'s param-name side. """
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())

    assert declared == mapped, (
        f"skeleton-only names: {sorted(declared - mapped)}; "
        f"map-only names: {sorted(mapped - declared)}"
    )
    # No name is declared twice under the same path.
    assert len(model.declared_parameter_names()) == len(declared)


def test_kv_spec_the_map_comparison_is_proved_live(model) -> None:
    """mutation arm: the set equality above must be able to fail. """
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())
    assert declared, "the skeleton declared no parameter names at all"
    assert mapped, "the map produced no parameter names at all"

    dropped = sorted(declared)[0]
    assert (declared - {dropped}) != mapped


def test_kv_spec_the_quantised_flag_does_not_move_the_param_name_side(model) -> None:
    """amended: the flag adds exactly the 44 scale names. """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
        DSA_SCALED_PROJECTIONS,
        FP8_SCALE_SUFFIX,
    )

    quantised = set(build_weight_mappings(model.text_config, quantised=True))
    plain = set(build_weight_mappings(model.text_config, quantised=False))
    dsa_indices = _layer_indices(model, DSA_LAYER_TYPE)

    # (i) The flag only ever adds parameter names.
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

    # (iv) The second assertion.
    assert set(model.declared_parameter_names()) == quantised

    # (v) Each scaled leaf's weight parameter maps to the weight key and its
    #     scale companion -- the pair the loader chooser downscales -- while the
    #     scale key also has the parameter of its own that (ii) counted.
    mappings = build_weight_mappings(model.text_config, quantised=True)
    weight_names = [
        f"model.layers.{index}.self_attn.{leaf}_weight"
        for index in dsa_indices
        for leaf in DSA_SCALED_PROJECTIONS
    ]
    assert len(weight_names) == 44
    unpaired = [
        n
        for n in weight_names
        if not (
            isinstance(mappings[n], list)
            and len(mappings[n]) == 2
            and mappings[n][1] == mappings[f"{n[: -len('_weight')]}_{FP8_SCALE_SUFFIX}"]
        )
    ]
    assert not unpaired, (
        f"{len(unpaired)} scaled-leaf weight parameters do not carry the "
        f"[weight, scale] pair the chooser downscales: {sorted(unpaired)[:4]}"
    )


def test_kv_spec_the_tied_head_condition_mirrors_the_map(raw: dict) -> None:
    """``lm_head_weight`` is declared iff the map declares it. """
    tied = copy.deepcopy(raw)
    tied["tie_word_embeddings"] = True

    model = _impl().Glm5NextForConditionalGeneration.from_configs(tied)
    declared = set(model.declared_parameter_names())
    mapped = set(build_weight_mappings(model.text_config).keys())

    assert model.text_config.tie_word_embeddings is True
    assert "lm_head_weight" not in declared
    assert declared == mapped


def test_kv_spec_kda_attention_declares_the_maps_fifteen_leaf_names(model) -> None:
    """The KDA half's declared names are the map's, per module and per path. """
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

    # Survivals, scoped to the family's own module paths: ``norm_weight`` is
    # also the legitimate ``model.norm_weight``, so an unscoped scan would
    # false-fire on a name this block must leave alone.
    survivals = sorted(
        name
        for name in declared
        if name.rsplit(".", 1)[0].endswith(".self_attn")
        and name.rsplit(".", 1)[-1] in RETIRED_KDA_FUSED_NAMES
    )
    assert survivals == []


def test_kv_spec_dsa_indexer_declares_the_maps_seven_leaf_names(model) -> None:
    """The sparse indexer's declared names are the map's seven. """
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


def test_kv_spec_both_layer_classes_declare_the_maps_mhc_names_flat(model) -> None:
    """The mHC weights sit flat on the layer, on both layer classes. """
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

    mhc_leaves = first - set(LAYER_LEVEL_LAYERNORMS)
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

    # Both classes, so a single-family regression cannot hide behind the other.
    class_names = {type(layer).__name__ for layer in layers}
    assert len(class_names) == 2, sorted(class_names)
    for layer in layers:
        assert mhc_leaves <= set(layer.declared_param_names)


def test_kv_spec_mla_attention_names_are_unchanged_and_match_the_map(model) -> None:
    """A measured negative: the MLA names already agreed and still do. """
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


