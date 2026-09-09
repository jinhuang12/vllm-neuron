# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-030d``: the mHC four-stream composition, one commit at a time.

WHAT THIS FILE IS FOR. ``inc-glm53f-030`` landed the mHC layer and measured its
arithmetic; ``inc-glm53f-030c`` corrected that arithmetic against the checkpoint's
own model file. Neither of them wired the layer into the decoder: the class was
bound to no layer at all. ``-030d`` does the wiring, in four commits, and this
file grows with the first three of them.

TWO COMMITS ARE IN THIS FILE SO FAR, and the items are grouped in that order.

**Commit 1, THE BIND.** The six mHC tensors the checkpoint carries per layer --
``hc_attn_{base,fn,scale}`` and ``hc_ffn_{base,fn,scale}``, 270 keys over layers
0-44 -- reach two :class:`Glm5NextHyperConnection` instances per layer ONCE, after
the load, on the same load-time-prep walk this tree already runs for its projection
and scale preps. Those items run no forward.

**Commit 2, ROUTE R3.** Each layer forward gains ONE optional ``streams``
keyword: absent is the one-stream residual add the method already did, present runs
the four-stream mHC pair around the same attention half. The branch REFUSES BOTH
WAYS, which is arm (4) of this block's acceptance -- two named refusals -- because
an optional keyword with a default is exactly how a silent wrong route gets in.
Arm (5), the three landed layer suites re-run unedited with zero edits to their
expected values, is a RUN rather than an item here.

The feed-forward site and the carrier are commit 3's, and the numeric comparison
against the reference at ``T = 128`` belongs to that fixture.

WHY THE TWO INSTANCES ARE NOT SUBMODULES, which is the one design decision this
commit had to make. Registering them would put their three parameters each into
``named_parameters()``, and two landed readings count that list exactly: it reads
``0`` on a declared-but-unloaded tree and the declared count after materialisation
(``test_load_weights.py:960-980``, with the pre-load zero read again at
``test_kv_spec.py:621``). So the sites are held in a plain dict, and
:func:`test_030d_the_bind_adds_no_name_to_named_parameters_or_the_state_dict`
measures that with a firing control -- a registered site DOES add three names, so
the counted zero is a reading rather than a tautology.

NO TOLERANCE AND NO COMPARATOR IS REGISTERED HERE. Every item so far is a
structural, bookkeeping or counted-route reading, and the one numeric comparison --
the one-stream route against the same three steps -- is BITWISE, so it needs no
tolerance at all. The registered pair stays ``test_mhc_layer.py``'s
``(1e-2, 1e-5)``, cited when commit 3's reference comparison needs it (P9).

THE IMPLEMENTATION MODULE IS IMPORTED INSIDE TEST BODIES, never at module scope,
for the reason ``test_mhc_layer.py:102-107`` records: ``test_factory.py``'s C03
asserts ``model_fp8`` is absent from ``sys.modules``, and pytest imports every
collected module before running any test.

Command (Tier N harness, plan rev 288)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/model/glm5_next/test_mhc_composition_030d.py \\
      -s -rA -p no:randomly -p no:cacheprovider
"""

from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SENT = "MHC030D"


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config(**overrides):
    """A two-layer hybrid config: one layer of each attention family, all dense.

    ``dataclasses.replace`` on the checkpoint's own config, the idiom
    ``test_dsa_layer.py:1013-1035`` records, so every field this file does not name
    keeps the checkpoint's value and a config drift reaches this file rather than
    being overwritten by it.

    NO WIDTH IS NARROWED, and that is deliberate. This tree DECLARES its
    parameters and allocates none until the load
    (``model_fp8.py:_declare_parameters``), so a layer at the checkpoint's own
    widths costs nothing to build -- which is why ``test_kda_layer.py:382`` builds
    its stack from an unmodified ``Glm5NextTextConfig()`` too. The only tensors
    this file allocates are the six it loads itself, and at these widths the
    largest is ``fn`` at ``[24, 16384]`` float32, about 1.5 MB.

    THREE FIELDS ARE NARROWED, none of them a width: the stack is two layers long,
    it holds one layer of each attention family so a reading over "both families"
    has both in it, and both layers are dense so no expert bank is built.
    """
    from vllm_neuron.model.glm5_next.config import (
        DSA_LAYER_TYPE,
        KDA_LAYER_TYPE,
        Glm5NextTextConfig,
    )

    fields: dict[str, object] = dict(
        num_hidden_layers=2,
        layer_types=[KDA_LAYER_TYPE, DSA_LAYER_TYPE],
        first_k_dense_replace=2,
    )
    fields.update(overrides)
    return dataclasses.replace(Glm5NextTextConfig(), **fields)


def _leaf_shapes(text_config) -> dict[str, tuple[int, ...]]:
    """The shape each of the six leaves has in the checkpoint, DERIVED.

    ``fn`` is ``[mix, hc_mult * hidden]``, ``base`` is ``[mix]`` and ``scale`` is
    ``[3]``, with ``mix = (2 + hc_mult) * hc_mult`` -- the target model's own
    closed forms (``design/reference/modeling_glm5_next.py:258-265``), which
    :class:`Glm5NextHyperConnection` restates as ``hc_mult3``. Derived from the
    config so a config change reaches this file.
    """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)
    mix = (2 + hc_mult) * hc_mult
    by_role = {"fn": (mix, hc_mult * hidden), "base": (mix,), "scale": (3,)}
    return {leaf: by_role[leaf.split("_")[2]] for leaf in MHC_LEAVES}


def _load_the_six(layer, text_config, *, seed: int = 30, skip: tuple[str, ...] = ()):
    """Put a real tensor on each of the six leaves, the way a load leaves them.

    ``setattr`` of an ``nn.Parameter`` over the declared ``None``, which is this
    suite's own idiom for a loaded leaf (``test_kda_layer.py:409-411``). Returns
    ``{leaf: tensor}`` for what was set, so an item can compare storage rather
    than values.
    """
    gen = torch.Generator().manual_seed(seed)
    placed: dict[str, torch.Tensor] = {}
    for leaf, shape in sorted(_leaf_shapes(text_config).items()):
        if leaf in skip:
            continue
        tensor = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
        setattr(layer, leaf, nn.Parameter(tensor, requires_grad=False))
        placed[leaf] = tensor
    return placed


def _kda_layer(text_config, layer_idx: int = 0):
    return _impl().Glm5NextKDALayer(text_config, layer_idx, 1)


def _dsa_layer(text_config, layer_idx: int = 1):
    return _impl().Glm5NextDSALayer(text_config, layer_idx, 1)


def _layer_of(family: str, text_config):
    return _kda_layer(text_config) if family == "kda" else _dsa_layer(text_config)


class _StubAttention(nn.Module):
    """Stands in for the attention half, and RECORDS what it was handed.

    The composition around the sublayer is what commit 2 wires, and a real
    attention half would make every reading here depend on two other increments'
    kernels. So the stub is deterministic and nonlinear -- ``tanh(x) * 1.5``, the
    same stand-in ``test_mhc_layer.py:345`` uses -- and it keeps the shape of every
    input it saw, which is how the items below tell the two routes apart.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[int, ...]] = []

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        self.seen.append(tuple(hidden_states.shape))
        return torch.tanh(hidden_states) * 1.5


def _install_stub(layer) -> _StubAttention:
    """Replace the attention half, under the attribute the class itself names."""
    stub = _StubAttention()
    setattr(layer, type(layer).ATTENTION_ATTR, stub)
    return stub


def _norm_ready(layer, text_config):
    """Load the one weight ``_input_norm`` reads, so the plain route can run."""
    setattr(
        layer,
        "input_layernorm_weight",
        nn.Parameter(
            torch.ones(int(text_config.hidden_size), dtype=torch.float32),
            requires_grad=False,
        ),
    )
    return layer


def _call_kwargs(family: str) -> dict[str, object]:
    """The carriers each family's forward requires, as values the stub ignores.

    Both forwards cast some of these (``int(start_position)``,
    ``float(softmax_scale)``), so the types here are the types those casts accept.
    Nothing below reads them: the attention half is the stub.
    """
    if family == "kda":
        return {
            "conv_state": torch.zeros(1, dtype=torch.float32),
            "recurrent_state": torch.zeros(1, dtype=torch.float32),
            "is_prefill": True,
        }
    return {
        "latent_cache": torch.zeros(1, dtype=torch.float32),
        "pool_cache": torch.zeros(1, dtype=torch.float32),
        "seq_lens": torch.ones(1, dtype=torch.int32),
        "start_position": 0,
        "softmax_scale": 1.0,
        "max_seq_len": 8,
        "page_size": 4,
    }


def _model_source() -> tuple[str, ast.Module]:
    """``model_fp8``'s own bytes and their parse tree, read off disk."""
    import vllm_neuron.model.glm5_next.model_fp8 as module

    text = Path(module.__file__).read_text()
    return text, ast.parse(text)


def _function_defs(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def _reference_path() -> str | None:
    """The campaign's reference copy, if this machine has it. Never required.

    The same resolver ``test_mhc_corrections_030c.py:231-253`` uses, so the two
    files cannot disagree about where the reference lives.
    """
    env = os.environ.get("GLM53F_REFERENCE_FILE")
    if env and os.path.exists(env):
        return env
    tail = os.path.join(
        "artifacts/campaigns/glm-5.3-flash-port/design/reference",
        "modeling_glm5_next.py",
    )
    here = os.path.abspath(__file__)
    for _ in range(12):
        here = os.path.dirname(here)
        for cand in (
            os.path.join(here, tail),
            os.path.join(here, "NeuronAgenticDevelopment", tail),
        ):
            if os.path.exists(cand):
                return cand
        if here == os.path.dirname(here):
            break
    return None


# --------------------------------------------------------------------------- #
# (1) The six leaves group by site off the map's own tuple.                    #
# --------------------------------------------------------------------------- #
def test_030d_the_six_leaves_group_by_site_off_the_maps_own_tuple() -> None:
    """The grouping is derived from ``MHC_LEAVES``, and a strange leaf raises.

    The six names live once, in ``weight_loaders_fp8.MHC_LEAVES``, and the bind
    reads the site and the role off the name shape ``hc_<site>_<role>``. This item
    reads the grouping and then plants a leaf the rule cannot resolve, so the
    refusal is measured rather than assumed.
    """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    impl = _impl()
    grouped = impl._mhc_leaves_by_site()
    flat = sorted(leaf for roles in grouped.values() for leaf in roles.values())

    say("group-sites", f"sites={sorted(grouped)}", f"leaves={len(flat)}")
    for site in sorted(grouped):
        say("group-site", site, sorted(grouped[site].items()))

    assert sorted(grouped) == ["attn", "ffn"], (
        f"the map's mHC leaves group into {sorted(grouped)}; the composition has "
        f"exactly two sites, the attention one and the feed-forward one"
    )
    assert flat == sorted(MHC_LEAVES), (
        f"the grouping covers {flat} and the map declares {sorted(MHC_LEAVES)}; a "
        f"leaf the grouping drops would leave a site holding its constructor's zeros"
    )
    for site, roles in grouped.items():
        assert sorted(roles) == ["base", "fn", "scale"], (
            f"site {site} resolved roles {sorted(roles)}; the class takes exactly "
            f"three, and {sorted(impl.MHC_ROLE_PARAMETERS)} is what it takes"
        )

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        impl._mhc_leaves_by_site(("hc_attn_gain",))
    say("group-control-refusal", str(raised.value)[:110])
    assert "hc_attn_gain" in str(raised.value), (
        f"the refusal does not name the leaf it could not resolve: {raised.value}"
    )


# --------------------------------------------------------------------------- #
# (2) A loaded layer binds both sites, and the tensors are the loaded ones.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_030d_a_loaded_layer_binds_both_sites_to_the_loaded_tensors(
    family: str,
) -> None:
    """Both families, both sites, and STORAGE identity rather than value equality.

    A value comparison would pass against a copy, and a copy is the defect this
    reading exists to exclude: a copied weight would go stale the moment anything
    wrote to the layer's own parameter. So each bound parameter's ``data_ptr`` is
    compared with the loaded tensor's, which is an identity of storage.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config) if family == "kda" else _dsa_layer(text_config)
    placed = _load_the_six(layer, text_config)

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)

    say(f"bind-{family}", f"returned={bound_sites}", f"sites={sorted(sites)}")
    say(f"bind-{family}-health", f"bound_sites={health['bound_sites']}",
        f"binds={health['binds']}", f"device={health['device']}")

    assert bound_sites == 2, (
        f"the bind reported {bound_sites} sites on a layer whose six leaves are "
        f"all loaded; the composition has an attention site and an FFN site"
    )
    assert sorted(sites) == ["attn", "ffn"]
    assert health["bound_sites"] == 2 and health["binds"] == 1

    grouped = impl._mhc_leaves_by_site()
    for site, roles in sorted(grouped.items()):
        instance = sites[site]
        assert isinstance(instance, impl.Glm5NextHyperConnection)
        for role, leaf in sorted(roles.items()):
            parameter = getattr(instance, impl.MHC_ROLE_PARAMETERS[role])
            loaded = getattr(layer, leaf)
            same_storage = parameter.data_ptr() == loaded.data_ptr()
            say(
                f"bind-{family}-leaf",
                leaf,
                f"parameter={impl.MHC_ROLE_PARAMETERS[role]}",
                f"shape={tuple(parameter.shape)}",
                f"same_storage={int(same_storage)}",
            )
            assert same_storage, (
                f"{leaf} reached site {site} as a COPY, not as the loaded tensor: "
                f"the site holds storage {parameter.data_ptr()} and the layer's "
                f"parameter holds {loaded.data_ptr()}. A copy goes stale silently"
            )
            assert tuple(parameter.shape) == tuple(placed[leaf].shape)
            assert health["sites"][site][leaf]["data_ptr"] == loaded.data_ptr(), (
                f"the record's pointer for {leaf} is not the loaded tensor's, so a "
                f"later staleness check would compare the wrong two numbers"
            )


# --------------------------------------------------------------------------- #
# (3) A layer carrying none of the six is skipped, and the skip is recorded.   #
# --------------------------------------------------------------------------- #
def test_030d_a_layer_carrying_none_of_the_six_is_skipped_and_recorded() -> None:
    """The draft head's case: all six declared, none loaded, nothing bound.

    Real rather than defensive. The map emits the six leaves for the layers in
    ``layer_types`` only (``weight_loaders_fp8.py:385-401`` calling ``_add_mhc``)
    and the checkpoint carries them on layers 0-44 and on no other -- 270 keys, 45
    layers, six each, read off the fixture index. A block built from the same layer
    class for the draft head therefore declares all six and is loaded none.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _dsa_layer(text_config)

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)

    say("skip", f"returned={bound_sites}", f"sites={len(sites)}")
    say("skip-record", f"unloaded={health['unloaded_leaves']}")

    assert bound_sites == 0, (
        f"the bind reported {bound_sites} sites on a layer that carries none of "
        f"the six tensors; there is nothing to bind and nothing to guess"
    )
    assert sites == {}
    assert health["bound_sites"] == 0
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    assert health["unloaded_leaves"] == sorted(MHC_LEAVES), (
        f"the record names {health['unloaded_leaves']} as unloaded and the map "
        f"declares {sorted(MHC_LEAVES)}; a skip that named a subset would read "
        f"like a partial load"
    )
    assert "carried none" in str(health["skipped_because"])


# --------------------------------------------------------------------------- #
# (4) A partially loaded layer refuses BY NAME, with its firing control.       #
# --------------------------------------------------------------------------- #
def test_030d_a_partially_loaded_layer_refuses_by_name() -> None:
    """Five of six loaded is the defect case, and the refusal names the missing one.

    THE CONTROL IS THE SAME LAYER WITH SIX. Without it the refusal could be an
    artefact of the fixture rather than of the missing leaf, so the item binds a
    fully loaded layer of the same family first and reads its two sites.
    """
    impl = _impl()
    text_config = _text_config()

    control = _kda_layer(text_config)
    _load_the_six(control, text_config)
    control_sites = control.bind_hyper_connection_sites(
        text_config, torch.device("cpu")
    )
    say("partial-control", f"sites={control_sites}")
    assert control_sites == 2, "the control fixture does not bind, so it controls nothing"

    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config, skip=("hc_ffn_scale",))
    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))

    message = str(raised.value)
    say("partial-refusal", message[:160])
    assert "hc_ffn_scale" in message, (
        f"the refusal does not name the leaf that is missing: {message}"
    )
    assert not hasattr(layer, impl.MHC_SITES_ATTR), (
        "the refusal left a sites attribute behind, so a caller cannot tell a "
        "refused bind from a completed one"
    )


# --------------------------------------------------------------------------- #
# (5) An unmaterialised placeholder refuses BY NAME.                          #
# --------------------------------------------------------------------------- #
def test_030d_an_unmaterialised_placeholder_refuses_by_name() -> None:
    """A registered placeholder is not a loaded tensor, and the bind says so.

    ``_materialise_declared_parameters`` registers ``UninitializedParameter``
    placeholders BEFORE the reader fills them, and torch reports such a parameter
    through ``named_parameters()`` while it still has no shape. So "not ``None``"
    is not enough: the bind asks ``is_lazy`` as well, on the same ground
    ``_require_prep_operands_on_device`` states for the three preps.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)
    layer.register_parameter(
        "hc_attn_fn",
        nn.parameter.UninitializedParameter(dtype=torch.float32, requires_grad=False),
    )

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))

    message = str(raised.value)
    say("lazy-refusal", message[:160])
    assert "hc_attn_fn" in message and "placeholder" in message, (
        f"the refusal does not name the placeholder it found: {message}"
    )


# --------------------------------------------------------------------------- #
# (6) A leaf that is not on the load's device refuses BY NAME.                 #
# --------------------------------------------------------------------------- #
def test_030d_a_leaf_off_the_loads_device_refuses_by_name() -> None:
    """The exposure that makes this check necessary is measured, not asserted.

    A plain dict is not visited by ``nn.Module._apply``
    (``probe-091-device-binding.out``, ``PLAIN_DICT_IS_LEFT_BEHIND=True``), so a
    site bound from the wrong device would stay there permanently while every
    later ``.to(device)`` reported success. The refusal is read on a device this
    machine can always name without allocating on it, and the CPU control shows
    the same fixture binds when the two agree.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.bind_hyper_connection_sites(text_config, torch.device("meta"))
    message = str(raised.value)
    say("device-refusal", message[:170])
    assert "cpu" in message and "meta" in message, (
        f"the refusal does not name both places: {message}"
    )

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    say("device-control", f"sites={bound_sites}")
    assert bound_sites == 2, (
        "the same layer does not bind on the device its tensors are on, so the "
        "refusal above says nothing about the device check"
    )


# --------------------------------------------------------------------------- #
# (7) The bind adds no name to named_parameters() or the state dict.           #
# --------------------------------------------------------------------------- #
def test_030d_the_bind_adds_no_name_to_named_parameters_or_the_state_dict() -> None:
    """The counted zero, with the control that makes it mean something.

    Two landed readings count ``named_parameters()`` exactly -- ``0`` on a
    declared-but-unloaded tree and the declared count after materialisation
    (``test_load_weights.py:960-980``) -- and the loader's map equality reads the
    parameter-name SET (``test_load_weights.py:2700``). A submodule holding three
    parameters would move all three readings.

    THE FIRING CONTROL registers one bound site as a submodule of a throwaway
    module and counts what that adds: three names. Without it, "the count did not
    move" would be satisfied by a reading that cannot see a move at all.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)

    before_params = {name for name, _ in layer.named_parameters()}
    before_state = set(layer.state_dict())
    before_modules = {name for name, _ in layer.named_modules()}

    layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))

    after_params = {name for name, _ in layer.named_parameters()}
    after_state = set(layer.state_dict())
    after_modules = {name for name, _ in layer.named_modules()}

    say("invisible", f"parameters={len(before_params)}->{len(after_params)}")
    say("invisible", f"state_dict={len(before_state)}->{len(after_state)}")
    say("invisible", f"modules={len(before_modules)}->{len(after_modules)}")

    assert after_params == before_params, (
        f"the bind added {sorted(after_params - before_params)} to "
        f"named_parameters(); the landed load readings count that list exactly"
    )
    assert after_state == before_state, (
        f"the bind added {sorted(after_state - before_state)} to the state dict, "
        f"which is the map equality's own key set"
    )
    assert after_modules == before_modules, (
        f"the bind registered {sorted(after_modules - before_modules)} as a "
        f"submodule; the two sites are held in a plain dict on purpose"
    )

    site = getattr(layer, impl.MHC_SITES_ATTR)["attn"]
    probe = nn.Module()
    probe.site = site
    added = {name for name, _ in probe.named_parameters()}
    say("invisible-control", f"a_registered_site_adds={len(added)}", sorted(added))
    assert len(added) == 3, (
        f"registering a site added {len(added)} parameter names, not the three the "
        f"class holds; the invisibility reading above measures nothing if a "
        f"registered site is invisible too"
    )


# --------------------------------------------------------------------------- #
# (8) One rule, two families, one production call site.                       #
# --------------------------------------------------------------------------- #
def test_030d_both_families_delegate_to_one_body() -> None:
    """The rule lives once; each family holds a signature and a return.

    Read from the module's own bytes with ``ast``, so a second copy of the rule
    cannot hide behind a docstring that says there is only one.
    """
    _impl()
    _text, tree = _model_source()

    methods = _function_defs(tree, "bind_hyper_connection_sites")
    bodies = _function_defs(tree, "_bind_hyper_connection_sites")
    say("one-body", f"methods={len(methods)}", f"module_functions={len(bodies)}")

    assert len(bodies) == 1, (
        f"{len(bodies)} definitions of the bind rule; two copies of a rule drift"
    )
    assert len(methods) == 2, (
        f"{len(methods)} layer methods declare the bind; the stack has two "
        f"attention families and both carry the six leaves"
    )
    for method in methods:
        statements = [
            node for node in method.body if not isinstance(node, ast.Expr)
        ]
        assert len(statements) == 1 and isinstance(statements[0], ast.Return), (
            f"the method at line {method.lineno} does more than delegate; the rule "
            f"belongs in one body"
        )
        call = statements[0].value
        assert isinstance(call, ast.Call) and call.func.id == "_bind_hyper_connection_sites"

    impl = _impl()
    text_config = _text_config()
    for builder in (_kda_layer, _dsa_layer):
        layer = builder(text_config)
        assert hasattr(type(layer), "bind_hyper_connection_sites"), (
            f"{type(layer).__name__} does not answer the walk's gate, so its six "
            f"loaded tensors would never reach a site"
        )
        say("one-body-gate", type(layer).__name__, "answers")
    assert impl.MHC_SITES_ATTR != impl.MHC_BIND_HEALTH_ATTR


def test_030d_the_load_path_is_the_single_production_call_site() -> None:
    """Exactly one call in the package, and it sits inside the prep walk.

    THE COUNTING RULE IS THE PARSER, on ``test_load_weights.py:1112-1155``'s
    precedent: a hit counts only when it is an :class:`ast.Call` whose callee
    carries the name, so a docstring or a comment cannot be counted as a call.
    Every excluded mention is printed, so the rule is read rather than trusted.
    """
    import vllm_neuron

    _impl()
    package = Path(vllm_neuron.__file__).parent
    calls: list[str] = []
    excluded: list[str] = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text()
        if "bind_hyper_connection_sites" not in text:
            continue
        lines: set[int] = set()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            named = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else callee.id
                if isinstance(callee, ast.Name)
                else None
            )
            if named == "bind_hyper_connection_sites":
                lines.add(node.lineno)
                calls.append(f"{path.name}:{node.lineno}")
        for number, line in enumerate(text.split("\n"), start=1):
            if "bind_hyper_connection_sites" in line and number not in lines:
                excluded.append(f"{path.name}:{number}: {line.strip()[:90]}")

    say("call-site", f"calls={len(calls)}", calls)
    for line in excluded:
        say("call-site-excluded", line)

    assert len(calls) == 1, (
        f"the package holds {len(calls)} calls of the bind method {calls}; one "
        f"production caller is the whole point of putting it on the load walk"
    )

    _text, tree = _model_source()
    walk = _function_defs(tree, "_run_load_time_preps")
    assert len(walk) == 1
    line = int(calls[0].split(":")[1])
    say("call-site-home", f"line={line}", f"walk={walk[0].lineno}-{walk[0].end_lineno}")
    assert walk[0].lineno < line <= int(walk[0].end_lineno), (
        f"the one call sits at line {line}, outside "
        f"_run_load_time_preps ({walk[0].lineno}-{walk[0].end_lineno}); the bind "
        f"must run where the weights are already on the device"
    )


# --------------------------------------------------------------------------- #
# (9) The whole stack answers the gate, and the pre-load zero still holds.     #
# --------------------------------------------------------------------------- #
def test_030d_every_decoder_layer_answers_the_gate_and_the_tree_stays_empty() -> None:
    """A hybrid stack: every layer is a bind candidate and nothing is allocated.

    The count comes off the tree rather than from a number written here, so it
    follows the config instead of pinning it. The pre-load zero is the landed
    reading this commit is not allowed to move (``test_kv_spec.py:621``,
    ``test_load_weights.py:960-971``), so it is read again here on a hybrid stack.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    impl = _impl()
    text_config = _text_config()
    model = impl.Glm5NextModel(Glm5NextConfig(text_config=text_config), 1)

    candidates = [
        path
        for path, module in model.named_modules()
        if hasattr(type(module), "bind_hyper_connection_sites")
    ]
    families = sorted({type(layer).__name__ for layer in model.layers})
    live = list(model.named_parameters())

    say("stack", f"layers={len(model.layers)}", f"candidates={len(candidates)}")
    say("stack", f"families={families}", f"named_parameters={len(live)}")

    assert len(candidates) == len(model.layers), (
        f"{len(candidates)} of {len(model.layers)} layers answer the gate; the map "
        f"emits the six leaves for every layer of the stack, so every layer is a "
        f"candidate"
    )
    assert len(families) == 2, (
        f"this stack holds only {families}; the reading above says nothing about "
        f"both attention families unless both are in it"
    )
    assert live == [], (
        f"the tree reports {len(live)} parameters before any load; this commit is "
        f"not allowed to move that counted zero"
    )


# --------------------------------------------------------------------------- #
# (10) The framework overrides reach both sites, and the default is the target's.
# --------------------------------------------------------------------------- #
def test_030d_the_framework_overrides_reach_both_sites() -> None:
    """``mhc_sinkhorn_iters`` and ``mhc_eps`` travel through ``neuron_config``.

    The overrides are the framework's, read by the mHC class itself, and they
    reach it because the bind passes the config's own ``neuron_config``. The
    control is the same fixture with no overrides, which must read the
    CHECKPOINT's two dials -- otherwise this item would pass against a bind that
    ignored the config entirely.
    """
    from vllm_neuron.model.neuron_config import NeuronConfig

    impl = _impl()

    plain = _text_config()
    plain_layer = _kda_layer(plain)
    _load_the_six(plain_layer, plain)
    plain_layer.bind_hyper_connection_sites(plain, torch.device("cpu"))
    plain_sites = getattr(plain_layer, impl.MHC_SITES_ATTR)
    for site in sorted(plain_sites):
        instance = plain_sites[site]
        say(
            "override-control",
            site,
            f"iters={instance.sinkhorn_iters}",
            f"hc_eps={instance.hc_eps}",
        )
        assert instance.sinkhorn_iters == int(plain.hc_sinkhorn_iters)
        assert instance.hc_eps == float(plain.hc_eps)

    overridden = _text_config(
        neuron_config=NeuronConfig(mhc_sinkhorn_iters=3, mhc_eps=1e-3)
    )
    layer = _kda_layer(overridden)
    _load_the_six(layer, overridden)
    layer.bind_hyper_connection_sites(overridden, torch.device("cpu"))
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    for site in sorted(sites):
        instance = sites[site]
        say(
            "override",
            site,
            f"iters={instance.sinkhorn_iters}",
            f"hc_eps={instance.hc_eps}",
        )
        assert instance.sinkhorn_iters == 3, (
            f"site {site} runs {instance.sinkhorn_iters} Sinkhorn iterations where "
            f"the framework asked for 3, so the override did not reach it"
        )
        assert instance.hc_eps == pytest.approx(1e-3)
    assert int(overridden.hc_sinkhorn_iters) != 3, (
        "the override equals the checkpoint's own value, so this item cannot tell "
        "an override that arrived from one that was ignored"
    )


def test_030d_the_post_gate_multiplier_is_the_targets_own_two() -> None:
    """Every bound site takes the target's factor, and the citation is READ.

    ``inc-glm53f-030c`` moved this class's default from ``1.0`` to ``2.0`` because
    the target computes ``post = 2 * torch.sigmoid(...)``. The bind must not pass a
    number of its own, so the reading is the bound instance's value; where the
    campaign's reference copy is reachable, the cited line is read off it rather
    than quoted from a comment.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)
    layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)

    for site, instance in sorted(getattr(layer, impl.MHC_SITES_ATTR).items()):
        say("post-mult", site, f"value={instance.post_mult_value}")
        assert instance.post_mult_value == 2.0, (
            f"site {site} carries a post multiplier of {instance.post_mult_value}; "
            f"the target's factor is 2 and the bind chooses no number"
        )
    assert health["post_mult_value"] == 2.0

    path = _reference_path()
    if path is None:
        say("post-mult-citation", "reference_file_unreachable=1")
        return
    line = Path(path).read_text().split("\n")[283]
    say("post-mult-citation", f"reference:284={line.strip()}")
    assert "2 * torch.sigmoid" in line, (
        f"reference line 284 reads {line.strip()!r}, which is not the post gate "
        f"this default cites; the citation and the file have drifted"
    )


# --------------------------------------------------------------------------- #
# (11) A second bind is counted rather than refused.                          #
# --------------------------------------------------------------------------- #
def test_030d_a_second_bind_is_counted_rather_than_refused() -> None:
    """Re-binding must work, because a landed item calls the prep walk twice.

    ``test_load_weights.py:2967`` runs a completing load and then calls
    ``_run_load_time_preps`` again to read its returned pair. A bind that refused
    the second visit would redden that item, so the second bind is allowed and
    COUNTED: the record's ``binds`` says how many times it ran, which is what a
    reader needs to tell one bind from two.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)

    first = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    first_health = dict(getattr(layer, impl.MHC_BIND_HEALTH_ATTR))
    second = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    second_health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)

    say("rebind", f"first={first}", f"binds={first_health['binds']}")
    say("rebind", f"second={second}", f"binds={second_health['binds']}")

    assert (first, second) == (2, 2)
    assert first_health["binds"] == 1 and second_health["binds"] == 2, (
        f"the record counted {second_health['binds']} binds after two calls; a "
        f"count that does not move cannot distinguish one bind from two"
    )
    for site, instance in sorted(getattr(layer, impl.MHC_SITES_ATTR).items()):
        loaded = getattr(layer, impl._mhc_leaves_by_site()[site]["fn"])
        assert instance.fn.data_ptr() == loaded.data_ptr(), (
            f"after the second bind site {site} points somewhere other than the "
            f"layer's own tensor"
        )


# --------------------------------------------------------------------------- #
# COMMIT 2 -- route R3: one optional keyword, and a branch that refuses both   #
# ways. Arm (4) of the block's acceptance is these two named refusals; arm (5)  #
# is the three landed layer suites re-run unedited, which is a run and not an   #
# item here.                                                                   #
# --------------------------------------------------------------------------- #
TOKENS = 8


def _streams(text_config, *, tokens: int = TOKENS, seed: int = 11) -> torch.Tensor:
    """``[T, S, H]`` residual streams, small enough to stay out of sigmoid saturation."""
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            tokens,
            int(text_config.hc_mult),
            int(text_config.hidden_size),
            generator=gen,
            dtype=torch.float32,
        )
        * 0.05
    )


def _tokens(text_config, *, tokens: int = TOKENS, seed: int = 12) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            tokens, int(text_config.hidden_size), generator=gen, dtype=torch.float32
        )
        * 0.05
    )


def test_030d_the_site_names_are_the_leaves_own_middle_words() -> None:
    """The two site constants are the derived keys, not two strings typed twice.

    ``MHC_ATTENTION_SITE`` and ``MHC_FFN_SITE`` are what the layer forwards index
    the bound sites with, and :func:`_mhc_leaves_by_site` derives the same two keys
    off ``MHC_LEAVES``. If those ever disagreed the forward would raise a
    ``KeyError`` on a served token, so the two are read against each other here.
    """
    impl = _impl()
    derived = sorted(impl._mhc_leaves_by_site())
    named = sorted({impl.MHC_ATTENTION_SITE, impl.MHC_FFN_SITE})
    say("site-names", f"derived={derived}", f"named={named}")
    assert derived == named, (
        f"the leaves group into {derived} and the forwards index {named}; a "
        f"mismatch is a KeyError on a served token"
    )
    assert impl.MHC_ATTENTION_SITE != impl.MHC_FFN_SITE


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_030d_the_one_stream_route_computes_what_it_computed_before(
    family: str,
) -> None:
    """No streams on a layer with no mHC weight: pre-norm, attend, plain add.

    The want is computed here from the layer's OWN norm and the same stub, so the
    reading is that the route still composes those three steps in that order --
    not that it agrees with a number written down. Bitwise equality, because both
    sides run the identical operations on the identical operands; a tolerance here
    would hide a reordering.

    This is the route the draft head takes (``mtp.py:158``) and the route every
    landed direct caller takes, which is why it must survive the keyword's arrival
    untouched.
    """
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    stub = _install_stub(layer)
    hidden = _tokens(text_config)

    got = layer.forward(hidden, **_call_kwargs(family))
    # The want runs the SAME stub on the SAME norm, so this item cannot pass by
    # agreeing with a formula copied out of the stub. That second entry is why
    # ``stub.seen`` reads two shapes below.
    want = hidden + stub(layer._input_norm(hidden))

    say(f"plain-{family}", f"in={tuple(hidden.shape)}", f"out={tuple(got.shape)}",
        f"stub_saw={stub.seen}")
    assert tuple(got.shape) == tuple(hidden.shape)
    assert got.dtype == hidden.dtype
    assert torch.equal(got, want), (
        "the one-stream route no longer computes pre-norm, attend, plain add on "
        f"the operands it is given; max deviation {float((got - want).abs().max())}"
    )
    assert stub.seen == [tuple(hidden.shape), tuple(hidden.shape)], (
        f"the attention half saw {stub.seen}; it must be entered once by the route "
        f"and once by the want above, each on the [T, H] tokens"
    )


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_030d_a_layer_carrying_mhc_weights_refuses_a_one_stream_call(
    family: str,
) -> None:
    """Refusal 1 of the two the block declares, named.

    A layer that carries the six loaded tensors and is called without streams
    would run the residual add this block exists to replace, and nothing
    downstream could tell -- the shapes agree and the numbers are plausible. So it
    is refused. THE CONTROL IS THE SAME LAYER WITH NO mHC WEIGHT, which the item
    above reads: without it this refusal could be a fixture artefact.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    _install_stub(layer)
    _load_the_six(layer, text_config)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.forward(_tokens(text_config), **_call_kwargs(family))

    message = str(raised.value)
    say(f"refusal1-{family}", message[:150])
    assert "no streams" in message, message
    assert any(leaf in message for leaf in impl.MHC_LEAVES if leaf.startswith("hc_")), (
        f"the refusal names no mHC weight, so a reader cannot see why it fired: "
        f"{message}"
    )


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_030d_a_layer_carrying_none_refuses_a_streams_call(family: str) -> None:
    """Refusal 2 of the two, named -- the draft head's layer called with streams.

    The checkpoint gives layer 45 none of the six leaves (270 keys over layers
    0-44), so a streams call on such a layer is a caller error rather than a
    configuration to serve.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    _install_stub(layer)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.forward(
            _tokens(text_config), streams=_streams(text_config), **_call_kwargs(family)
        )

    message = str(raised.value)
    say(f"refusal2-{family}", message[:150])
    assert "carries none of the six" in message, message
    assert "pass no streams" in message, message


def test_030d_a_loaded_but_unbound_layer_refuses_by_naming_the_bind() -> None:
    """The third way to get this wrong, and it names the step that was skipped.

    A layer can hold the six tensors and still have no site, because the bind runs
    on the load path. That call cannot be served and it is not the draft head's
    case either, so the refusal names ``_run_load_time_preps`` rather than telling
    the caller to drop the streams.
    """
    impl = _impl()
    text_config = _text_config()
    layer = _norm_ready(_kda_layer(text_config), text_config)
    _install_stub(layer)
    _load_the_six(layer, text_config)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.forward(
            _tokens(text_config), streams=_streams(text_config), **_call_kwargs("kda")
        )

    message = str(raised.value)
    say("refusal-unbound", message[:170])
    assert "no site is bound" in message and "_run_load_time_preps" in message, message


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_030d_the_streams_route_runs_the_pair_around_the_same_attention_half(
    family: str,
) -> None:
    """The streams route: collapse, norm, attend, re-mix -- one entry per seam.

    FIVE READINGS, all counted rather than asserted:

    1. the attention half is entered exactly ONCE, on the collapsed ``[T, H]``
       single stream -- so the pair runs AROUND the sublayer, not beside it;
    2. the return is ``[T, S, H]``, the streams the carrier will keep;
    3. ``inc-glm53f-028``'s Sinkhorn seam is entered once per layer call;
    4. ``inc-glm53f-029``'s combine seam is entered once per layer call;
    5. both torch-fallback counters stay at ZERO -- a fallback would mean the
       reading measured torch against torch (P13, and the route predicate this
       block declares in form R-2: it reads the counters the two seams own).

    THE SEAMS HAVE NO TORCH PATH, so this item needs the NKI simulator
    (``NKI_SIMULATOR=1``, the Tier N harness this file's header records). A bare
    CPU-mode run raises from the seam, which is the intended behaviour rather than
    a failure of this route.
    """
    from vllm_neuron.functional.mhc import hyper_connection as combine_mod
    from vllm_neuron.functional.mhc import sinkhorn as sinkhorn_mod

    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    stub = _install_stub(layer)
    _load_the_six(layer, text_config)
    layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    streams = _streams(text_config)

    sinkhorn_mod.reset_dispatch_counters()
    combine_mod.reset_dispatch_counters()
    got = layer.forward(_tokens(text_config), streams=streams, **_call_kwargs(family))
    sink, comb = sinkhorn_mod.dispatch_counters(), combine_mod.dispatch_counters()

    say(f"streams-{family}", f"out={tuple(got.shape)}", f"stub_saw={stub.seen}")
    say(f"streams-{family}-route", f"sinkhorn={sink}", f"combine={comb}")

    tokens, sites, hidden = tuple(streams.shape)
    assert stub.seen == [(tokens, hidden)], (
        f"the attention half saw {stub.seen}; the pair must hand it exactly one "
        f"collapsed [T, H] stream per layer call"
    )
    assert tuple(got.shape) == (tokens, sites, hidden), (
        f"the route returned {tuple(got.shape)}, not the [T, S, H] streams the "
        f"carrier expects"
    )
    assert sink == (1, 0), (
        f"the Sinkhorn seam read {sink} for one layer call; one NKI dispatch and "
        f"no fallback is what per-layer-call means"
    )
    assert comb == (1, 0), (
        f"the combine seam read {comb} for one layer call; one NKI dispatch and no "
        f"fallback is what per-layer-call means"
    )


# --------------------------------------------------------------------------- #
# COMMIT 3 -- part (a), THE CARRIER. The embedding expands across the stream    #
# axis, every layer is handed streams, the FEED-FORWARD site runs here around   #
# ``_ffn_half``'s unchanged return, and an UNWEIGHTED mean collapses the        #
# streams before the final norm. Arms (2) and (3) of the block's acceptance run #
# at ``T = 128`` on the host; the items below read the composition's shape and   #
# its three cited steps at ``T = 8``.                                           #
# --------------------------------------------------------------------------- #
VOCAB = 16
QUANT = object()
BLOCKS = object()
MOE_GROUP = object()


class _StubSite:
    """Stands in for a bound mHC site, and RECORDS what it collapsed and mixed.

    NOT the real collapse. The real site weights the streams with ``mhc_pre``'s
    learned ``pre`` row (``reference:294``) and commit 2's items measure that seam;
    this stand-in collapses with a mean, because what the items below read is WHICH
    tensor the sublayer is handed and WHETHER its return comes back unchanged, not
    what the learned collapse computes. A plain object rather than an
    ``nn.Module``, so a stub cannot add a name to the tree the bind items count.
    """

    def __init__(self) -> None:
        self.residuals: list[tuple[int, ...]] = []
        self.given: list[torch.Tensor] = []
        self.returned: list[torch.Tensor] = []
        self.results: list[torch.Tensor] = []

    def forward(self, residual: torch.Tensor, sublayer: object) -> torch.Tensor:
        self.residuals.append(tuple(residual.shape))
        collapsed = residual.mean(dim=1)
        self.given.append(collapsed)
        produced = sublayer(collapsed)
        self.returned.append(produced)
        # FP32 OUT, as the landed combine seam does on purpose
        # (``inc-glm53f-030``): the carrier decides what to cast back to, and an
        # item that ran everything in one dtype could not read that decision.
        result = (residual.to(torch.float32) + produced.unsqueeze(1).to(torch.float32) * 0.5)
        self.results.append(result)
        return result


class _StubLayer(nn.Module):
    """Stands in for a decoder layer and records what the carrier handed it.

    It CHANGES the streams it is given, which is what makes the readings below
    non-vacuous: if it returned them untouched, "the mean collapsed the last
    layer's streams" and "the mean collapsed the embedding" would be the same
    sentence.
    """

    def __init__(self, family: str) -> None:
        super().__init__()
        self.family = family
        self.calls: list[dict] = []
        self.seen_streams: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor, *, streams=None, **kwargs):
        self.calls.append(
            {
                "input_shape": tuple(hidden_states.shape),
                "streams_shape": None if streams is None else tuple(streams.shape),
                "input_is_streams": hidden_states is streams,
                "carrier_keys": sorted(kwargs),
            }
        )
        self.seen_streams.append(streams)
        if streams is None:
            return hidden_states
        return streams * 1.25 + 0.01


def _stub_stack(text_config, *, tokens: int = TOKENS, dtype=torch.float32):
    """A ``Glm5NextModel`` whose layers are stubs, with the two mapped tensors loaded.

    The embedding table is allocated HERE at ``[VOCAB, H]`` rather than at the
    checkpoint's vocabulary, because this tree declares its parameters and
    allocates none until the load: nothing in the model reads ``vocab_size`` after
    construction, and the carrier only indexes the table it is given.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    impl = _impl()
    model = impl.Glm5NextModel(Glm5NextConfig(text_config=text_config), 1)
    hidden = int(text_config.hidden_size)
    stubs = [_StubLayer("kda"), _StubLayer("dsa")]
    for stub in stubs:
        stub.ffn_site = _StubSite()
        setattr(
            stub,
            impl.MHC_SITES_ATTR,
            {impl.MHC_ATTENTION_SITE: _StubSite(), impl.MHC_FFN_SITE: stub.ffn_site},
        )
    model.layers = nn.ModuleList(stubs)
    gen = torch.Generator().manual_seed(303)
    table = (torch.randn(VOCAB, hidden, generator=gen, dtype=torch.float32) * 0.05).to(dtype)
    model.embed_tokens_weight = nn.Parameter(table, requires_grad=False)
    model.norm_weight = nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)
    input_ids = torch.arange(tokens, dtype=torch.long) % VOCAB
    return model, stubs, table, input_ids


def _ffn_recorder():
    """A stand-in for ``_ffn_half`` that records every call and its own return."""
    seen: list[dict] = []
    produced: list[torch.Tensor] = []

    def _ffn_half(self, layer, hidden_states, **kwargs):
        seen.append(
            {
                "layer": layer,
                "shape": tuple(hidden_states.shape),
                "tensor": hidden_states,
                "kwargs": dict(kwargs),
            }
        )
        out = torch.tanh(hidden_states) * 0.75
        produced.append(out)
        return out

    return _ffn_half, seen, produced


def _run_carrier(model, input_ids, *, carriers=None):
    """Run the carrier. ``carriers`` defaults to empty mappings, which only the stub
    layers accept -- both real layer forwards declare required keyword-only carriers,
    so an item that builds real layers passes :func:`_call_kwargs` for each family."""
    return model.forward(
        input_ids,
        layer_carriers=carriers if carriers is not None else [{} for _ in model.layers],
        quant_config=QUANT,
        block_size=BLOCKS,
        moe_group=MOE_GROUP,
        tp_degree=4,
        expert_parallel_rank=2,
    )


def test_030d_the_carrier_expands_the_embedding_across_the_stream_axis() -> None:
    """Every layer is handed ``[T, hc_mult, H]``, and at entry every stream is the token.

    ``reference:1477`` expands the embeddings across a new stream axis before the
    stack runs, so the first layer's four streams are four copies of the same token
    vector. Bitwise, because an expand copies rather than computes.

    The stream count is READ OFF THE CONFIG, so a checkpoint that carried a
    different ``hc_mult`` would reach this item rather than being overwritten by it.
    """
    text_config = _text_config()
    model, stubs, table, input_ids = _stub_stack(text_config)
    ffn_half, seen, _ = _ffn_recorder()
    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        out = _run_carrier(model, input_ids)

    embedded = table[input_ids]
    first = stubs[0].seen_streams[0]
    per_stream_equal = [
        bool(torch.equal(first[:, s, :], embedded)) for s in range(hc_mult)
    ]
    say("carrier-expand", f"ids={tuple(input_ids.shape)}", f"out={tuple(out.shape)}")
    say("carrier-expand", f"first_layer_streams={tuple(first.shape)}",
        f"streams_equal_the_embedding={per_stream_equal}")
    say("carrier-expand", f"ffn_half_calls={len(seen)}",
        f"layer_calls={[len(s.calls) for s in stubs]}")

    assert tuple(first.shape) == (len(input_ids), hc_mult, hidden), (
        f"the first layer was handed {tuple(first.shape)}; the carrier is "
        f"[T, hc_mult, H] and hc_mult comes off the config as {hc_mult}"
    )
    assert all(per_stream_equal), (
        f"at entry the streams read {per_stream_equal} against the embedding; "
        f"reference:1477 expands one token vector into every stream"
    )
    assert [c["streams_shape"] for s in stubs for c in s.calls] == [
        (len(input_ids), hc_mult, hidden)
    ] * len(stubs), "every layer in the stack must be handed the streams"
    assert all(c["input_is_streams"] for s in stubs for c in s.calls), (
        "the positional input and the streams keyword must be the SAME tensor; "
        "reference:1481 hands the decoder layer the four-stream tensor itself"
    )
    # NON-VACUOUS: the stub layer changes the streams, so the SECOND layer's input
    # is no longer the embedding. A carrier that expanded once per layer would pass
    # the reading above and fail this one.
    second = stubs[1].seen_streams[0]
    assert not torch.equal(second[:, 0, :], embedded), (
        "the second layer saw the embedding again; the streams must be the first "
        "layer's output, not a fresh expand"
    )


def test_030d_the_carrier_collapses_with_an_unweighted_mean() -> None:
    """The final norm sees ``streams.mean(dim=1)``, not a weighted sum.

    ``reference:302`` is the one collapse in this model that carries no learned
    weight -- the target model's own class comment says "unlike DeepSeek-V4, this is
    an unweighted mean" -- and ``reference:1493`` puts it BEFORE the norm. Both are
    read here: the tensor the norm was handed, and the order it was handed it in.

    THE CONTROL IS A WEIGHTED COLLAPSE over the same streams. It differs, printed,
    so this item distinguishes the mean from the mHC-style collapse the two sites
    use -- which is the mistake a reader of this file would most plausibly make.
    """
    text_config = _text_config()
    # BF16 TABLE, FP32 SEAM RETURN: the streams come back from each site in fp32, so
    # the output's dtype reads the carrier's cast rather than the fixture's one dtype.
    model, stubs, table, input_ids = _stub_stack(text_config, dtype=torch.bfloat16)
    ffn_half, _, _ = _ffn_recorder()
    normed: list[torch.Tensor] = []
    real_norm = type(model)._rms_norm

    def _recording_norm(self, hidden_states, gain):
        normed.append(hidden_states)
        return real_norm(self, hidden_states, gain)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        patch.setattr(type(model), "_rms_norm", _recording_norm)
        out = _run_carrier(model, input_ids)

    last = stubs[-1].ffn_site.results[-1]
    embedded = table[input_ids]
    want = last.mean(dim=1).to(embedded.dtype)
    # THE CONTROL'S WEIGHTS ARE DERIVED from the stream count, so a config carrying a
    # different hc_mult reaches this item instead of an index error.
    weights = torch.linspace(0.1, 0.4, int(text_config.hc_mult), dtype=torch.float32)
    weighted = (last * weights[None, :, None]).sum(dim=1).to(embedded.dtype)
    spread = float((last.max(dim=1).values - last.min(dim=1).values).abs().max())

    say("carrier-mean", f"norm_calls={len(normed)}", f"out={tuple(out.shape)}",
        f"out_dtype={out.dtype}")
    say("carrier-mean", f"stream_spread={spread:.6g}",
        f"weighted_control_delta={float((weighted - want).abs().max()):.6g}")

    assert len(normed) == 1, (
        f"the final norm ran {len(normed)} times; the carrier norms once, after the "
        f"collapse"
    )
    assert torch.equal(normed[0], want), (
        "the norm was handed something other than the unweighted mean of the last "
        f"layer's streams; max deviation {float((normed[0] - want).abs().max())}"
    )
    assert tuple(normed[0].shape) == (len(input_ids), int(text_config.hidden_size))
    assert out.dtype == embedded.dtype, (
        f"the carrier returned {out.dtype}; the combine seam returns fp32 and the "
        f"cast back to the table's dtype is this carrier's decision"
    )
    assert spread > 0.0, (
        "the four streams reaching the collapse are identical, so a mean and a "
        "weighted sum would agree and this item would read nothing"
    )
    assert not torch.equal(weighted, want), (
        "a weighted collapse over these streams equals the mean, so the reading "
        "above cannot tell the two apart"
    )


def test_030d_the_ffn_site_runs_over_ffn_halfs_unchanged_return() -> None:
    """``_ffn_half`` is CALLED with one collapsed stream and its return is mixed back.

    ``reference:1321-1327`` runs ``ffn_hc`` around ``mlp``: collapse, feed forward,
    mix. In this tree the feed-forward half is ``Glm5NextModel._ffn_half``
    (``inc-glm53f-054a``), so the site composes in the carrier and the call itself is
    unchanged -- same ``layer`` object, same ``[T, H]`` shape, same five forwarded
    keywords, and the object it returns is the object the site mixes.

    IDENTITY, NOT EQUALITY, on that last reading: ``is`` cannot pass by two tensors
    happening to agree.
    """
    text_config = _text_config()
    model, stubs, _, input_ids = _stub_stack(text_config)
    ffn_half, seen, produced = _ffn_recorder()
    hidden = int(text_config.hidden_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        _run_carrier(model, input_ids)

    sites = [stub.ffn_site for stub in stubs]
    say("ffn-site", f"ffn_half_calls={len(seen)}", f"layers={len(stubs)}")
    say("ffn-site", f"shapes={[c['shape'] for c in seen]}",
        f"site_residuals={[s.residuals for s in sites]}")
    say("ffn-site", f"kwargs={sorted(seen[0]['kwargs'])}",
        f"returns_mixed_unchanged="
        f"{[s.returned[0] is p for s, p in zip(sites, produced)]}")

    assert len(seen) == len(stubs), (
        f"_ffn_half ran {len(seen)} times for {len(stubs)} layers; the carrier "
        f"composes one feed-forward half per layer"
    )
    assert [c["shape"] for c in seen] == [(len(input_ids), hidden)] * len(stubs), (
        "the feed-forward half must be handed ONE collapsed [T, H] stream, which is "
        "the shape it has always taken"
    )
    assert [c["layer"] for c in seen] == list(stubs), (
        "each feed-forward half must run against its own layer"
    )
    for site, call in zip(sites, seen):
        assert site.residuals == [(len(input_ids), int(text_config.hc_mult), hidden)]
        # IDENTITY: the tensor the feed-forward half was handed is the tensor this
        # site collapsed, so no copy or re-collapse sits between them.
        assert call["tensor"] is site.given[0], (
            "the feed-forward half was handed a different tensor from the one the "
            "site collapsed"
        )
        assert sorted(call["kwargs"]) == [
            "block_size",
            "expert_parallel_rank",
            "moe_group",
            "quant_config",
            "tp_degree",
        ]
        assert call["kwargs"]["quant_config"] is QUANT
        assert call["kwargs"]["block_size"] is BLOCKS
        assert call["kwargs"]["moe_group"] is MOE_GROUP
        assert call["kwargs"]["tp_degree"] == 4
        assert call["kwargs"]["expert_parallel_rank"] == 2
    assert [s.returned[0] is p for s, p in zip(sites, produced)] == [True] * len(stubs), (
        "the site mixed something other than the object _ffn_half returned; the "
        "plan's word is that _ffn_half is called, never edited"
    )


def test_030d_the_hybrid_stack_runs_both_families_through_real_sites() -> None:
    """One KDA layer and one DSA layer, both bound, both composed, end to end.

    This is the block's arm (2) at this file's token count: the composition runs at
    BOTH sublayer sites of BOTH attention families, through the REAL
    :class:`Glm5NextHyperConnection` -- the attention site inside each layer forward
    and the feed-forward site in the carrier. Only two things are stood in for: the
    attention half (another increment's kernels) and ``_ffn_half`` (the expert bank).
    The arm's numeric equality against the independent reference is the host run's,
    at ``T = 128``.

    The bind is the layer's own public method, so this item runs the same code path
    the load path runs.
    """
    impl = _impl()
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    text_config = _text_config()
    model = impl.Glm5NextModel(Glm5NextConfig(text_config=text_config), 1)
    hidden = int(text_config.hidden_size)
    stubs = []
    for layer in model.layers:
        _norm_ready(layer, text_config)
        _load_the_six(layer, text_config)
        stubs.append(_install_stub(layer))
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    gen = torch.Generator().manual_seed(304)
    table = torch.randn(VOCAB, hidden, generator=gen, dtype=torch.float32) * 0.05
    model.embed_tokens_weight = nn.Parameter(table, requires_grad=False)
    model.norm_weight = nn.Parameter(
        torch.ones(hidden, dtype=torch.float32), requires_grad=False
    )
    input_ids = torch.arange(TOKENS, dtype=torch.long) % VOCAB
    ffn_half, seen, _ = _ffn_recorder()

    carriers = [_call_kwargs("kda"), _call_kwargs("dsa")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        out = _run_carrier(model, input_ids, carriers=carriers)

    families = [type(layer).__name__ for layer in model.layers]
    site_pairs = [
        sorted(getattr(layer, impl.MHC_SITES_ATTR)) for layer in model.layers
    ]
    distinct = [
        getattr(layer, impl.MHC_SITES_ATTR)[impl.MHC_ATTENTION_SITE]
        is not getattr(layer, impl.MHC_SITES_ATTR)[impl.MHC_FFN_SITE]
        for layer in model.layers
    ]
    say("hybrid", f"families={families}", f"sites={site_pairs}",
        f"two_instances_per_layer={distinct}")
    say("hybrid", f"attention_saw={[s.seen for s in stubs]}",
        f"ffn_saw={[c['shape'] for c in seen]}")
    say("hybrid", f"out={tuple(out.shape)}", f"dtype={out.dtype}",
        f"finite={bool(torch.isfinite(out).all())}")

    assert len(set(families)) == 2, (
        f"this stack holds only {families}; arm (2) wants both attention families"
    )
    assert all(distinct), "each layer must hold TWO sites, one per sublayer"
    assert [s.seen for s in stubs] == [[(len(input_ids), hidden)]] * len(stubs), (
        "each attention half must be entered exactly once, on the collapsed "
        "[T, H] stream the site handed it"
    )
    assert [c["shape"] for c in seen] == [(len(input_ids), hidden)] * len(stubs)
    assert tuple(out.shape) == (len(input_ids), hidden)
    assert out.dtype == table.dtype
    assert bool(torch.isfinite(out).all()), (
        "the four-stream composition returned a non-finite value through the real "
        "sinkhorn and combine seams"
    )


def test_030d_a_stack_whose_layers_carry_no_mhc_weight_refuses_by_name() -> None:
    """The unconditional carrier's other half: no weights, no service.

    The carrier passes streams to every layer whatever the checkpoint holds, so the
    refusal is what keeps a weightless stack from being served a different network.
    This is the FIRING CONTROL for that design decision: same carrier, same call,
    six leaves absent, and the message names the one-stream path.
    """
    impl = _impl()
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    text_config = _text_config()
    model = impl.Glm5NextModel(Glm5NextConfig(text_config=text_config), 1)
    hidden = int(text_config.hidden_size)
    for layer in model.layers:
        _norm_ready(layer, text_config)
        _install_stub(layer)
    gen = torch.Generator().manual_seed(305)
    model.embed_tokens_weight = nn.Parameter(
        torch.randn(VOCAB, hidden, generator=gen, dtype=torch.float32) * 0.05,
        requires_grad=False,
    )
    model.norm_weight = nn.Parameter(
        torch.ones(hidden, dtype=torch.float32), requires_grad=False
    )
    ffn_half, seen, _ = _ffn_recorder()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        with pytest.raises(impl.Glm5NextHyperConnectionError) as caught:
            _run_carrier(
                model,
                torch.arange(TOKENS, dtype=torch.long) % VOCAB,
                carriers=[_call_kwargs("kda"), _call_kwargs("dsa")],
            )

    message = str(caught.value)
    say("weightless-stack", f"ffn_half_calls={len(seen)}", f"message={message[:110]}")
    assert "carries none of the six" in message, message
    assert "pass no streams" in message, message
    assert seen == [], (
        "the refusal must come before any feed-forward half runs, so a weightless "
        "stack cannot half-serve a request"
    )


def test_030d_the_carrier_refuses_a_stream_count_that_is_not_positive() -> None:
    """``hc_mult`` 0 cannot build a carrier, and the message says which value it is.

    The stream count is the checkpoint's, read on every call. A zero would expand to
    an empty stream axis, every collapse would be a mean over nothing, and the stack
    would return NaNs rather than refusing.
    """
    text_config = _text_config(hc_mult=0)
    model, _, _, input_ids = _stub_stack(text_config)
    ffn_half, seen, _ = _ffn_recorder()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        with pytest.raises(ValueError) as caught:
            _run_carrier(model, input_ids)

    message = str(caught.value)
    say("hc-mult-zero", f"message={message[:110]}", f"ffn_half_calls={len(seen)}")
    assert "hc_mult=0" in message, message
    assert seen == []


def test_030d_the_role_map_agrees_with_the_reference_by_name() -> None:
    """The three roles are matched to the reference's own three parameters BY NAME.

    Every other item in this file reads the role map off the product, so a
    ``base``-for-``scale`` swap in ``MHC_ROLE_PARAMETERS`` would pass all of them and
    be caught only numerically, by arm (3), on the host. This item closes that gap
    cheaply and independently: the reference's ``__init__`` declares ``fn``, ``base``
    and ``scale`` with three DIFFERENT shapes (``reference:259-265``), so comparing
    each role's bound parameter shape against the reference's same-named parameter
    catches a swap here, on this machine, with no reference run.

    The extents are EVALUATED FROM THE REFERENCE'S OWN SPELLING, not retyped: the
    two names its expressions use are bound to this config's values, and any other
    name refuses.
    """
    path = _reference_path()
    if path is None:
        pytest.skip("the campaign reference copy is not on this machine")
    impl = _impl()
    text_config = _text_config()
    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)

    tree = ast.parse(Path(path).read_text())
    declared: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith("HyperConnection")):
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            target = stmt.targets[0]
            if not (isinstance(target, ast.Attribute) and isinstance(stmt.value.func, ast.Attribute)):
                continue
            if stmt.value.func.attr != "Parameter":
                continue
            inner = stmt.value.args[0]
            if not isinstance(inner, ast.Call):
                continue
            declared[target.attr] = tuple(ast.unparse(a) for a in inner.args)

    allowed = {"mix": (2 + hc_mult) * hc_mult, "hidden": hidden, "hc_mult": hc_mult}

    def extent(expr: str) -> int:
        spelled = expr.replace("self.hc_mult", "hc_mult").replace(
            "config.hidden_size", "hidden"
        )
        names = {n.id for n in ast.walk(ast.parse(spelled, mode="eval")) if isinstance(n, ast.Name)}
        assert names <= set(allowed), (
            f"the reference spells an extent this item cannot resolve: {expr!r} uses "
            f"{sorted(names - set(allowed))}"
        )
        return int(eval(compile(ast.parse(spelled, mode="eval"), "<ref>", "eval"), {"__builtins__": {}}, allowed))

    want = {role: tuple(extent(e) for e in args) for role, args in declared.items()}
    site = impl.Glm5NextHyperConnection(text_config, neuron_config=None)
    got = {
        role: tuple(getattr(site, impl.MHC_ROLE_PARAMETERS[role]).shape)
        for role in sorted(impl.MHC_ROLE_PARAMETERS)
    }

    say("role-map", f"reference_declares={want}")
    say("role-map", f"this_tree_binds={got}", f"map={impl.MHC_ROLE_PARAMETERS}")

    assert sorted(want) == sorted(impl.MHC_ROLE_PARAMETERS), (
        f"the reference declares {sorted(want)} and this tree maps "
        f"{sorted(impl.MHC_ROLE_PARAMETERS)}; the three roles must be the same three"
    )
    assert got == want, (
        f"role-to-parameter mapping disagrees with the reference by shape: this tree "
        f"binds {got}, the reference declares {want}. A swapped pair would compute a "
        f"plausible number and only arm (3) would object"
    )
    assert len({tuple(v) for v in want.values()}) == len(want), (
        "the reference's three parameters do not have three distinct shapes, so this "
        "item cannot catch a swap and the gap it closes is still open"
    )
