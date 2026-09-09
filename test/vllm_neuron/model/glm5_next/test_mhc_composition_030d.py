# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-030d``: the mHC four-stream composition, one commit at a time.

WHAT THIS FILE IS FOR. ``inc-glm53f-030`` landed the mHC layer and measured its
arithmetic; ``inc-glm53f-030c`` corrected that arithmetic against the checkpoint's
own model file. Neither of them wired the layer into the decoder: the class was
bound to no layer at all. ``-030d`` does the wiring, in four commits, and this
file grows with the first three of them.

COMMIT 1 IS THE BIND, and it is the whole subject of every item below. The six
mHC tensors the checkpoint carries per layer -- ``hc_attn_{base,fn,scale}`` and
``hc_ffn_{base,fn,scale}``, 270 keys over layers 0-44 -- reach two
:class:`Glm5NextHyperConnection` instances per layer ONCE, after the load, on the
same load-time-prep walk this tree already runs for its projection and scale
preps. Nothing here runs a forward: what a bound site COMPUTES is commit 2's and
commit 3's subject, measured against the reference pair.

WHY THE TWO INSTANCES ARE NOT SUBMODULES, which is the one design decision this
commit had to make. Registering them would put their three parameters each into
``named_parameters()``, and two landed readings count that list exactly: it reads
``0`` on a declared-but-unloaded tree and the declared count after materialisation
(``test_load_weights.py:960-980``, with the pre-load zero read again at
``test_kv_spec.py:621``). So the sites are held in a plain dict, and
:func:`test_030d_the_bind_adds_no_name_to_named_parameters_or_the_state_dict`
measures that with a firing control -- a registered site DOES add three names, so
the counted zero is a reading rather than a tautology.

NO TOLERANCE AND NO COMPARATOR IS REGISTERED HERE. Commit 1 touches no
arithmetic: every item is a structural or a bookkeeping reading. The registered
pair stays ``test_mhc_layer.py``'s ``(1e-2, 1e-5)``, cited when commits 2 and 3
need it (P9).

THE IMPLEMENTATION MODULE IS IMPORTED INSIDE TEST BODIES, never at module scope,
for the reason ``test_mhc_layer.py:102-107`` records: ``test_factory.py``'s C03
asserts ``model_fp8`` is absent from ``sys.modules``, and pytest imports every
collected module before running any test.

Command (Tier N harness, plan rev 286)::

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
