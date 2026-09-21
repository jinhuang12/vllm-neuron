# SPDX-License-Identifier: Apache-2.0
"""The mHC four-stream composition: the load-time bind, the two sites, the stack.

Covers what the bind accepts and refuses, which route a call takes, and the
stream carrier the decoder stack expands, mixes and collapses.
"""

from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config(**overrides):
    """A two-layer hybrid config: one layer of each attention family, all dense. """
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
    """The shape each of the six leaves has in the checkpoint, derived. """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)
    mix = (2 + hc_mult) * hc_mult
    by_role = {"fn": (mix, hc_mult * hidden), "base": (mix,), "scale": (3,)}
    return {leaf: by_role[leaf.split("_")[2]] for leaf in MHC_LEAVES}


def _load_the_six(layer, text_config, *, seed: int = 30, skip: tuple[str, ...] = ()):
    """Put a real tensor on each of the six leaves, the way a load leaves them. """
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
    """Stands in for the attention half, and records what it was handed. """

    def __init__(self, half=None) -> None:
        super().__init__()
        self.seen: list[tuple[int, ...]] = []
        # The dtype is recorded beside the shape rather than inside it, because the
        # tests compare `seen` to a list of shapes and this commit does not
        # move their readings (commit 4, the cast points).
        self.seen_dtypes: list[torch.dtype] = []
        self._half = half

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        self.seen.append(tuple(hidden_states.shape))
        self.seen_dtypes.append(hidden_states.dtype)
        if self._half is not None:
            return self._half(hidden_states)
        return torch.tanh(hidden_states) * 1.5


def _install_stub(layer, half=None) -> _StubAttention:
    """Replace the attention half, under the attribute the class itself names. """
    stub = _StubAttention(half)
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
    """The carriers each family's forward requires, as values the stub ignores. """
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
        # The paged operands the sparse forward declares. Shapes, not values: this
        # family's carrier names the blocks its request holds and the bank rows its
        # tokens are written to, and the stub below reads neither.
        "block_table_row": torch.zeros((1, 1), dtype=torch.int32),
        "latent_slots": torch.zeros(1, dtype=torch.int64),
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


# --------------------------------------------------------------------------- #
# (1) The six leaves group by site off the map's own tuple.                    #
# --------------------------------------------------------------------------- #
def test_the_six_leaves_group_by_site_off_the_maps_own_tuple() -> None:
    """The grouping is derived from ``MHC_LEAVES``, and a strange leaf raises. """
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    impl = _impl()
    grouped = impl._mhc_leaves_by_site()
    flat = sorted(leaf for roles in grouped.values() for leaf in roles.values())


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
    assert "hc_attn_gain" in str(raised.value), (
        f"the refusal does not name the leaf it could not resolve: {raised.value}"
    )


# --------------------------------------------------------------------------- #
# (2) A loaded layer binds both sites, and the tensors are the loaded ones.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_a_loaded_layer_binds_both_sites_to_the_loaded_tensors(
    family: str,
) -> None:
    """Both families, both sites, and storage identity rather than value equality. """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config) if family == "kda" else _dsa_layer(text_config)
    placed = _load_the_six(layer, text_config)

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)


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
def test_a_layer_carrying_none_of_the_six_is_skipped_and_recorded() -> None:
    """The draft head's case: all six declared, none loaded, nothing bound. """
    impl = _impl()
    text_config = _text_config()
    layer = _dsa_layer(text_config)

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)


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


def test_a_partially_loaded_layer_refuses_by_name() -> None:
    """Five of six loaded is the defect case, and the refusal names the missing one. """
    impl = _impl()
    text_config = _text_config()

    control = _kda_layer(text_config)
    _load_the_six(control, text_config)
    control_sites = control.bind_hyper_connection_sites(
        text_config, torch.device("cpu")
    )
    assert control_sites == 2, "the control fixture does not bind, so it controls nothing"

    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config, skip=("hc_ffn_scale",))
    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))

    message = str(raised.value)
    assert "hc_ffn_scale" in message, (
        f"the refusal does not name the leaf that is missing: {message}"
    )
    assert not hasattr(layer, impl.MHC_SITES_ATTR), (
        "the refusal left a sites attribute behind, so a caller cannot tell a "
        "refused bind from a completed one"
    )


# --------------------------------------------------------------------------- #
# (5) An unmaterialised placeholder refuses by name.                          #
# --------------------------------------------------------------------------- #
def test_an_unmaterialised_placeholder_refuses_by_name() -> None:
    """A registered placeholder is not a loaded tensor, and the bind says so. """
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
    assert "hc_attn_fn" in message and "placeholder" in message, (
        f"the refusal does not name the placeholder it found: {message}"
    )


# --------------------------------------------------------------------------- #
# (6) A leaf that is not on the load's device refuses by name.                 #
# --------------------------------------------------------------------------- #
def test_a_leaf_off_the_loads_device_refuses_by_name() -> None:
    """The exposure that makes this check necessary is measured, not asserted. """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.bind_hyper_connection_sites(text_config, torch.device("meta"))
    message = str(raised.value)
    assert "cpu" in message and "meta" in message, (
        f"the refusal does not name both places: {message}"
    )

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    assert bound_sites == 2, (
        "the same layer does not bind on the device its tensors are on, so the "
        "refusal above says nothing about the device check"
    )


# --------------------------------------------------------------------------- #
# (7) The bind adds no name to named_parameters or the state dict.           #
# --------------------------------------------------------------------------- #
def test_the_bind_adds_no_name_to_named_parameters_or_the_state_dict() -> None:
    """The counted zero, with the control that makes it mean something. """
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


    assert after_params == before_params, (
        f"the bind added {sorted(after_params - before_params)} to "
        f"named_parameters(); the load readings count that list exactly"
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
    assert len(added) == 3, (
        f"registering a site added {len(added)} parameter names, not the three the "
        f"class holds; the invisibility reading above measures nothing if a "
        f"registered site is invisible too"
    )


# --------------------------------------------------------------------------- #
# (8) One rule, two families, one production call site.                       #
# --------------------------------------------------------------------------- #
def test_both_families_delegate_to_one_body() -> None:
    """The rule lives once; each family holds a signature and a return. """
    _impl()
    _text, tree = _model_source()

    methods = _function_defs(tree, "bind_hyper_connection_sites")
    bodies = _function_defs(tree, "_bind_hyper_connection_sites")

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
    assert impl.MHC_SITES_ATTR != impl.MHC_BIND_HEALTH_ATTR


def test_the_load_path_is_the_single_production_call_site() -> None:
    """Exactly one call in the package, and it sits inside the prep walk. """
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


    assert len(calls) == 1, (
        f"the package holds {len(calls)} calls of the bind method {calls}; one "
        f"production caller is the whole point of putting it on the load walk"
    )

    _text, tree = _model_source()
    walk = _function_defs(tree, "_run_load_time_preps")
    assert len(walk) == 1
    line = int(calls[0].split(":")[1])
    assert walk[0].lineno < line <= int(walk[0].end_lineno), (
        f"the one call sits at line {line}, outside "
        f"_run_load_time_preps ({walk[0].lineno}-{walk[0].end_lineno}); the bind "
        f"must run where the weights are already on the device"
    )


# --------------------------------------------------------------------------- #
# (9) The whole stack answers the gate, and the pre-load zero still holds.     #
# --------------------------------------------------------------------------- #
def test_every_decoder_layer_answers_the_gate_and_the_tree_stays_empty() -> None:
    """A hybrid stack: every layer is a bind candidate and nothing is allocated. """
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
def test_the_framework_overrides_reach_both_sites() -> None:
    """``mhc_sinkhorn_iters`` and ``mhc_eps`` travel through ``neuron_config``. """
    from vllm_neuron.model.neuron_config import NeuronConfig

    impl = _impl()

    plain = _text_config()
    plain_layer = _kda_layer(plain)
    _load_the_six(plain_layer, plain)
    plain_layer.bind_hyper_connection_sites(plain, torch.device("cpu"))
    plain_sites = getattr(plain_layer, impl.MHC_SITES_ATTR)
    for site in sorted(plain_sites):
        instance = plain_sites[site]
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
        assert instance.sinkhorn_iters == 3, (
            f"site {site} runs {instance.sinkhorn_iters} Sinkhorn iterations where "
            f"the framework asked for 3, so the override did not reach it"
        )
        assert instance.hc_eps == pytest.approx(1e-3)
    assert int(overridden.hc_sinkhorn_iters) != 3, (
        "the override equals the checkpoint's own value, so this test cannot tell "
        "an override that arrived from one that was ignored"
    )


def test_the_post_gate_multiplier_is_the_targets_own_two() -> None:
    """Every bound site takes the target's factor of 2."""
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)
    layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)

    for site, instance in sorted(getattr(layer, impl.MHC_SITES_ATTR).items()):
        assert instance.post_mult_value == 2.0, (
            f"site {site} carries a post multiplier of {instance.post_mult_value}; "
            f"the target's factor is 2 and the bind chooses no number"
        )
    assert health["post_mult_value"] == 2.0


# --------------------------------------------------------------------------- #
# (11) A second bind is counted rather than refused.                          #
# --------------------------------------------------------------------------- #
def test_a_second_bind_is_counted_rather_than_refused() -> None:
    """Re-binding must work, because a test calls the prep walk twice. """
    impl = _impl()
    text_config = _text_config()
    layer = _kda_layer(text_config)
    _load_the_six(layer, text_config)

    first = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    first_health = dict(getattr(layer, impl.MHC_BIND_HEALTH_ATTR))
    second = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    second_health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)


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
# The route: one optional keyword, and a branch that refuses both            #
# ways. Arm (4) of the block's acceptance is these two named refusals; arm (5)  #
# is the three layer suites re-run unedited, which is a run and not an   #
# test here.                                                                   #
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


def test_the_site_names_are_the_leaves_own_middle_words() -> None:
    """The two site constants are the derived keys, not two strings typed twice. """
    impl = _impl()
    derived = sorted(impl._mhc_leaves_by_site())
    named = sorted({impl.MHC_ATTENTION_SITE, impl.MHC_FFN_SITE})
    assert derived == named, (
        f"the leaves group into {derived} and the forwards index {named}; a "
        f"mismatch is a KeyError on a served token"
    )
    assert impl.MHC_ATTENTION_SITE != impl.MHC_FFN_SITE


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_the_one_stream_route_computes_what_it_computed_before(
    family: str,
) -> None:
    """No streams on a layer with no mHC weight: pre-norm, attend, plain add. """
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    stub = _install_stub(layer)
    hidden = _tokens(text_config)

    got = layer.forward(hidden, **_call_kwargs(family))
    # The want runs the same stub on the same norm, so this test cannot pass by
    # agreeing with a formula copied out of the stub. That second entry is why
    # ``stub.seen`` reads two shapes below.
    want = hidden + stub(layer._input_norm(hidden))

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
def test_a_layer_carrying_mhc_weights_refuses_a_one_stream_call(
    family: str,
) -> None:
    """Refusal 1 of the two the block declares, named. """
    impl = _impl()
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    _install_stub(layer)
    _load_the_six(layer, text_config)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.forward(_tokens(text_config), **_call_kwargs(family))

    message = str(raised.value)
    assert "no streams" in message, message
    assert any(leaf in message for leaf in impl.MHC_LEAVES if leaf.startswith("hc_")), (
        f"the refusal names no mHC weight, so a reader cannot see why it fired: "
        f"{message}"
    )


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_a_layer_carrying_none_refuses_a_streams_call(family: str) -> None:
    """Refusal 2 of the two, named -- the draft head's layer called with streams. """
    impl = _impl()
    text_config = _text_config()
    layer = _norm_ready(_layer_of(family, text_config), text_config)
    _install_stub(layer)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        layer.forward(
            _tokens(text_config), streams=_streams(text_config), **_call_kwargs(family)
        )

    message = str(raised.value)
    assert "carries none of the six" in message, message
    assert "pass no streams" in message, message


def test_a_loaded_but_unbound_layer_refuses_by_naming_the_bind() -> None:
    """The third way to get this wrong, and it names the step that was skipped. """
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
    assert "no site is bound" in message and "_run_load_time_preps" in message, message


@pytest.mark.parametrize("family", ["kda", "dsa"])
def test_the_streams_route_runs_the_pair_around_the_same_attention_half(
    family: str,
) -> None:
    """The streams route: collapse, norm, attend, re-mix -- one entry per seam. """
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
# commit 3 -- part (a), the carrier. The embedding expands across the stream    #
# axis, every layer is handed streams, the feed-forward site runs here around   #
# ``_ffn_half``'s unchanged return, and an unweighted mean collapses the        #
# streams before the final norm. Arms (2) and (3) of the block's acceptance run #
# at ``T = 128`` on the host; the tests below read the composition's shape and   #
# its three cited steps at ``T = 8``.                                           #
# --------------------------------------------------------------------------- #
VOCAB = 16
QUANT = object()
BLOCKS = object()
MOE_GROUP = object()


class _StubSite:
    """Stands in for a bound mHC site, and records what it collapsed and mixed. """

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
        # FP32 out, as the combine seam does on purpose
        # -- the carrier decides what to cast back to, and an
        # test that ran everything in one dtype could not read that decision.
        result = (residual.to(torch.float32) + produced.unsqueeze(1).to(torch.float32) * 0.5)
        self.results.append(result)
        return result


class _StubLayer(nn.Module):
    """Stands in for a decoder layer and records what the carrier handed it. """

    def __init__(self, family: str) -> None:
        super().__init__()
        self.family = family
        self.calls: list[dict] = []
        self.seen_streams: list[torch.Tensor] = []
        # Per-stream gains, off by default. A 1-D tensor of one gain per stream, or
        # none for the single gain below.
        self.stream_gains = None

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
        if self.stream_gains is None:
            return streams * 1.25 + 0.01
        # The gain is indexed on the stream axis, so a config carrying a different
        # ``hc_mult`` reaches the test's own check rather than a broadcast error here.
        gains = self.stream_gains.to(streams.dtype)
        return streams * gains[None, : streams.shape[1], None] + 0.01


def _stub_stack(text_config, *, tokens: int = TOKENS, dtype=torch.float32):
    """A ``Glm5NextModel`` whose layers are stubs, with the two mapped tensors loaded.
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
    """Run the carrier. """
    return model.forward(
        input_ids,
        layer_carriers=carriers if carriers is not None else [{} for _ in model.layers],
        quant_config=QUANT,
        block_size=BLOCKS,
        moe_group=MOE_GROUP,
        tp_degree=4,
        expert_parallel_rank=2,
    )


def test_the_carrier_expands_the_embedding_across_the_stream_axis() -> None:
    """Every layer is handed ``[T, hc_mult, H]``, and at entry every stream is the token.
    """
    text_config = _text_config()
    model, stubs, table, input_ids = _stub_stack(text_config)
    ffn_half, _seen, _ = _ffn_recorder()
    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        _run_carrier(model, input_ids)

    embedded = table[input_ids]
    first = stubs[0].seen_streams[0]
    per_stream_equal = [
        bool(torch.equal(first[:, s, :], embedded)) for s in range(hc_mult)
    ]

    assert tuple(first.shape) == (len(input_ids), hc_mult, hidden), (
        f"the first layer was handed {tuple(first.shape)}; the carrier is "
        f"[T, hc_mult, H] and hc_mult comes off the config as {hc_mult}"
    )
    assert all(per_stream_equal), (
        f"at entry the streams read {per_stream_equal} against the embedding; "
        f"the reference expands one token vector into every stream"
    )
    assert [c["streams_shape"] for s in stubs for c in s.calls] == [
        (len(input_ids), hc_mult, hidden)
    ] * len(stubs), "every layer in the stack must be handed the streams"
    assert all(c["input_is_streams"] for s in stubs for c in s.calls), (
        "the positional input and the streams keyword must be the SAME tensor; "
        "the reference hands the decoder layer the four-stream tensor itself"
    )
    # Non-vacuous: the stub layer changes the streams, so the second layer's input
    # is no longer the embedding. A carrier that expanded once per layer would pass
    # the reading above and fail this one.
    second = stubs[1].seen_streams[0]
    assert not torch.equal(second[:, 0, :], embedded), (
        "the second layer saw the embedding again; the streams must be the first "
        "layer's output, not a fresh expand"
    )


def test_the_carrier_collapses_with_an_unweighted_mean() -> None:
    """The final norm sees ``streams.mean(dim=1)``, not a weighted sum. """
    text_config = _text_config()
    # BF16 table, FP32 seam return: the streams come back from each site in fp32, so
    # the output's dtype reads the carrier's cast rather than the fixture's one dtype.
    model, stubs, table, input_ids = _stub_stack(text_config, dtype=torch.bfloat16)
    # Distinct gains, one per stream, derived from the stream count so a different
    # ``hc_mult`` reaches this test rather than an index error. They are not all equal,
    # which is the whole point, and the guard below reads the spread they produce
    # instead of trusting them to have worked.
    gains = torch.linspace(1.0, 1.3, int(text_config.hc_mult), dtype=torch.float32)
    assert float(gains.max() - gains.min()) > 0.0, (
        "the per-stream gains are all equal, so this fixture cannot tell a mean from a "
        "weighted collapse and every reading below would be vacuous"
    )
    for stub in stubs:
        stub.stream_gains = gains
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
    # The control's weights are derived from the stream count, so a config carrying a
    # different hc_mult reaches this test instead of an index error.
    weights = torch.linspace(0.1, 0.4, int(text_config.hc_mult), dtype=torch.float32)
    weighted = (last * weights[None, :, None]).sum(dim=1).to(embedded.dtype)
    spread = float((last.max(dim=1).values - last.min(dim=1).values).abs().max())


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
        "weighted sum would agree and this test would read nothing"
    )
    assert not torch.equal(weighted, want), (
        "a weighted collapse over these streams equals the mean, so the reading "
        "above cannot tell the two apart"
    )


def test_the_ffn_site_runs_over_ffn_halfs_unchanged_return() -> None:
    """``_ffn_half`` is called with one collapsed stream and its return is mixed back.
    """
    text_config = _text_config()
    model, stubs, _, input_ids = _stub_stack(text_config)
    ffn_half, seen, produced = _ffn_recorder()
    hidden = int(text_config.hidden_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        _run_carrier(model, input_ids)

    sites = [stub.ffn_site for stub in stubs]

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
        # Identity: the tensor the feed-forward half was handed is the tensor this
        # site collapsed, so no copy or re-collapse sits between them.
        assert call["tensor"] is site.given[0], (
            "the feed-forward half was handed a different tensor from the one the "
            "site collapsed"
        )
        assert sorted(call["kwargs"]) == [
            "block_size",
            "collector",
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


def test_the_hybrid_stack_runs_both_families_through_real_sites() -> None:
    """One KDA layer and one DSA layer, both bound, both composed, end to end. """
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
    distinct = [
        getattr(layer, impl.MHC_SITES_ATTR)[impl.MHC_ATTENTION_SITE]
        is not getattr(layer, impl.MHC_SITES_ATTR)[impl.MHC_FFN_SITE]
        for layer in model.layers
    ]

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


def test_a_stack_whose_layers_carry_no_mhc_weight_refuses_by_name() -> None:
    """The unconditional carrier's other half: no weights, no service. """
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
    assert "carries none of the six" in message, message
    assert "pass no streams" in message, message
    assert seen == [], (
        "the refusal must come before any feed-forward half runs, so a weightless "
        "stack cannot half-serve a request"
    )


def test_the_carrier_refuses_a_stream_count_that_is_not_positive() -> None:
    """``hc_mult`` 0 cannot build a carrier, and the message says which value it is. """
    text_config = _text_config(hc_mult=0)
    model, _, _, input_ids = _stub_stack(text_config)
    ffn_half, seen, _ = _ffn_recorder()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_half)
        with pytest.raises(ValueError) as caught:
            _run_carrier(model, input_ids)

    message = str(caught.value)
    assert "hc_mult=0" in message, message
    assert seen == []


ARM_TOKENS = 128


def _registered_pair() -> tuple[float, float]:
    """``(rtol, atol)`` read out of the file that declares them. """
    path = Path(__file__).with_name("test_mhc_layer.py")
    tree = ast.parse(path.read_text())
    found: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("RTOL", "ATOL") and isinstance(node.value, ast.Constant):
                found[name] = float(node.value.value)
    assert sorted(found) == ["ATOL", "RTOL"], (
        f"the registered pair could not be read from {path.name}; found {sorted(found)}"
    )
    return found["RTOL"], found["ATOL"]


# ---- the independent reference ------------------------------------------------
# Transcribed from the published reference modeling file, and from nothing in the
# implementation under test: a reference read off the code it checks proves only
# that the code agrees with itself.


def _ref_unweighted_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """the reference -- the mHC's own input norm, no learned gain."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def _ref_weighted_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """the reference -- the model's RMSNorm, computed in fp32, returned in dtype."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return gain * h.to(input_dtype)


def _ref_hc(streams, fn, base, scale, *, hc_mult, iters, hc_eps, rms_eps):
    """the reference -- ``(post, comb, collapsed)`` from the mHC mapping."""
    flat = _ref_unweighted_norm(streams.flatten(start_dim=1).float(), rms_eps)
    mixes = torch.nn.functional.linear(flat, fn.float())
    pre_w, post_w, comb_w = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_b, post_b, comb_b = base.split([hc_mult, hc_mult, hc_mult * hc_mult])
    pre_scale, post_scale, comb_scale = scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    # `the reference` -- the collapse, cast back to the streams' dtype.
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)
    return post, comb, collapsed


def _ref_mix(sub_out, residual, post, comb):
    """the reference (and the identical) -- the post-and-comb mix."""
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * sub_out.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def _ref_stack(embedded, layers, gain, *, hc_mult, iters, hc_eps, rms_eps,
               attention_half, ffn_half):
    """The whole composition, reference-side: expand, both sites per layer, mean, norm.
    """
    streams = embedded.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
    for index, (attn_w, ffn_w, input_gain) in enumerate(layers):
        post, comb, collapsed = _ref_hc(
            streams, *attn_w, hc_mult=hc_mult, iters=iters, hc_eps=hc_eps,
            rms_eps=rms_eps,
        )
        normed = _ref_weighted_norm(collapsed, input_gain, rms_eps)
        streams = _ref_mix(attention_half(normed, index), streams, post, comb)

        post, comb, collapsed = _ref_hc(
            streams, *ffn_w, hc_mult=hc_mult, iters=iters, hc_eps=hc_eps,
            rms_eps=rms_eps,
        )
        streams = _ref_mix(ffn_half(collapsed, index), streams, post, comb)
    collapsed = streams.mean(dim=1)
    return _ref_weighted_norm(collapsed, gain, rms_eps)


def _bound_stack(text_config, *, tokens: int = ARM_TOKENS, dtype=torch.float32,
                 seed: int = 401):
    """A real two-family stack: six loaded per layer, both sites bound, halves stubbed.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    impl = _impl()
    model = impl.Glm5NextModel(Glm5NextConfig(text_config=text_config), 1)
    hidden = int(text_config.hidden_size)
    gen = torch.Generator().manual_seed(seed)
    weights = []
    for index, layer in enumerate(model.layers):
        _norm_ready(layer, text_config)
        _load_the_six(layer, text_config, seed=seed + index)
        _install_stub(layer)
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
        by_site: dict[str, list] = {}
        for leaf in MHC_LEAVES:
            site, role = leaf.split("_")[1], leaf.split("_")[2]
            by_site.setdefault(site, {})[role] = getattr(layer, leaf).detach().clone()
        weights.append(
            (
                (by_site["attn"]["fn"], by_site["attn"]["base"], by_site["attn"]["scale"]),
                (by_site["ffn"]["fn"], by_site["ffn"]["base"], by_site["ffn"]["scale"]),
                layer.input_layernorm_weight.detach().clone(),
            )
        )
    table = (
        torch.randn(VOCAB, hidden, generator=gen, dtype=torch.float32) * 0.05
    ).to(dtype)
    model.embed_tokens_weight = nn.Parameter(table, requires_grad=False)
    model.norm_weight = nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)
    input_ids = torch.arange(tokens, dtype=torch.long) % VOCAB
    return model, table, input_ids, weights


def _stub_halves():
    """The two sublayer stand-ins, and the records of what each one saw. """
    attention_seen: list[tuple] = []
    ffn_seen: list[tuple] = []

    def attention_half(x, _index=None):
        attention_seen.append((tuple(x.shape), x.dtype))
        return torch.tanh(x) * 1.5

    def ffn_half(x, _index=None):
        ffn_seen.append((tuple(x.shape), x.dtype))
        return torch.tanh(x) * 0.75

    return attention_half, ffn_half, attention_seen, ffn_seen


def test_perturbing_one_bound_mhc_weight_moves_the_output() -> None:
    """arm (1): perturb one loaded mHC weight and the stack's output moves. """
    impl = _impl()
    text_config = _text_config()
    model, _, input_ids, _ = _bound_stack(text_config)
    _attention_half, _ffn_half, _, _ = _stub_halves()
    ffn_recorder, _, _ = _ffn_recorder()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_recorder)
        before = _run_carrier(model, input_ids,
                              carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])

        # The control first: an instance bound to no layer, same edit, same magnitude.
        unbound = impl.Glm5NextHyperConnection(text_config, neuron_config=None)
        with torch.no_grad():
            unbound.fn[0, 0] += 0.25
        control = _run_carrier(model, input_ids,
                              carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])

        # Then the arm: one weight the bind actually handed to a site.
        target = model.layers[0].hc_attn_fn
        with torch.no_grad():
            target[0, 0] += 0.25
        after = _run_carrier(model, input_ids,
                             carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])

    control_delta = float((control - before).abs().max())
    arm_delta = float((after - before).abs().max())

    assert control_delta == 0.0, (
        f"perturbing an UNBOUND instance moved the stack by {control_delta}; the "
        f"composition must read only the tensors the bind handed it"
    )
    assert arm_delta > 0.0, (
        "perturbing a bound mHC weight left the stack's output bit-identical; the "
        "bound tensors are not the tensors the composition reads"
    )


def test_the_hybrid_stack_composes_at_both_sites_at_128_tokens() -> None:
    """arm (2): a hybrid KDA+DSA stack exercises both sublayer sites at ``T = 128``. """
    impl = _impl()
    text_config = _text_config()
    model, table, input_ids, _ = _bound_stack(text_config)
    hidden = int(text_config.hidden_size)
    ffn_recorder, seen, _ = _ffn_recorder()
    stubs = [getattr(layer, type(layer).ATTENTION_ATTR) for layer in model.layers]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", ffn_recorder)
        out = _run_carrier(model, input_ids,
                           carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])

    families = [type(layer).__name__ for layer in model.layers]
    sites = [sorted(getattr(layer, impl.MHC_SITES_ATTR)) for layer in model.layers]

    assert len(input_ids) == ARM_TOKENS == 128
    assert len(set(families)) == 2, f"arm (2) wants both families, got {families}"
    want_sites = sorted([impl.MHC_ATTENTION_SITE, impl.MHC_FFN_SITE])
    assert all(pair == want_sites for pair in sites), f"{sites} != {want_sites}"
    assert [s.seen for s in stubs] == [[(ARM_TOKENS, hidden)]] * len(stubs)
    assert [c["shape"] for c in seen] == [(ARM_TOKENS, hidden)] * len(stubs)
    assert tuple(out.shape) == (ARM_TOKENS, hidden)
    assert out.dtype == table.dtype
    assert bool(torch.isfinite(out).all())


def test_the_stack_equals_the_independent_reference() -> None:
    """arm (3): the stack equals the reference within the registered pair, at ``T = 128``.
    """
    text_config = _text_config()
    rtol, atol = _registered_pair()
    model, table, input_ids, weights = _bound_stack(text_config)
    attention_half, ffn_half, _attention_seen, _ffn_seen = _stub_halves()

    def _ffn_as_model_calls_it(self, layer, hidden_states, **_kwargs):
        return ffn_half(hidden_states)

    # The attention half is installed as the stub module wrapping the shared
    # callable: assigning a bare function over a registered submodule is a
    # TypeError, and writing the formula twice is how the two sides drift.
    for layer in model.layers:
        _install_stub(layer, half=attention_half)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model), "_ffn_half", _ffn_as_model_calls_it)
        got = _run_carrier(model, input_ids,
                           carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])

    want = _ref_stack(
        table[input_ids],
        weights,
        model.norm_weight.detach(),
        hc_mult=int(text_config.hc_mult),
        iters=int(text_config.hc_sinkhorn_iters),
        hc_eps=float(text_config.hc_eps),
        rms_eps=float(text_config.rms_norm_eps),
        attention_half=lambda x, _i=None: attention_half(x),
        ffn_half=lambda x, _i=None: ffn_half(x),
    )

    max_abs = float((got - want).abs().max())
    max_rel = float(((got - want).abs() / (want.abs() + atol)).max())
    outside = int((~torch.isclose(got, want, rtol=rtol, atol=atol)).sum())

    assert float(want.abs().max()) > 0.0, (
        "the reference output is all zeros, so any tolerance would pass and this "
        "test would read nothing"
    )
    assert outside == 0, (
        f"{outside} of {got.numel()} elements fall outside the registered pair "
        f"(rtol={rtol}, atol={atol}); max_abs={max_abs:.6e} max_rel={max_rel:.6e}"
    )


def test_the_cast_points_are_the_references_own() -> None:
    """Every sublayer input, both mixes, the mean and the return carry the carrier's dtype.
    """
    text_config = _text_config()
    seen: dict[str, list] = {}
    for label, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model, table, input_ids, _ = _bound_stack(text_config, tokens=ARM_TOKENS,
                                                 dtype=dtype)
        ffn_recorder, ffn_calls, _ = _ffn_recorder()
        stubs = [getattr(layer, type(layer).ATTENTION_ATTR) for layer in model.layers]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(model), "_ffn_half", ffn_recorder)
            out = _run_carrier(model, input_ids,
                              carriers=[_call_kwargs("kda"), _call_kwargs("dsa")])
        attention_dtypes = [d for stub in stubs for d in stub.seen_dtypes]
        ffn_dtypes = [c["tensor"].dtype for c in ffn_calls]
        seen[label] = [table.dtype, out.dtype, ffn_dtypes, attention_dtypes]
        assert out.dtype == dtype, (
            f"a {label} carrier returned {out.dtype}; the norm returns the input "
            f"dtype (the reference) and the mean keeps it (the reference)"
        )
        assert attention_dtypes == [dtype] * len(model.layers), (
            f"the attention half was handed {attention_dtypes} from a {label} "
            f"carrier; the reference casts the collapse back to the streams' dtype"
        )
        assert ffn_dtypes == [dtype] * len(model.layers), (
            f"the feed-forward half was handed {ffn_dtypes} from a {label} carrier; "
            f"the reference casts the collapse back to the streams' dtype and the "
            f"dense seam's contract is bfloat16 activations (model_fp8.py)"
        )
    assert seen["fp32"][1] is torch.float32 and seen["bf16"][1] is torch.bfloat16


