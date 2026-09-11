# SPDX-License-Identifier: Apache-2.0
"""The hyper-connection bind on a weights-free tree, where the load targets meta.

A CPU-compile start never loads weights: the runner calls no loader and moves the
whole module to meta (``neuron_model_runner.py:1358-1367``). A bind for that route
therefore receives meta operands, and the four items here read what it does with
them. The CPU arm is measured beside the meta one, because the load path that ships
today must bind exactly as it did.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from .test_load_weights import (  # noqa: F401 -- the fixture is used by name
    _dense_config,
    _dense_model,
    _mappings_for,
    _write_miniature_checkpoint,
    single_rank_process_group,
)

SENT = "WFLOAD"


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config(**overrides):
    """A two-layer hybrid config at the checkpoint's own widths, all dense."""
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
    """The shape each of the six leaves carries, derived from the config."""
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    hc_mult = int(text_config.hc_mult)
    mix = (2 + hc_mult) * hc_mult
    by_role = {
        "fn": (mix, hc_mult * int(text_config.hidden_size)),
        "base": (mix,),
        "scale": (3,),
    }
    return {leaf: by_role[leaf.split("_")[2]] for leaf in MHC_LEAVES}


def _load_the_six(layer, text_config, *, seed: int = 120):
    """Put a real tensor on each of the six leaves, the way a load leaves them."""
    gen = torch.Generator().manual_seed(seed)
    placed: dict[str, torch.Tensor] = {}
    for leaf, shape in sorted(_leaf_shapes(text_config).items()):
        tensor = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
        setattr(layer, leaf, nn.Parameter(tensor, requires_grad=False))
        placed[leaf] = tensor
    return placed


def _loaded_layer(text_config, *, on_meta: bool):
    """One KDA layer with its six mHC leaves loaded, on meta or on the CPU."""
    layer = _impl().Glm5NextKDALayer(text_config, 0, 1)
    _load_the_six(layer, text_config)
    if on_meta:
        layer.to("meta")
    return layer


# --------------------------------------------------------------------------- #
# (1) The bind completes when the load targets meta.                           #
# --------------------------------------------------------------------------- #
def test_the_bind_completes_when_the_load_targets_meta() -> None:
    """Six meta leaves bind to two sites, and the count comes from the bind."""
    impl = _impl()
    text_config = _text_config()
    layer = _loaded_layer(text_config, on_meta=True)
    devices = {str(getattr(layer, leaf).device) for leaf in sorted(_leaf_shapes(text_config))}
    say("meta-operands", f"devices={sorted(devices)}")
    assert devices == {"meta"}, (
        f"the fixture did not put the six leaves on meta, so this item would "
        f"measure the CPU bind a second time: {sorted(devices)}"
    )

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("meta"))
    say("meta-bind", f"sites={bound_sites}")
    assert bound_sites == 2, (
        f"a load that targets meta bound {bound_sites} sites, so a weights-free "
        f"start has no site to run and the mix has nothing to read"
    )


# --------------------------------------------------------------------------- #
# (2) The site the bind builds sits on the device the load named.              #
# --------------------------------------------------------------------------- #
def test_the_site_sits_on_the_device_the_load_named() -> None:
    """Every parameter of a meta bind's site is on meta, and the record says so."""
    impl = _impl()
    text_config = _text_config()
    layer = _loaded_layer(text_config, on_meta=True)
    layer.bind_hyper_connection_sites(text_config, torch.device("meta"))

    sites = getattr(layer, impl.MHC_SITES_ATTR)
    places = {
        str(parameter.device)
        for site in sites.values()
        for parameter in site.parameters()
    }
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)
    say("site-devices", f"places={sorted(places)}|recorded={health['device']}")
    assert places == {"meta"}, (
        f"a site built for a meta load holds parameters on {sorted(places)}; the "
        f"sites are held in a plain dict that no later move visits, so anything "
        f"left on another device is stranded there"
    )
    assert health["device"] == "meta", (
        f"the bind recorded {health['device']!r} for a load that named meta"
    )


# --------------------------------------------------------------------------- #
# (3) The CPU load path is unmoved: same count, same device, same storage.     #
# --------------------------------------------------------------------------- #
def test_a_cpu_load_binds_the_loaded_storage_as_before() -> None:
    """A CPU bind still binds two sites and hands over the loaded storage itself."""
    impl = _impl()
    text_config = _text_config()
    layer = _impl().Glm5NextKDALayer(text_config, 0, 1)
    placed = _load_the_six(layer, text_config)

    bound_sites = layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    health = getattr(layer, impl.MHC_BIND_HEALTH_ATTR)
    sites = getattr(layer, impl.MHC_SITES_ATTR)
    shared = 0
    for site_name, site in sorted(sites.items()):
        for leaf, entry in sorted(health["sites"][site_name].items()):
            parameter = getattr(site, entry["parameter"])
            if parameter.data_ptr() == placed[leaf].data_ptr():
                shared += 1
    say("cpu-bind", f"sites={bound_sites}|device={health['device']}|shared={shared}")
    assert (bound_sites, health["device"], shared) == (2, "cpu", 6), (
        f"the CPU bind moved: {bound_sites} sites on {health['device']!r} with "
        f"{shared} of six parameters sharing the loaded storage, want 2, 'cpu', 6"
    )


# --------------------------------------------------------------------------- #
# (4) The control: without a bind, a streams call refuses by class.            #
# --------------------------------------------------------------------------- #
def test_a_loaded_but_unbound_layer_refuses_a_streams_call() -> None:
    """The failure a weights-free start hits today, raised by its own class."""
    impl = _impl()
    text_config = _text_config()
    layer = _impl().Glm5NextKDALayer(text_config, 0, 1)
    _load_the_six(layer, text_config)

    streams = torch.zeros(1, int(text_config.hc_mult), int(text_config.hidden_size))
    with pytest.raises(impl.Glm5NextHyperConnectionError) as raised:
        impl._mhc_site(layer, streams, impl.MHC_ATTENTION_SITE)
    say("unbound-refusal", str(raised.value)[:120])

    layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
    site = impl._mhc_site(layer, streams, impl.MHC_ATTENTION_SITE)
    say("unbound-control", f"after_bind={type(site).__name__}")
    assert site is not None, (
        "the same call returns no site after a bind, so the refusal above says "
        "nothing about the bind being what is missing"
    )


def _lite_loaded(tmp_path):
    """A dense miniature tree after the weights-free hook, with its checkpoint."""
    directory = tmp_path / "weights-free"
    model = _dense_model()
    written = _write_miniature_checkpoint(directory, _mappings_for(_dense_config()), model)
    say("fixture", f"tensors_written={written}|declared={len(model.declared_parameter_names())}")
    model.load_weights_lite(str(directory), torch.device("cpu"), None)
    return model


# --------------------------------------------------------------------------- #
# (5) The hook leaves every declared leaf a meta tensor of its own shape.       #
# --------------------------------------------------------------------------- #
def test_the_hook_materialises_every_declared_leaf_on_meta(
    tmp_path, single_rank_process_group
) -> None:
    """No leaf is left None or shape-free, and every one of them is on meta."""
    model = _lite_loaded(tmp_path)

    declared = model.declared_parameter_names()
    by_name = dict(model.named_parameters())
    none_count = sum(1 for name in declared if by_name.get(name) is None)
    lazy_count = sum(
        1
        for parameter in by_name.values()
        if torch.nn.parameter.is_lazy(parameter)
    )
    off_meta = sorted(
        name for name, parameter in by_name.items() if parameter.device.type != "meta"
    )
    shapeless = sorted(
        name
        for name, parameter in by_name.items()
        if not torch.nn.parameter.is_lazy(parameter) and parameter.dim() == 0
    )
    say(
        "materialised",
        f"declared={len(declared)}|named={len(by_name)}|none={none_count}",
        f"lazy={lazy_count}|off_meta={len(off_meta)}|shapeless={len(shapeless)}",
    )
    assert (none_count, lazy_count, off_meta, shapeless) == (0, 0, [], []), (
        f"the hook left {none_count} leaves unset, {lazy_count} shape-free, "
        f"{len(off_meta)} off meta and {len(shapeless)} without a shape; a load "
        f"that leaves any of those cannot reach a forward"
    )
    assert len(by_name) == len(declared), (
        f"the hook materialised {len(by_name)} of {len(declared)} declared "
        f"parameters, so the walk visited a different set than the declaration"
    )


# --------------------------------------------------------------------------- #
# (6) Both refusals that stop a weights-free forward are cleared.               #
# --------------------------------------------------------------------------- #
def test_the_hook_clears_both_refusals_the_forward_hits(
    tmp_path, single_rank_process_group
) -> None:
    """The embedding table is set, and every mHC layer has its two sites."""
    impl = _impl()
    model = _lite_loaded(tmp_path)

    table = model.model.embed_tokens_weight
    layers = [
        layer
        for layer in model.model.modules()
        if hasattr(type(layer), "bind_hyper_connection_sites")
    ]
    counts = sorted(
        len(getattr(layer, impl.MHC_SITES_ATTR, {}))
        for layer in layers
        if any(getattr(layer, leaf, None) is not None for leaf in impl.MHC_LEAVES)
    )
    say(
        "forward-preconditions",
        f"table={None if table is None else tuple(table.shape)}",
        f"mhc_layers={len(counts)}|site_counts={sorted(set(counts))}",
    )
    assert table is not None and table.dim() == 2, (
        "the embedding table is still unset after the hook, so the root forward "
        "refuses before any layer runs"
    )
    assert counts and set(counts) == {2}, (
        f"the layers carrying mHC weights hold site counts {sorted(set(counts))}, "
        f"so a streams call on one of them has nothing to run"
    )
