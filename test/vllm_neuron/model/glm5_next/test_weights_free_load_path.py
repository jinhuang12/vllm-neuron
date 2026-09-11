# SPDX-License-Identifier: Apache-2.0
"""The hyper-connection bind on a weights-free tree, where the load targets meta.

A CPU-compile start never loads weights: the runner calls no loader and moves the
whole module to meta (``neuron_model_runner.py:1358-1367``). A bind for that route
therefore receives meta operands, and the items here read what it does with them:
the bind, the whole shape-only load through the real loader, and the two value
censuses that load walks into. The CPU arm is measured beside the meta one, because
the load path that ships today must bind exactly as it did.
"""

import dataclasses
import threading

import pytest
import torch
import torch.nn as nn

from .test_load_weights import (  # noqa: F401 -- the fixture is used by name
    _dense_config,
    _dense_model,
    _mappings_for,
    _stacked_checkpoint,
    _stacked_config,
    _stacked_model,
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


class _RecordingStore:
    """The two methods the pipelined load uses on its distributed store."""

    def __init__(self) -> None:
        self.added: list[str] = []

    def add(self, key: str, value: int) -> None:
        self.added.append(key)

    def check(self, keys) -> bool:
        return all(key in self.added for key in keys)


def _meta_load(tmp_path, name: str, *, with_bank: bool = False):
    """Run the real loader on meta through the reader, and hand both back.

    ``with_bank`` picks the landed routed fixture whose load COMPLETES rather than
    the all-dense one: one dense MLP at layer 0 and three expert banks written at
    256-blocked extents. It is not extended here, because the suite that owns it
    already carries both halves this file needs.
    """
    impl = _impl()
    directory = tmp_path / name
    model = _stacked_model() if with_bank else _dense_model()
    config = _stacked_config() if with_bank else _dense_config()
    mappings = _mappings_for(config)
    written = (
        _stacked_checkpoint(directory, mappings, model)
        if with_bank
        else _write_miniature_checkpoint(directory, mappings, model)
    )
    reader = impl._MetaShapeCheckpoint(str(directory), None)
    model.to(torch.device("meta"))
    model.load_weights(str(directory), torch.device("meta"), None, reader=reader)
    say("meta-load", f"tree={name}|written={written}|files={reader.get_num_files()}")
    return model, reader


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
    """The table carries what the real load leaves, and every mHC layer has its sites.

    TWO READINGS, AND NEITHER TYPES A RANK. The refusal this item is about is the
    root's own: it reads the table before any layer runs and raises when the table is
    ``None`` (``model_fp8.py:7448``), which is the condition asked here in the same
    form. What shape that table should carry is a different question, and the answer
    is the checkpoint's -- this fixture writes every plain-family key at a 1-D
    miniature placeholder, so a rank written here would measure the fixture. The
    reference is therefore the same loader run on the CPU, compared on shape AND
    dtype: an unset table, a reshaped one and a retyped one all fail against it.
    """
    impl = _impl()
    model = _lite_loaded(tmp_path)

    reference_directory = tmp_path / "cpu-reference"
    reference = _dense_model()
    _write_miniature_checkpoint(
        reference_directory, _mappings_for(_dense_config()), reference
    )
    reference.load_weights(str(reference_directory), torch.device("cpu"), None)
    reference_table = reference.model.embed_tokens_weight
    want = None if reference_table is None else (
        tuple(reference_table.shape),
        str(reference_table.dtype),
    )

    table = model.model.embed_tokens_weight
    got = None if table is None else (tuple(table.shape), str(table.dtype))
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
        f"table={got}|cpu_reference={want}",
        f"mhc_layers={len(counts)}|site_counts={sorted(set(counts))}",
    )
    assert table is not None, (
        "the embedding table is None after the hook, which is the exact condition the "
        "root forward raises on: it reads the table before any layer runs and refuses "
        "when nothing was loaded onto it"
    )
    assert got == want, (
        f"the hook left the embedding table at {got} where the same loader on the CPU "
        f"leaves {want} on the same checkpoint; a shape-only pass has to reproduce the "
        f"checkpoint's own header, in shape and in dtype"
    )
    assert counts and set(counts) == {2}, (
        f"the layers carrying mHC weights hold site counts {sorted(set(counts))}, "
        f"so a streams call on one of them has nothing to run"
    )


# --------------------------------------------------------------------------- #
# (7) The pipelined load completes: every file is announced and processed.      #
# --------------------------------------------------------------------------- #
def test_the_pipelined_load_completes_on_the_shape_only_reader(
    tmp_path, single_rank_process_group
) -> None:
    """The load's own loop waits on store keys, so the reader must still write them."""
    impl = _impl()
    directory = tmp_path / "announced"
    model = _dense_model()
    _write_miniature_checkpoint(directory, _mappings_for(_dense_config()), model)
    reader = impl._MetaShapeCheckpoint(str(directory), None)
    store = _RecordingStore()
    reader._load_to_page_cache(0, 1, store, threading.Event())
    files = list(reader._source.get_file_names())
    say("announced", f"added={store.added}|files={files}")
    assert store.added == files and store.check(files), (
        f"the reader announced {store.added} of {files}; the load's loop turns a "
        f"file on only when its key is in the store, so a file left out leaves "
        f"that loop spinning until the step's time bound cuts it off"
    )

    _model, loaded_reader = _meta_load(tmp_path, "pipelined")
    processed = len(loaded_reader._open_safetensor_files)
    say("pipelined", f"processed_files={processed}|files={loaded_reader.get_num_files()}")
    assert processed == loaded_reader.get_num_files() >= 1, (
        f"the load opened {processed} of {loaded_reader.get_num_files()} checkpoint "
        f"files, so it returned without processing them all"
    )


# --------------------------------------------------------------------------- #
# (8) Both censuses that read values are reached on the meta pass, and skip.     #
# --------------------------------------------------------------------------- #
def test_the_meta_pass_reaches_both_value_censuses_and_skips_them(
    tmp_path, single_rank_process_group
) -> None:
    """Reached, not merely guarded: the load walks into both and both record it.

    One tree carries both halves: the dense MLP at layer 0 reaches the block-scale
    compensation through its own scale-grid prep, and the three expert banks reach
    the retile through theirs. A pass on a tree that reaches neither would read
    green off code no load had entered.
    """
    from vllm_neuron.functional.moe import blockwise_fp8_retile
    from vllm_neuron.model.glm5_next import weight_loaders_fp8

    weight_loaders_fp8.SKIPPED_VALUE_CENSUSES.clear()
    blockwise_fp8_retile.SKIPPED_VALUE_CENSUSES.clear()
    _meta_load(tmp_path, "census", with_bank=True)
    compensated = list(weight_loaders_fp8.SKIPPED_VALUE_CENSUSES)
    retiled = list(blockwise_fp8_retile.SKIPPED_VALUE_CENSUSES)
    say("census-skips", f"compensate={len(compensated)}|retile={len(retiled)}")
    assert compensated.count("compensate_block_scales") >= 1, (
        f"the shape-only load recorded {sorted(set(compensated))} in the loader "
        f"module; the block-scale census reads three numbers before the platform "
        f"gate, so a pass that never reaches it proves nothing"
    )
    assert retiled.count("retile_block_scales") >= 1, (
        f"the shape-only load recorded {sorted(set(retiled))} in the retile "
        f"module; the retile allocates from values it reads, so a pass that never "
        f"reaches it leaves that branch unread"
    )
