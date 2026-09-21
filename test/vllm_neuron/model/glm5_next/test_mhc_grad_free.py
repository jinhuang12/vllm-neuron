# SPDX-License-Identifier: Apache-2.0
"""The hyper-connection site's three parameters are grad-free, bound or not."""

import dataclasses

import torch
from torch import nn


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config():
    """A two-layer hybrid config, one layer of each attention family, both dense. """
    from vllm_neuron.model.glm5_next.config import (
        DSA_LAYER_TYPE,
        KDA_LAYER_TYPE,
        Glm5NextTextConfig,
    )

    return dataclasses.replace(
        Glm5NextTextConfig(),
        num_hidden_layers=2,
        layer_types=[KDA_LAYER_TYPE, DSA_LAYER_TYPE],
        first_k_dense_replace=2,
    )


def _leaf_shapes(text_config):
    """The shape each of the six leaves has, derived from the config."""
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    hc_mult = int(text_config.hc_mult)
    hidden = int(text_config.hidden_size)
    mix = (2 + hc_mult) * hc_mult
    by_role = {"fn": (mix, hc_mult * hidden), "base": (mix,), "scale": (3,)}
    return {leaf: by_role[leaf.split("_")[2]] for leaf in MHC_LEAVES}


def _load_the_six(layer, text_config, *, seed: int = 126):
    """Put a real grad-free tensor on each of the six leaves, as a load leaves them."""
    gen = torch.Generator().manual_seed(seed)
    for leaf, shape in sorted(_leaf_shapes(text_config).items()):
        tensor = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
        setattr(layer, leaf, nn.Parameter(tensor, requires_grad=False))


def _layer_of(family, text_config):
    impl = _impl()
    if family == "kda":
        return impl.Glm5NextKDALayer(text_config, 0, 1)
    return impl.Glm5NextDSALayer(text_config, 1, 1)


def test_a_fresh_site_holds_three_grad_free_parameters() -> None:
    """The unbound path: ``nn.Parameter`` defaults to tracking, and this class must not.
    """
    impl = _impl()
    text_config = _text_config()
    instance = impl.Glm5NextHyperConnection(
        text_config, neuron_config=text_config.neuron_config
    )

    names = sorted(set(impl.MHC_ROLE_PARAMETERS.values()))
    flags = {
        name: bool(getattr(instance, name).requires_grad) for name in names
    }
    tracked = sorted(name for name, flag in flags.items() if flag)

    assert flags, "this test read no parameters at all, so it measured nothing"
    assert not tracked, (
        f"a fresh site asks autograd to track {tracked}; nothing here is trained, "
        f"and a tracked parameter puts a grad_fn on every tensor computed from it"
    )


def test_a_bound_leaf_and_its_site_parameter_agree_on_requires_grad() -> None:
    """One storage, one flag -- read on both families, twelve pairs in all. """
    impl = _impl()
    text_config = _text_config()
    grouped = impl._mhc_leaves_by_site()

    pairs = 0
    differed = []
    unshared = []
    for family in ("kda", "dsa"):
        layer = _layer_of(family, text_config)
        _load_the_six(layer, text_config)
        layer.bind_hyper_connection_sites(text_config, torch.device("cpu"))
        sites = getattr(layer, impl.MHC_SITES_ATTR)
        for site, roles in sorted(grouped.items()):
            for role, leaf in sorted(roles.items()):
                parameter = getattr(sites[site], impl.MHC_ROLE_PARAMETERS[role])
                loaded = getattr(layer, leaf)
                shared = parameter.data_ptr() == loaded.data_ptr()
                site_flag = bool(parameter.requires_grad)
                leaf_flag = bool(loaded.requires_grad)
                pairs += 1
                if not shared:
                    unshared.append((family, leaf))
                if site_flag != leaf_flag:
                    differed.append((family, leaf, site_flag, leaf_flag))


    assert pairs == 12, (
        f"this test read {pairs} bound pairs; both families hold six each"
    )
    assert not unshared, (
        f"{unshared} reached a site as a COPY, so their flags would agree for a reason "
        f"this test is not about"
    )
    assert not differed, (
        f"one storage carries two flags after the bind: {differed} as "
        f"(family, leaf, site_flag, leaf_flag); graph extraction refuses exactly this"
    )
