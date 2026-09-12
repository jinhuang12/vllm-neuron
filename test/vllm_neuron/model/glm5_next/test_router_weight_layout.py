# SPDX-License-Identifier: Apache-2.0
"""The orientation the router weight is stored in, and the one its seam consumes.

The checkpoint stores ``mlp.gate.weight`` as ``[E, H]``, the orientation every
projection in this package is stored in. The router's consumer does not take that
orientation: it hands the tensor to an NKI kernel whose operand layout is
``[H, E]`` and validates it before either route. The items here read the map that
reaches the loader, the orientation the loader now produces, and the substrate's
own verdict on both orientations.

Every fixture here starts from the layout the CHECKPOINT holds and goes through the
loader. That is what closes the gap: the landed fixtures assign the parameter in the
consumed layout themselves -- ``test_router.py:236``, ``test_moe_path.py:2796`` and
``tiny/test_tiny_glm5next_forward.py:2371-2373``, which states it as design -- so no
landed item ever read what the loader produced, and the stored layout reached the
validator for the first time on hardware.
"""

import pytest
import torch

from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import (
    QuantizationType,
    RouterActFnType,
    _validate_inputs as substrate_validate,
)
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as loaders
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MAPPED_KEY_PLAIN,
    Glm5NextWeightMapError,
    SafetensorsWeightLoader,
    ShardGeometry,
    build_weight_mappings,
    classify_mapped_keys,
    loader_for_mapped_keys,
)

SENT = "ROUTERLAYOUT"

#: The leaf these items are about, spelled here so the file loads on a tree that
#: does not declare it yet -- which is what lets one control arm run this same file
#: against pre-repair product code. Item (a) is where the two are required to agree.
ROUTER_LEAF = "experts.router_weight"
DECLARED_LEAF = getattr(loaders, "TRANSPOSED_AT_LOAD_LEAF", None)

#: The checkpoint key the router weight is mapped from, as a leaf.
GATE_LEAF = "mlp.gate.weight"

#: A token extent. ``_validate_inputs`` reads it only through ``B * S`` and
#: constrains it nowhere, so it carries no acceptance value here.
TOKENS = 256


def say(*fields):
    """Print one machine-readable row, prefixed so a launcher can anchor on it."""
    print(f"{SENT}|" + "|".join(str(field) for field in fields))


class _WholeSlice:
    """The one thing a loader transform does to a slice: ``slice[:]``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, item):
        return self._tensor[item]


def published_map():
    """The map the published checkpoint's own settings build, with its config."""
    text_config = Glm5NextTextConfig()
    return text_config, build_weight_mappings(text_config)


def router_entries(mappings):
    """``{parameter: key}`` for every entry the transposing loader answers."""
    return {
        name: keys for name, keys in mappings.items() if name.endswith(ROUTER_LEAF)
    }


def a_sibling_plain_entry(mappings):
    """``(parameter, key)`` of one plain single-key entry that is NOT the router's.

    The kind is asked of the production classifier, so the control is the case
    the router's entry classifies as and differs from it only in the name.
    """
    for name, keys in sorted(mappings.items()):
        if name.endswith(ROUTER_LEAF):
            continue
        if isinstance(keys, str) and classify_mapped_keys(keys) == MAPPED_KEY_PLAIN:
            return name, keys
    raise AssertionError("the map holds no plain single-key entry outside the router")


def test_a_one_router_weight_per_routed_layer_reaches_the_transposing_loader() -> None:
    """The leaf the loader matches on is the leaf the real map writes."""
    text_config, mappings = published_map()
    routed_layers = text_config.num_hidden_layers - text_config.first_k_dense_replace
    entries = router_entries(mappings)
    say("map", f"entries={len(entries)}", f"routed_layers={routed_layers}",
        f"declared_leaf={DECLARED_LEAF}", f"this_files_leaf={ROUTER_LEAF}",
        f"all_single_key={all(isinstance(k, str) for k in entries.values())}",
        f"from_the_gate={all(str(k).endswith(GATE_LEAF) for k in entries.values())}")
    assert DECLARED_LEAF == ROUTER_LEAF, (
        f"the loader declares {DECLARED_LEAF!r} as the leaf it transposes and this "
        f"file is about {ROUTER_LEAF!r}; a rename on either side drops the transpose "
        f"in silence, because the loader is chosen by that name"
    )
    assert len(entries) == routed_layers, (
        f"the map holds {len(entries)} entries ending in {ROUTER_LEAF!r} for "
        f"{routed_layers} routed layers"
    )
    for name, keys in entries.items():
        assert isinstance(keys, str) and keys.endswith(GATE_LEAF), (
            f"{name} maps to {keys!r} rather than one {GATE_LEAF!r} key"
        )


def test_b_the_router_loads_transposed_and_a_sibling_plain_key_does_not() -> None:
    """The loaded tensor is the checkpoint's transpose; nothing else moves."""
    text_config, mappings = published_map()
    experts, hidden = text_config.n_routed_experts, text_config.hidden_size
    name, keys = sorted(router_entries(mappings).items())[0]

    written = torch.arange(experts * hidden, dtype=torch.float32).reshape(
        experts, hidden
    )
    loader = loader_for_mapped_keys(keys, param_name=name)
    assert loader is not None, f"{name} was left on the default loader"
    loaded = loader.load([_WholeSlice(written)], 0)

    # ``loaded[j, i]`` has to hold ``written[i, j]``, which is ``i * hidden + j``
    # for this fill. Built from index arithmetic rather than from a second
    # transpose, so the reading does not quote the operation it grades.
    expected = torch.arange(hidden, dtype=torch.float32).unsqueeze(1) + (
        torch.arange(experts, dtype=torch.float32).unsqueeze(0) * hidden
    )

    sibling_name, sibling_keys = a_sibling_plain_entry(mappings)
    sibling_loader = loader_for_mapped_keys(sibling_keys, param_name=sibling_name)
    untouched = SafetensorsWeightLoader().load([_WholeSlice(written)], 0)

    say("loaded", f"written={tuple(written.shape)}", f"loaded={tuple(loaded.shape)}",
        f"contiguous={loaded.is_contiguous()}",
        f"values_agree={torch.equal(loaded, expected)}")
    say("control", f"sibling={sibling_name}",
        f"sibling_loader={sibling_loader!r}", f"untouched={tuple(untouched.shape)}")

    assert tuple(loaded.shape) == (hidden, experts), (
        f"{name} loaded {tuple(loaded.shape)}; the seam consumes "
        f"[H={hidden}, E={experts}]"
    )
    assert loaded.is_contiguous(), "the parameter kept a transposed view"
    assert torch.equal(loaded, expected), (
        "the shape moved but the values did not, so the load reinterpreted the "
        "bytes instead of transposing them"
    )
    assert sibling_loader is None and tuple(untouched.shape) == (experts, hidden), (
        f"{sibling_name} is a plain replicated key and must keep the "
        f"checkpoint's orientation, which is this package's rule for every "
        f"family whose consumer is a linear"
    )


def test_c_what_the_loader_produces_is_what_the_router_call_takes() -> None:
    """One checkpoint-layout fixture, judged by the substrate through the loader.

    Both arms start from the tensor the checkpoint holds. One goes through the
    loader the map chooses and reaches the seam's validator; the other reaches it
    as the checkpoint stored it, which is what the landed fixtures cannot do
    because they assign the parameter themselves.
    """
    text_config, mappings = published_map()
    experts, hidden = text_config.n_routed_experts, text_config.hidden_size
    name, keys = sorted(router_entries(mappings).items())[0]
    meta = torch.device("meta")

    as_written = torch.empty((experts, hidden), device=meta, dtype=torch.bfloat16)
    # Whatever the map hands over, or the stored tensor when it hands over nothing.
    # The second case is the pre-repair tree, where this arm READS the refusal.
    loader = loader_for_mapped_keys(keys, param_name=name)
    as_loaded = loader.load([_WholeSlice(as_written)], 0) if loader else as_written
    hidden_states = torch.empty(1, TOKENS, hidden, device=meta, dtype=torch.bfloat16)
    gamma = torch.empty(1, hidden, device=meta, dtype=torch.bfloat16)

    def verdict(router_weights):
        try:
            substrate_validate(
                hidden_states,
                gamma,
                router_weights,
                None,
                text_config.num_experts_per_tok,
                None,
                QuantizationType.NONE,
                RouterActFnType.SIGMOID,
            )
        except AssertionError as refusal:
            return str(refusal)
        return ""

    wanted = f"router_weights must be [H={hidden}, E]"
    stored_verdict = verdict(as_written)
    loaded_verdict = verdict(as_loaded)
    say("verdict", f"written={tuple(as_written.shape)}",
        f"loaded={tuple(as_loaded.shape)}",
        f"stored_refused={bool(stored_verdict)}",
        f"carries_the_text={wanted in stored_verdict}",
        f"loaded_refused={bool(loaded_verdict)}",
        f"message={loaded_verdict or 'none'}")

    assert wanted in stored_verdict, (
        f"the checkpoint orientation {tuple(as_written.shape)} was expected to be "
        f"refused with {wanted!r}; read {stored_verdict!r}"
    )
    assert not loaded_verdict, (
        f"the loader's own output was refused by the seam it feeds: "
        f"{loaded_verdict!r}"
    )


def test_d_a_router_entry_the_loader_cannot_serve_is_refused_by_name() -> None:
    """A sharded or multi-key router entry refuses rather than loading unturned."""
    text_config, mappings = published_map()
    name, keys = sorted(router_entries(mappings).items())[0]
    cases = {
        "sharded": dict(
            geometry=ShardGeometry(0, text_config.n_routed_experts // 2, 2)
        ),
        "two_keys": dict(checkpoint_keys=[keys, keys.replace("gate", "gate2")]),
    }
    for label, override in cases.items():
        arguments = dict(checkpoint_keys=keys, param_name=name, geometry=None)
        arguments.update(override)
        with pytest.raises(Glm5NextWeightMapError) as refusal:
            loader_for_mapped_keys(
                arguments.pop("checkpoint_keys"), **arguments
            )
        message = str(refusal.value)
        say("refusal", f"case={label}", f"names_the_parameter={name in message}")
        assert name in message, (
            f"the {label} refusal does not name {name}, so a load failure could "
            f"not be attributed to this parameter"
        )
