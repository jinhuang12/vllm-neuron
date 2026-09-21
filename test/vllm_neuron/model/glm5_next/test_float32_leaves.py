# SPDX-License-Identifier: Apache-2.0
"""Leaves the checkpoint stores in float32 keep that width through the load.

The linear-attention decay and gate bias, the four hyper-connection mix leaves
and the router correction bias are float32 in the checkpoint and in the GPU
reference; a config-dtype placeholder would narrow them before any consumer
saw them."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import torch

from vllm_neuron.model.glm5_next import model_fp8 as _MODEL_FP8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig
from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

from .test_expert_bank_load import (
    _blocked_bank_overrides,
    _stacked_config,
    _stacked_model,
)
from .test_load_weights import (  # noqa: F401 -- fixtures are used by name
    FLOAT32_CHECKPOINT_LEAVES,
    FLOAT32_MIX_FAMILIES,
    MINI_PLAIN_SHAPE,
    MINI_ROUTED_EXPERTS,
    REAL_CONFIG_PATH,
    _keys_of,
    _mappings_for,
    _write_miniature_checkpoint,
    single_rank_process_group,
)


#: The reference deployment's world size; 64 heads shard one per rank.
KDA_STATE_READING_WORLD = 64


def test_the_kda_state_leaves_take_the_reference_dtype_and_per_rank_extent(
    single_rank_process_group,
) -> None:
    """``A_log`` shards one per head and ``dt_bias`` one per channel, both float32."""
    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    text_config = real_config.text_config
    linear = text_config.linear_attn_config
    heads = int(linear["num_heads"])
    head_dim = int(linear["head_dim"])
    assert heads % KDA_STATE_READING_WORLD == 0, (
        f"{heads} heads do not divide across {KDA_STATE_READING_WORLD} ranks, so "
        f"neither reference extent is a whole number here"
    )

    module = _MODEL_FP8.Glm5NextKDAAttention(text_config, KDA_STATE_READING_WORLD)
    expected = {
        "A_log": heads // KDA_STATE_READING_WORLD,
        "dt_bias": (heads * head_dim) // KDA_STATE_READING_WORLD,
    }
    for leaf, want_extent in sorted(expected.items()):
        geometry = _MODEL_FP8._shard_geometry_for(
            module, leaf, KDA_STATE_READING_WORLD
        )
        assert geometry is not None, (
            f"{leaf} is replicated at world size {KDA_STATE_READING_WORLD}; the "
            f"reference shards it on dim 0, so every rank above 0 would read "
            f"another rank's numbers"
        )
        dim = int(getattr(geometry, "shard_dim", -1))
        size = int(getattr(geometry, "shard_size", -1))
        assert (dim, size) == (0, want_extent), (
            f"{leaf} lands (dim {dim}, {size}) per rank where the reference gives "
            f"(dim 0, {want_extent}) at this degree"
        )

    model = Glm5NextForConditionalGeneration(real_config)
    mappings = _mappings_for(real_config)
    typed: dict[str, torch.dtype] = {}
    for leaf in sorted(FLOAT32_CHECKPOINT_LEAVES):
        name = next(
            candidate
            for candidate in sorted(mappings)
            if candidate.rsplit(".", 1)[-1] == leaf
        )
        typed[leaf] = model._placeholder_dtype(
            mappings[name], param_name=name, mappings=mappings
        )
    assert set(typed.values()) == {torch.float32}, (
        f"the state leaves take {sorted(str(d) for d in typed.values())}; the "
        f"reference keeps both float32 and the checkpoint holds both float32, so "
        f"anything narrower is a cast on every rank"
    )

    control_name = next(
        candidate
        for candidate in sorted(mappings)
        if candidate.endswith(".o_norm_weight")
    )
    control_dtype = model._placeholder_dtype(
        mappings[control_name], param_name=control_name, mappings=mappings
    )
    assert control_dtype is text_config.torch_dtype, (
        f"{control_name} takes {control_dtype} where the config declares "
        f"{text_config.torch_dtype}, so the float32 rule above is not keyed on the "
        f"state pair at all"
    )


def test_the_float32_mix_families_take_the_checkpoint_dtype(
    single_rank_process_group,
) -> None:
    """The four mHC leaves and the router bias take a float32 placeholder on the real config."""
    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    model = Glm5NextForConditionalGeneration(real_config)
    mappings = _mappings_for(real_config)

    typed: dict[str, torch.dtype] = {}
    for parameter_leaf, _ in FLOAT32_MIX_FAMILIES:
        names = [
            name
            for name in sorted(mappings)
            if name.rsplit(".", 1)[-1] == parameter_leaf
        ]
        assert names, f"no mapped parameter has the leaf {parameter_leaf}"
        typed[parameter_leaf] = model._placeholder_dtype(
            mappings[names[0]], param_name=names[0], mappings=mappings
        )

    assert sorted(typed) == sorted(leaf for leaf, _ in FLOAT32_MIX_FAMILIES), (
        f"the reading covered {sorted(typed)} where the declared families are "
        f"{sorted(leaf for leaf, _ in FLOAT32_MIX_FAMILIES)}"
    )
    assert set(typed.values()) == {torch.float32}, (
        f"the mix families take {sorted(str(dtype) for dtype in typed.values())}; "
        f"the checkpoint holds all five in float32 and the reference declares all "
        f"five float32, so anything narrower is a cast on every rank"
    )


def test_a_miniature_load_publishes_the_float32_mix_families_unchanged(
    tmp_path, single_rank_process_group
) -> None:
    """A real load hands the five families through value for value, with no cast."""
    directory = tmp_path / "mixfamilies"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())

    covered: dict[str, str] = {}
    for parameter_leaf, checkpoint_leaf in FLOAT32_MIX_FAMILIES:
        for name in sorted(mappings):
            if name.rsplit(".", 1)[-1] != parameter_leaf:
                continue
            keys = _keys_of(mappings, name)
            assert len(keys) == 1, (
                f"{name} maps to {len(keys)} checkpoint keys; these five are plain "
                f"one-key entries and a fused entry would need a different reading"
            )
            assert keys[0].rsplit(".", 1)[-1] == checkpoint_leaf, (
                f"{name} maps to {keys[0]}, whose leaf is not the declared "
                f"checkpoint name {checkpoint_leaf}, so the writers below would "
                f"type this family the way the code expects it and not the way "
                f"the checkpoint holds it"
            )
            covered[name] = keys[0]

    assert {name.rsplit(".", 1)[-1] for name in covered} == {
        leaf for leaf, _ in FLOAT32_MIX_FAMILIES
    }, (
        f"the routed fixture carries "
        f"{sorted({name.rsplit('.', 1)[-1] for name in covered})}, so at least one "
        f"declared family would go unread by this load"
    )

    # Values a bfloat16 round trip would change, so a narrowing cast shows.
    witness: dict[str, torch.Tensor] = {}
    for index, name in enumerate(sorted(covered), start=1):
        ramp = torch.arange(
            1, MINI_PLAIN_SHAPE[0] + 1, dtype=torch.float32
        ) * 2.0**-12
        value = (float(index) + ramp).reshape(MINI_PLAIN_SHAPE)
        assert not torch.equal(value, value.to(torch.bfloat16).to(torch.float32)), (
            f"the witness values for {name} survive a bfloat16 round trip"
        )
        witness[covered[name]] = value

    _write_miniature_checkpoint(
        directory,
        mappings,
        model,
        extra_overrides={**_blocked_bank_overrides(model, mappings), **witness},
    )
    model.load_weights(str(directory), torch.device("cpu"), None)

    published = dict(model.named_parameters())
    for name in sorted(covered):
        assert name in published, (
            f"{name} is mapped but absent from the loaded model's parameters"
        )
        got = published[name].detach()
        expected = witness[covered[name]]
        assert got.dtype is torch.float32, (
            f"{name} published {got.dtype} where the checkpoint holds float32"
        )
        assert torch.equal(got, expected), (
            f"{name} published {got.flatten()[:4].tolist()} where the checkpoint "
            f"holds {expected.flatten()[:4].tolist()}, so the load moved the "
            f"values it was meant to hand through"
        )


def test_the_router_projection_still_takes_the_config_dtype(
    single_rank_process_group,
) -> None:
    """The router weight beside the bias keeps the config dtype."""
    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    model = Glm5NextForConditionalGeneration(real_config)
    mappings = _mappings_for(real_config)

    control_name = next(
        name for name in sorted(mappings) if name.endswith(".router_weight")
    )
    control_dtype = model._placeholder_dtype(
        mappings[control_name], param_name=control_name, mappings=mappings
    )
    assert control_dtype is real_config.text_config.torch_dtype, (
        f"{control_name} takes {control_dtype} where the config declares "
        f"{real_config.text_config.torch_dtype}, so the float32 rule is not keyed "
        f"on the five names at all"
    )


def test_without_the_declared_leaves_the_mix_families_narrow_to_the_config_dtype(
    monkeypatch, single_rank_process_group
) -> None:
    """Emptying ``FLOAT32_PLAIN_LEAVES`` makes all five families take the config dtype."""
    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    model = Glm5NextForConditionalGeneration(real_config)
    mappings = _mappings_for(real_config)
    config_dtype = real_config.text_config.torch_dtype

    monkeypatch.setattr(_MODEL_FP8, "FLOAT32_PLAIN_LEAVES", ())
    for parameter_leaf, _ in FLOAT32_MIX_FAMILIES:
        name = next(
            candidate
            for candidate in sorted(mappings)
            if candidate.rsplit(".", 1)[-1] == parameter_leaf
        )
        narrowed = model._placeholder_dtype(
            mappings[name], param_name=name, mappings=mappings
        )
        assert narrowed is config_dtype, (
            f"{parameter_leaf} still takes {narrowed} with the declared tuple "
            f"emptied, so something other than that tuple is typing it"
        )


def test_the_router_correction_bias_reaches_the_seam_in_float32() -> None:
    """The router hands ``router_bias`` to the seam, which keeps float32 corrections distinct where bfloat16 ties them."""
    from vllm_neuron.functional.moe.router import (
        _legalize_correction_bias,
        noaux_tc_rmsnorm_router_topk,
    )

    call = next(
        node
        for node in ast.walk(
            ast.parse(
                textwrap.dedent(
                    inspect.getsource(_MODEL_FP8.Glm5NextRoutedExperts.route_tokens)
                )
            )
        )
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "noaux_tc_rmsnorm_router_topk"
    )
    handed = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg == "correction_bias"
    }
    assert handed == {"correction_bias": "self.router_bias"}, (
        f"the router enters the seam with {handed}, so router_bias is not the "
        f"tensor it corrects with"
    )
    assert "_legalize_correction_bias" in inspect.getsource(
        noaux_tc_rmsnorm_router_topk
    ), "the seam no longer legalises its correction bias"

    experts = MINI_ROUTED_EXPERTS
    bias = torch.tensor(
        [1.0 + step * 2.0**-12 for step in range(1, experts + 1)],
        dtype=torch.float32,
    )
    legalised = _legalize_correction_bias(bias, experts)
    narrowed = _legalize_correction_bias(bias.to(torch.bfloat16), experts)
    distinct = len(set(legalised.flatten().tolist()))
    distinct_narrowed = len(set(narrowed.flatten().tolist()))
    assert legalised.dtype is torch.float32 and narrowed.dtype is torch.float32, (
        f"the seam returned {legalised.dtype} and {narrowed.dtype}; it is declared "
        f"to return fp32 for either input"
    )
    assert distinct == experts, (
        f"{distinct} of {experts} corrections stay distinct through the seam from a "
        f"float32 parameter, so this fixture cannot tell the two widths apart"
    )
    assert distinct_narrowed < experts, (
        f"all {experts} corrections stay distinct through a bfloat16 parameter, so "
        f"the load-time width would not decide the router's choice here"
    )
