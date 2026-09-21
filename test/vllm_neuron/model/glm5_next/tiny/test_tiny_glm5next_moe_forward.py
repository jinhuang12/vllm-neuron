"""Forward-pass tests for the tiny GLM-5.3-Flash MoE layer.

The routed expert bank, the always-on shared expert, and the MoE block that
routes, runs the experts and adds the shared expert once. Fixtures and torch
references come from ``test_tiny_glm5next_forward``.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.tiny.test_tiny_glm5next_forward import (
    ATOL,
    DENSE_BLOCK,
    HIDDEN_SCALE_EXPONENT,
    HIDDEN_SIZE,
    MAX_PARAMETERS,
    MOE_ATOL,
    MOE_ROUTER_BIAS_SCALE,
    MOE_ROUTER_WEIGHT_SCALE,
    MOE_RTOL,
    NUM_KEY_VALUE_HEADS,
    ROUTED_EXPERTS,
    ROUTED_EXPERTS_PER_TOKEN,
    ROUTED_HIDDEN_SIZE,
    ROUTED_INTERMEDIATE_SIZE,
    ROUTED_MAX_BLOCK_SHARE,
    ROUTED_MAX_CONDITION,
    RTOL,
    SEED_MOE_HIDDEN,
    SEED_MOE_ROUTER,
    TOKENS,
    ReferenceShapeError,
    VacuousControlError,
    _POST_SCALE,
    _PRE_SCALE,
    _assert_route_predicate,
    _attach,
    _dense_operands,
    _dense_output,
    _ffn_gamma,
    _ffn_norm,
    _fp8_grid_values,
    _impl,
    _pinned_raw_config,
    _prep_operands_from_the_module,
    _quant_config,
    _read_seam_counters,
    _reset_seam_counters,
    _routed_operands,
    _routed_output,
    _routed_text_config,
    _scale_grid_attribute,
    _shared_at_routed_operands,
    _tiny_text_config,
)
from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

pytestmark = [pytest.mark.fast]

# Channel pairs to swap when looking for a perturbation that changes the
# router's selected expert set; spread over both halves of the hidden axis.
MOE_SWAP_CHANNEL_CANDIDATES = (
    (0, 1),
    (0, 256),
    (1, 257),
    (7, 263),
    (13, 400),
    (64, 320),
    (128, 384),
    (0, 511),
)

MOE_SWAP_TOKEN_SCAN = 16


def test_tiny_routed_experts_forward_matches_the_reference() -> None:
    """The routed bank on one MoE layer, against the checkpoint's POST_SCALE reference."""
    model_fp8 = _impl()
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import ExpertAffinityScaleMode

    text_config = _routed_text_config()
    module = model_fp8.Glm5NextRoutedExperts(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the bank resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )
    for label in (_POST_SCALE, _PRE_SCALE):
        if getattr(ExpertAffinityScaleMode, label).name != label:
            raise VacuousControlError(
                f"this file reasons about a scaling point it calls {label!r} and the "
                f"kernel's enum spells that member "
                f"{getattr(ExpertAffinityScaleMode, label).name!r}"
            )

    operands = _routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny routed bank holds {parameters} parameters, at or above the "
            f"tiny-config bound of {MAX_PARAMETERS}"
        )

    built = module.prepare_scale_operands(
        *_prep_operands_from_the_module(module, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), operands)
    )
    health = getattr(module, module.RETILE_HEALTH_ATTR)
    if built != 2:
        raise VacuousControlError(
            f"prepare_scale_operands built {built} operands and the bank's forward "
            f"looks up 2"
        )
    inexact = {bank: counts[2] for bank, counts in health.items() if counts[2]}
    if inexact:
        raise VacuousControlError(
            f"the publisher reports inexact rescales {inexact}; it emits the "
            f"checkpoint's own grid and rescales nothing, so a nonzero count means a "
            f"remapping has come back onto the load path"
        )

    reference = _routed_output(
        operands, mode=_POST_SCALE, gate_max=limit, gate_min=None,
        up_max=limit, up_min=-limit,
    )

    gate, up = reference["gate"], reference["up"]
    counts = {
        "gate above the bound": int((gate > limit).sum()),
        "gate inside the bound": int(((gate >= -limit) & (gate <= limit)).sum()),
        "gate below the negated bound": int((gate < -limit).sum()),
        "up above the bound": int((up > limit).sum()),
        "up inside the bound": int(((up >= -limit) & (up <= limit)).sum()),
        "up below the negated bound": int((up < -limit).sum()),
    }
    for name, count in counts.items():
        if count == 0:
            raise VacuousControlError(
                f"no element has {name}, so the test cannot tell the reference's "
                f"clamp from its absence"
            )

    if reference["condition"] > ROUTED_MAX_CONDITION:
        raise VacuousControlError(
            f"the block contributions sum to {reference['condition']:.2f} times the "
            f"output they produce, above the bound of "
            f"{ROUTED_MAX_CONDITION}: the fixture has drifted into near-cancellation "
            f"and every control gap it reports is inflated"
        )
    if reference["share"] > ROUTED_MAX_BLOCK_SHARE:
        raise VacuousControlError(
            f"one 256-block carries {reference['share'] * 100:.0f}% of the output, "
            f"above the bound of {ROUTED_MAX_BLOCK_SHARE * 100:.0f}%"
        )

    variants = {
        "gate upper clamp removed": dict(
            mode=_POST_SCALE, gate_max=None, gate_min=None, up_max=limit, up_min=-limit),
        "up upper clamp removed": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=None, up_max=None, up_min=-limit),
        "up lower clamp removed": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=None, up_max=limit, up_min=None),
        "gate lower clamp wrongly added": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=-limit, up_max=limit,
            up_min=-limit),
        "scaling point moved to PRE_SCALE": dict(
            mode=_PRE_SCALE, gate_max=limit, gate_min=None, up_max=limit, up_min=-limit),
    }
    for name, keywords in variants.items():
        variant = _routed_output(operands, **keywords)
        moved = not torch.allclose(
            variant["out"], reference["out"], rtol=MOE_RTOL, atol=MOE_ATOL
        )
        if not moved:
            raise VacuousControlError(
                f"changing the reference so that its {name} leaves the result inside "
                f"rtol={MOE_RTOL}, atol={MOE_ATOL}; the test would pass with that "
                f"branch wrong in the module"
            )

    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(
        operands["hidden"],
        operands["expert_affinities"],
        _quant_config(),
    )
    after = _read_seam_counters()
    _assert_route_predicate(
        "routed experts", {"moe_fused": 1},
        before, after,
    )

    if tuple(got.shape) != (TOKENS, ROUTED_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, ROUTED_HIDDEN_SIZE)}; the padding-token row is the callee's "
            f"to slice off"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=MOE_RTOL, atol=MOE_ATOL
    )


def test_tiny_shared_experts_forward_matches_the_reference() -> None:
    """The always-on shared expert on one MoE layer, against the dense reference."""
    model_fp8 = _impl()
    text_config = _tiny_text_config()
    module = model_fp8.Glm5NextSharedExperts(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the module resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )

    operands = _dense_operands()
    leaves = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    for leaf in leaves:
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny shared expert holds {parameters} parameters, at or above "
            f"the tiny-config bound of {MAX_PARAMETERS}"
        )

    with pytest.raises(model_fp8.Glm5NextSharedExpertRouteError):
        module.forward(operands["hidden"], quant_config=_quant_config())

    built = module.prepare_scale_operands(
        *(getattr(module, leaf) for leaf in leaves),
        *(getattr(module, _scale_grid_attribute(leaf)) for leaf in leaves),
    )
    if built != 3:
        raise VacuousControlError(
            f"the load-time prep reported {built} operands, not the 3 the shared "
            f"expert's three projections need"
        )

    reference = _dense_output(operands, limit, -limit, limit)

    above_gate = int((reference["gate"] > limit).sum())
    below_gate = int((reference["gate"] <= limit).sum())
    above_up = int((reference["up"] > limit).sum())
    within_up = int(((reference["up"] >= -limit) & (reference["up"] <= limit)).sum())
    below_up = int((reference["up"] < -limit).sum())
    for name, count in (
        ("gate above the bound", above_gate),
        ("gate at or below the bound", below_gate),
        ("up above the bound", above_up),
        ("up inside the bound", within_up),
        ("up below the negated bound", below_up),
    ):
        if count == 0:
            raise VacuousControlError(
                f"no element has {name}, so the test cannot tell the reference's "
                f"clamp from its absence"
            )

    for name, variant in (
        ("gate upper clamp", _dense_output(operands, None, -limit, limit)),
        ("up upper clamp", _dense_output(operands, limit, -limit, None)),
        ("up lower clamp", _dense_output(operands, limit, None, limit)),
    ):
        moved = not torch.allclose(
            variant["out"], reference["out"], rtol=RTOL, atol=ATOL
        )
        if not moved:
            raise VacuousControlError(
                f"removing the {name} leaves the result inside rtol={RTOL}, "
                f"atol={ATOL}; the test would pass with that branch deleted from "
                f"the module"
            )

    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(operands["hidden"], quant_config=_quant_config())
    after = _read_seam_counters()
    _assert_route_predicate("shared experts", {"blockwise_fp8_mm": 3}, before, after)

    if tuple(got.shape) != (TOKENS, HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    # The same weights arranged the way the checkpoint loader leaves them.
    tiles_per_block = DENSE_BLOCK // TILE_SIZE
    loaded = model_fp8.Glm5NextSharedExperts(text_config)
    for leaf in leaves:
        compute_weight, public_grid = operands[leaf]
        checkpoint_grid = public_grid.repeat_interleave(
            tiles_per_block, dim=0
        ).repeat_interleave(tiles_per_block, dim=1)
        _attach(
            loaded,
            leaf,
            compute_weight.t().contiguous(),
            checkpoint_grid.t().contiguous(),
            prep_will_compensate=True,
        )

    republished = loaded.retile_checkpoint_scale_grids()
    if republished != 3:
        raise VacuousControlError(
            f"the load-path prep published {republished} projections, not 3; every "
            f"extent here is a whole {DENSE_BLOCK} block"
        )
    health = getattr(loaded, loaded.SHARED_RETILE_HEALTH_ATTR)
    for leaf in leaves:
        compute_weight, public_grid = operands[leaf]
        assert leaf in health, f"the retile health record has no entry for {leaf}"
        if tuple(getattr(loaded, leaf).shape) != tuple(compute_weight.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf} is {tuple(getattr(loaded, leaf).shape)}, "
                f"not the kernel's frame {tuple(compute_weight.shape)}"
            )
        if tuple(getattr(loaded, _scale_grid_attribute(leaf)).shape) != tuple(
            public_grid.shape
        ):
            raise ReferenceShapeError(
                f"after the prep {leaf}'s grid is not the public grid "
                f"{tuple(public_grid.shape)}"
            )

    built_after = loaded.prepare_scale_operands(
        *(getattr(loaded, leaf) for leaf in leaves),
        *(getattr(loaded, _scale_grid_attribute(leaf)) for leaf in leaves),
    )
    if built_after != 3:
        raise VacuousControlError(
            f"the scale prep reported {built_after} operands after the republish, "
            f"not 3"
        )

    _reset_seam_counters()
    before_loaded = _read_seam_counters()
    got_loaded = loaded.forward(operands["hidden"], quant_config=_quant_config())
    after_loaded = _read_seam_counters()
    _assert_route_predicate(
        "shared experts from the loader frame",
        {"blockwise_fp8_mm": 3},
        before_loaded,
        after_loaded,
    )
    torch.testing.assert_close(
        got_loaded.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    removed = _scale_grid_attribute("down_proj_weight")
    delattr(module, removed)
    with pytest.raises(model_fp8.Glm5NextSharedExpertRouteError) as missing:
        module.forward(operands["hidden"], quant_config=_quant_config())
    if removed not in str(missing.value):
        raise VacuousControlError(
            f"the refusal for a missing grid does not name {removed}: "
            f"{missing.value}"
        )


def test_tiny_moe_block_forward_matches_the_reference() -> None:
    """One sparse layer's MLP: route, run the experts, add the shared expert once."""
    model_fp8 = _impl()
    text_config = _routed_text_config()
    quant_config = _quant_config()

    if int(text_config.n_shared_experts) < 1:
        raise VacuousControlError(
            f"the test needs a shared expert and the config declares "
            f"n_shared_experts={text_config.n_shared_experts}"
        )

    raw = _pinned_raw_config()
    raw_text = raw.get("text_config", raw)
    groups = int(raw_text.get("n_group", 1))
    kept = int(raw_text.get("topk_group", 1))
    if groups != kept:
        raise VacuousControlError(
            f"the pinned checkpoint declares n_group={groups} and "
            f"topk_group={kept}, so the reference's group mask is not an identity "
            f"and this router, which takes no group arguments, computes a "
            f"different selection"
        )

    block = model_fp8.Glm5NextMoEBlock(text_config, world_size=1, ep_degree=1)
    if getattr(block, "shared_experts", None) is None:
        raise VacuousControlError(
            "the block built no shared expert, so the one add under test is not on "
            "its route at all"
        )

    routed = _routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(block.experts, leaf, *routed[leaf])
    bank_built = block.experts.prepare_scale_operands(
        *_prep_operands_from_the_module(block.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), routed)
    )

    shared = _shared_at_routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(block.shared_experts, leaf, *shared[leaf])
    shared_built = block.shared_experts.prepare_scale_operands(
        *block.shared_experts.scale_route_operands()
    )

    generator = torch.Generator().manual_seed(SEED_MOE_ROUTER)
    block.experts.router_weight = torch.nn.Parameter(
        (
            torch.randn(
                ROUTED_HIDDEN_SIZE, ROUTED_EXPERTS, generator=generator
            )
            * MOE_ROUTER_WEIGHT_SCALE
        ).to(torch.bfloat16),
        requires_grad=False,
    )
    block.experts.router_bias = torch.nn.Parameter(
        (
            torch.randn(ROUTED_EXPERTS, generator=generator)
            * MOE_ROUTER_BIAS_SCALE
        ).to(torch.bfloat16),
        requires_grad=False,
    )

    parameters = sum(int(p.numel()) for p in block.parameters() if p is not None)
    if bank_built != 2 or shared_built != 3:
        raise VacuousControlError(
            f"the load-time preps built {bank_built} bank and {shared_built} "
            f"shared operands; the two forwards look up 2 and 3"
        )
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny MoE block holds {parameters} parameters, at or above the "
            f"tiny-config bound of {MAX_PARAMETERS}"
        )

    pre_norm = (
        _fp8_grid_values(SEED_MOE_HIDDEN, TOKENS, ROUTED_HIDDEN_SIZE)
        * float(2.0**HIDDEN_SCALE_EXPONENT)
    ).to(torch.bfloat16)
    gamma = _ffn_gamma()
    eps = float(text_config.rms_norm_eps)
    normed = _ffn_norm(pre_norm, gamma, eps)

    _logits_a, _index_a, affinities = block.experts.route_tokens(
        pre_norm.unsqueeze(0), gamma, text_config
    )
    _logits_b, _index_b, again = block.experts.route_tokens(
        pre_norm.unsqueeze(0), gamma, text_config
    )
    if not torch.equal(affinities, again):
        raise VacuousControlError(
            "route_tokens returned different affinities for the same operands, so "
            "the reference cannot be built from one call and compared against "
            "another"
        )
    selected = int((affinities != 0).sum(dim=1).min())
    if tuple(affinities.shape) != (TOKENS, ROUTED_EXPERTS):
        raise ReferenceShapeError(
            f"route_tokens returned {tuple(affinities.shape)}, expected "
            f"{(TOKENS, ROUTED_EXPERTS)}"
        )
    if selected != int(text_config.num_experts_per_tok):
        raise VacuousControlError(
            f"a token carries {selected} nonzero router columns and the config "
            f"declares top-{int(text_config.num_experts_per_tok)}"
        )

    limit = float(text_config.swiglu_limit)
    routed_reference = _routed_output(
        {**routed, "hidden": normed, "expert_affinities": affinities},
        mode=_POST_SCALE,
        gate_max=limit,
        gate_min=None,
        up_max=limit,
        up_min=-limit,
    )
    shared_reference = _dense_output(
        {**shared, "hidden": normed}, limit, -limit, limit
    )
    expected = routed_reference["out"] + shared_reference["out"]

    for name, variant in (
        ("shared half omitted", routed_reference["out"]),
        ("shared half added twice", expected + shared_reference["out"]),
    ):
        moved = not torch.allclose(variant, expected, rtol=MOE_RTOL, atol=MOE_ATOL)
        if not moved:
            raise VacuousControlError(
                f"with the {name} the result is still inside rtol={MOE_RTOL}, "
                f"atol={MOE_ATOL}; the test cannot tell the one add from the wrong "
                f"count"
            )

    _reset_seam_counters()
    before = _read_seam_counters()
    got = block.forward(
        pre_norm,
        normed,
        router_gamma=gamma,
        text_config=text_config,
        quant_config=quant_config,
    )
    after = _read_seam_counters()
    _assert_route_predicate(
        "MoE block",
        {"moe_fused": 1,
         "blockwise_fp8_mm": 3, "noaux_tc_router": 1},
        before,
        after,
    )

    if tuple(got.shape) != (TOKENS, ROUTED_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, ROUTED_HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(got.float(), expected.float(),
                               rtol=MOE_RTOL, atol=MOE_ATOL)

    # The block must route on its first argument (the pre-norm states) and not
    # on its second (the normalised states). Find a channel swap that changes the
    # selected expert set, then plant it in each argument in turn.
    _real_route_tokens = block.experts.route_tokens
    _had_own_route_tokens = "route_tokens" in vars(block.experts)
    _recorded_sets = []

    def _recording_route_tokens(*args, **kwargs):
        """The block's own router, with the set it returns recorded."""
        _logits, _index, _affinities = _real_route_tokens(*args, **kwargs)
        _selected = (_affinities != 0).clone()
        if _selected.dim() == 3 and int(_selected.shape[0]) == 1:
            _selected = _selected[0]
        _recorded_sets.append(_selected)
        return _logits, _index, _affinities

    def _forward_recording_the_router(_first, _second):
        """``block.forward(_first, _second)``, and every router call it made."""
        del _recorded_sets[:]
        block.experts.route_tokens = _recording_route_tokens
        try:
            _out = block.forward(
                _first,
                _second,
                router_gamma=gamma,
                text_config=text_config,
                quant_config=quant_config,
            )
        finally:
            if _had_own_route_tokens:
                block.experts.route_tokens = _real_route_tokens
            else:
                del block.experts.route_tokens
        return _out, list(_recorded_sets)

    def _routed_or_raise(_sets, _which):
        """The first recorded set, or a refusal naming the router path."""
        if not _sets:
            raise VacuousControlError(
                f"the {_which} forward made no call to block.experts.route_tokens, "
                f"the method this check wrapped and the one "
                f"Glm5NextMoEBlock.forward routes with, so nothing the check "
                f"perturbs can be read from the router's output"
            )
        return _sets[0]

    _ref_out, _ref_sets = _forward_recording_the_router(pre_norm, normed)
    _ref_set = _routed_or_raise(_ref_sets, "reference")
    _base_set = affinities != 0
    _probes = 0
    _candidates = []
    for _i, _j in MOE_SWAP_CHANNEL_CANDIDATES:
        for _t in range(min(int(pre_norm.shape[0]), MOE_SWAP_TOKEN_SCAN)):
            _candidate = pre_norm.clone()
            _candidate[_t, [_i, _j]] = _candidate[_t, [_j, _i]]
            if torch.equal(_candidate[_t], pre_norm[_t]):
                continue
            _probes += 1
            _, _, _probe_affinities = block.experts.route_tokens(
                _candidate.unsqueeze(0), gamma, text_config
            )
            if not torch.equal((_probe_affinities != 0)[_t], _base_set[_t]):
                _candidates.append((_t, _i, _j, _candidate))
                break
    if not _candidates:
        raise VacuousControlError(
            f"no swap among {len(MOE_SWAP_CHANNEL_CANDIDATES)} channel pairs on any "
            f"of the first {MOE_SWAP_TOKEN_SCAN} tokens changed the selected expert "
            f"set in {_probes} probes of this block's own router, so no perturbation "
            f"this check can make would be visible in the forward's output"
        )
    _evaluated = []
    for _t, _i, _j, _candidate in _candidates:
        _probe_out, _probe_sets = _forward_recording_the_router(_candidate, normed)
        _probe_set = _routed_or_raise(_probe_sets, "perturbed")
        _through = not torch.equal(_probe_set[_t], _ref_set[_t])
        _probe_gap = float(
            (_probe_out[_t].float() - expected[_t].float()).abs().max()
            / expected[_t].abs().max()
        )
        if _through:
            _evaluated.append((_probe_gap, _t, _i, _j))
    if not _evaluated:
        raise VacuousControlError(
            f"{len(_candidates)} channel swaps changed the selected expert set when "
            f"this check called block.experts.route_tokens itself, and not one of "
            f"them changed the set the same router returned inside block.forward: "
            f"the forward does not route on the argument it was handed the swap in"
        )
    _, _swap_token, _swap_i, _swap_j = max(_evaluated, key=lambda item: item[0])
    _arm_second = normed.clone()
    _arm_second[_swap_token, [_swap_i, _swap_j]] = _arm_second[
        _swap_token, [_swap_j, _swap_i]
    ]
    _arm_out, _arm_sets = _forward_recording_the_router(pre_norm, _arm_second)
    _arm_set = _routed_or_raise(_arm_sets, "second-argument")
    _arm_changed = not torch.equal(_arm_set[_swap_token], _ref_set[_swap_token])
    if _arm_changed:
        raise VacuousControlError(
            f"the swap placed in argument 2 changed the selected expert set inside "
            f"the forward at token {_swap_token} while argument 1 was the main "
            f"compare's own tensor, so this forward routes on the states it hands "
            f"the experts instead of the pre-norm states"
        )

    # With n_shared_experts=0 the block must run the bank alone.
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    dense_free_config = Glm5NextTextConfig(
        hidden_size=ROUTED_HIDDEN_SIZE,
        intermediate_size=ROUTED_INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        n_routed_experts=ROUTED_EXPERTS,
        num_experts_per_tok=ROUTED_EXPERTS_PER_TOKEN,
        n_shared_experts=0,
    )
    bare = model_fp8.Glm5NextMoEBlock(dense_free_config, world_size=1, ep_degree=1)
    if getattr(bare, "shared_experts", None) is not None:
        raise VacuousControlError(
            "the block still built a shared expert at n_shared_experts=0, so this "
            "check does not exercise the branch it names"
        )
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(bare.experts, leaf, *routed[leaf])
    bare.experts.prepare_scale_operands(
        *_prep_operands_from_the_module(bare.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), routed)
    )
    bare.experts.router_weight = block.experts.router_weight
    bare.experts.router_bias = block.experts.router_bias

    _reset_seam_counters()
    before = _read_seam_counters()
    bare_got = bare.forward(
        pre_norm,
        normed,
        router_gamma=gamma,
        text_config=dense_free_config,
        quant_config=quant_config,
    )
    after = _read_seam_counters()
    _assert_route_predicate(
        "MoE block, no shared expert",
        {"moe_fused": 1, "noaux_tc_router": 1},
        before,
        after,
    )
    torch.testing.assert_close(
        bare_got.float(), routed_reference["out"].float(),
        rtol=MOE_RTOL, atol=MOE_ATOL
    )
