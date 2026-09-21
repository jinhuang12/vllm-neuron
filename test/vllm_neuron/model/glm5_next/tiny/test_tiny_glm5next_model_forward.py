"""Forward-pass tests for the tiny GLM-5.3-Flash decoder stack and root model.

The three-layer stack with its hyper-connection streams is compared to its torch
composition layer boundary by layer boundary; the root is compared to the head
projection of the rows it was asked to sample. Fixtures come from
``test_tiny_glm5next_forward``.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.tiny.test_tiny_glm5next_forward import (
    ATOL,
    MHC_SITES_PER_LAYER,
    MOE_ATOL,
    MOE_RTOL,
    ROOT_MAX_CONDITION,
    ROOT_SAMPLING_POSITIONS,
    RTOL,
    SEED_STACK_EMBED,
    SEED_STACK_IDS,
    STACK_DENSE_LAYERS,
    STACK_HIDDEN_SIZE,
    STACK_LAYERS,
    STACK_MAX_CONDITION,
    STACK_MOE_LAYERS,
    STACK_PAGES,
    STACK_TOKENS,
    STACK_VOCAB_SIZE,
    ReferenceShapeError,
    VacuousControlError,
    _SEAMS,
    _assert_route_predicate,
    _clamped,
    _declare_bound_and_sentinel,
    _ffn_norm,
    _fp8_grid_values,
    _impl,
    _mla_selection_operands,
    _quant_config,
    _read_seam_counters,
    _reset_seam_counters,
    _root_config,
    _root_fixture,
    _root_reference,
    _stack_attention_half,
    _stack_carriers,
    _stack_ffn_half,
    _stack_fixture,
    _stack_mhc_post,
    _stack_mhc_pre,
    _stack_mhc_site,
)

pytestmark = [pytest.mark.fast]

# Boundary comparisons use a band scaled by the reference's peak: the streams
# grow through the stack, and a fixed atol would be either blind or too tight.
STACK_RECOMPUTE_RTOL = 2.0**-8
STACK_RECOMPUTE_ATOL_FACTOR = 2.0**-8
STACK_RECOMPUTE_PEAK_CEILING = 256.0

# How far past the tolerance band a planted hidden-state change moves a logit.
ROOT_PLANT_MULTIPLE = 4.0


def _stack_peak_band(expected: torch.Tensor, label: str) -> tuple:
    """``(rtol, atol)`` for one boundary comparison, scaled by the reference's peak."""
    peak = float(expected.abs().max())
    if not peak > 0.0:
        raise VacuousControlError(
            f"{label}: the reference's peak is {peak}, so a peak-scaled band would "
            f"be a bare equality test and this comparison could not fail on any "
            f"error the recompute makes"
        )
    if peak > STACK_RECOMPUTE_PEAK_CEILING:
        raise VacuousControlError(
            f"{label}: the reference peaks at {peak:.6g}, above the ceiling of "
            f"{STACK_RECOMPUTE_PEAK_CEILING}; a reference that grew this far may "
            f"not widen its own tolerance"
        )
    return STACK_RECOMPUTE_RTOL, peak * STACK_RECOMPUTE_ATOL_FACTOR


def _stack_outside_tolerance(label: str, moved: torch.Tensor,
                             base: torch.Tensor) -> None:
    """``moved`` must fall outside the test's tolerance band around ``base``."""
    outside = not torch.allclose(moved.float(), base.float(), rtol=RTOL, atol=ATOL)
    if not outside:
        raise VacuousControlError(
            f"{label}: the change leaves the reference inside rtol={RTOL}, "
            f"atol={ATOL}, so the test cannot tell the two apart"
        )


def _max_row_spread(rows: torch.Tensor) -> float:
    """The largest relative L2 distance between any two rows of ``rows``."""
    flat = rows.reshape(rows.shape[0], -1).float()
    dist = (flat.unsqueeze(1) - flat.unsqueeze(0)).norm(dim=2)
    norms = flat.norm(dim=1)
    mean = (norms.unsqueeze(1) + norms.unsqueeze(0)) / 2.0
    spread = dist / mean.clamp_min(1e-12)
    upper = torch.triu(torch.ones_like(spread), diagonal=1) > 0
    vals = spread[upper]
    if vals.numel() == 0:
        return 0.0
    return float(vals.max())


def test_tiny_model_forward_matches_the_reference() -> None:
    """The decoder stack equals its torch composition, layer boundary by boundary."""
    fixture = _stack_fixture()
    model, cfg, layers = fixture["model"], fixture["cfg"], fixture["layers"]
    quant_config = _quant_config()
    selection = _mla_selection_operands(tokens=STACK_TOKENS, pages=STACK_PAGES)
    carriers = _stack_carriers(layers, selection)

    input_ids = torch.randint(
        0, STACK_VOCAB_SIZE, (STACK_TOKENS,),
        generator=torch.Generator().manual_seed(SEED_STACK_IDS),
        dtype=torch.int64,
    )
    if int(input_ids.unique().numel()) < STACK_LAYERS:
        raise VacuousControlError(
            f"the token ids take only {int(input_ids.unique().numel())} distinct "
            f"values; a near-constant embedding makes every row of the residual "
            f"stream alike and the comparison stops discriminating"
        )

    recorded_in: list = []
    recorded_out: list = []
    recorded_mlp: list = []

    def _record_input(module, args, kwargs):
        recorded_in.append((module, args, kwargs))

    def _record_output(module, args, kwargs, output):
        recorded_out.append((module, output))

    def _record_mlp(module, args, kwargs, output):
        recorded_mlp.append((module, args, kwargs, output))

    handles = []
    for layer in layers:
        handles.append(
            layer.register_forward_pre_hook(_record_input, with_kwargs=True)
        )
        handles.append(
            layer.register_forward_hook(_record_output, with_kwargs=True)
        )
    handles.append(
        layers[-1].mlp.register_forward_hook(_record_mlp, with_kwargs=True)
    )

    try:
        _reset_seam_counters()
        before = _read_seam_counters()
        got = model.forward(
            input_ids,
            layer_carriers=carriers,
            quant_config=quant_config,
        )
        after = _read_seam_counters()
    finally:
        for handle in handles:
            handle.remove()

    route_expected = {
        "mla_projection": 9 * STACK_LAYERS,
        "mla_absorb": 2 * STACK_LAYERS,
        "mla_sparse": 1 * STACK_LAYERS,
        "dsa_kpool_hadamard": 2 * STACK_LAYERS,
        "dsa_paged_gather": 1 * STACK_LAYERS,
        "dsa_score_gemm": 1 * STACK_LAYERS,
        "dsa_topk_select": 1 * STACK_LAYERS,
        "dsa_index_expand": 1 * STACK_LAYERS,
        "blockwise_fp8_mm": 3 * STACK_DENSE_LAYERS,
        "moe_fused": 1 * STACK_MOE_LAYERS,
        "noaux_tc_router": 1 * STACK_MOE_LAYERS,
        "mhc_sinkhorn": MHC_SITES_PER_LAYER * STACK_LAYERS,
        "mhc_hyper_connection": MHC_SITES_PER_LAYER * STACK_LAYERS,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("decoder stack", route_expected, before, after)

    if len(recorded_in) != len(layers) or len(recorded_out) != len(layers):
        raise VacuousControlError(
            f"the hooks recorded {len(recorded_in)} entries and "
            f"{len(recorded_out)} exits for {len(layers)} layers, so the forward "
            f"did not run each layer exactly once"
        )
    for index, layer in enumerate(layers):
        if recorded_in[index][0] is not layer or recorded_out[index][0] is not layer:
            raise VacuousControlError(
                f"position {index} of the recorded order is not layer {index} of "
                f"the stack, so the loop did not run the layers in config order"
            )

    # The dense layers' activations must be non-degenerate at this scale: the
    # SwiGLU clamp live on both projections, rows still different from each
    # other, and the stream mix carrying more than the FFN half alone.
    from torch.nn.functional import silu

    for dense_index in fixture["dense_at"]:
        streams = recorded_out[dense_index][1]
        site = _stack_mhc_site(layers[dense_index], _impl().MHC_FFN_SITE,
                               f"dense layer {dense_index}")
        post_mix, comb_mix, collapsed = _stack_mhc_pre(
            site, streams, f"dense layer {dense_index}"
        )
        limit = float(cfg.swiglu_limit)
        half = _stack_ffn_half(
            layers[dense_index], collapsed, cfg,
            fixture["mlp_operands"][dense_index], routed=False,
        )
        gate, up = half["gate"], half["up"]
        activated = silu(_clamped(gate, None, limit)) * _clamped(up, -limit, limit)
        mixed = site.mhc_post(half["out"], streams, post_mix, comb_mix)
        half_only = site.mhc_post(
            half["out"], torch.zeros_like(streams), post_mix, comb_mix
        )
        gate_at_limit = float((gate >= limit).float().mean())
        up_at_limit = float((up.abs() >= limit).float().mean())
        absorbed = float((mixed.float() == half_only.float()).float().mean())
        activated_spread = _max_row_spread(activated)
        mixed_spread = _max_row_spread(mixed)
        ffn_spread = _max_row_spread(half["out"])
        if not (gate_at_limit > 0.0 and up_at_limit > 0.0 and activated_spread > 0.0):
            raise VacuousControlError(
                f"layer {dense_index}: gate at-limit fraction {gate_at_limit:.6f}, up "
                f"at-limit fraction {up_at_limit:.6f}, activated row spread "
                f"{activated_spread:.6g}. The test needs the SwiGLU clamp live on "
                f"both projections and the activated rows still different: a zero "
                f"fraction means the clamp is dead at this scale, and a zero spread "
                f"means it saturated every row into the same constant"
            )
        if not (absorbed < 1.0 and mixed_spread > 0.0):
            raise VacuousControlError(
                f"layer {dense_index}: the mHC mix equals the same mix with the "
                f"streams zeroed in {absorbed:.6f} of elements and the mix's row "
                f"spread is {mixed_spread:.6g}. The FFN half peaks at "
                f"{float(half['out'].abs().max()):.6g} against a collapsed "
                f"stream peak of {float(collapsed.float().abs().max()):.6g}, so the "
                f"mix carries the FFN half alone and every comparison below it is "
                f"blind to the streams it was supposed to mix"
            )
        if not ffn_spread > 0.0:
            raise VacuousControlError(
                f"layer {dense_index}: the dense half's own output rows have pairwise "
                f"spread {ffn_spread:.6g}, so every token leaves this half holding "
                f"the same vector and no comparison below can see a per-token defect"
            )

    first_input = recorded_in[0][1][0]
    embedded = fixture["table"][input_ids]
    expanded = embedded.unsqueeze(1).expand(-1, fixture["hc_mult"], -1)
    if tuple(first_input.shape) != tuple(expanded.shape):
        raise ReferenceShapeError(
            f"the first layer was handed {tuple(first_input.shape)} and the expanded "
            f"embedding is {tuple(expanded.shape)}; the stack's carrier is "
            f"[T, hc_mult, H] and a different rank is a different network, not a "
            f"numeric difference"
        )
    if not torch.equal(first_input, expanded):
        raise VacuousControlError(
            "the first layer was handed a tensor that is not "
            "embed_tokens_weight[input_ids] expanded across the stream axis, so "
            "either the embedding is not that index or the streams did not start "
            "as one token vector"
        )
    for stream in range(int(first_input.shape[1])):
        if not torch.equal(first_input[:, stream, :], embedded):
            raise VacuousControlError(
                f"stream {stream} of the first layer's input is not the embedding "
                f"itself, so the expand carried something other than the token vector"
            )

    streams_keyword = "streams"
    for index, carrier in enumerate(carriers):
        got_kwargs = recorded_in[index][2]
        if set(got_kwargs) != set(carrier) | {streams_keyword}:
            raise VacuousControlError(
                f"layer {index} was handed the keywords {sorted(got_kwargs)}; "
                f"expected its carrier's {sorted(carrier)} plus "
                f"{streams_keyword!r}, the route selector the streams path adds"
            )
        wrong = [key for key in carrier if got_kwargs[key] is not carrier[key]]
        if wrong:
            raise VacuousControlError(
                f"layer {index} received {wrong} from some other object than its "
                f"own carrier mapping, so the per-layer state is not bound to the "
                f"layer that owns it"
            )
        if got_kwargs[streams_keyword] is not recorded_in[index][1][0]:
            raise VacuousControlError(
                f"layer {index} was handed one tensor positionally and a different "
                f"object as {streams_keyword!r}; the stack passes one tensor as both "
                f"arguments on purpose, so a split here means the mHC pre and the "
                f"one-stream operand disagree about what this layer's input is"
            )

    # Attention half: each layer's output streams equal the mix of its input
    # streams with the torch attention reference.
    for index, layer in enumerate(layers):
        raw, gains = fixture["attention_operands"][index]
        hidden = recorded_in[index][1][0]
        site = _stack_mhc_site(layer, _impl().MHC_ATTENTION_SITE,
                               f"attention half, layer {index}")
        post_mix, comb_mix, layer_input = _stack_mhc_pre(
            site, hidden, f"attention half, layer {index}"
        )
        _, topk_indices, attended = _stack_attention_half(
            layer, raw, gains, layer_input, cfg, selection
        )
        expected = _stack_mhc_post(site, attended, hidden, post_mix, comb_mix)
        produced = recorded_out[index][1].float()
        if tuple(produced.shape) != tuple(expected.shape):
            raise ReferenceShapeError(
                f"layer {index} returned {tuple(produced.shape)} and the mixed "
                f"reference is {tuple(expected.shape)}; the streams route returns the "
                f"streams, so a rank disagreement is a route disagreement"
            )
        rtol, atol = _stack_peak_band(expected, f"attention half, layer {index}")
        torch.testing.assert_close(produced, expected, rtol=rtol, atol=atol)

    # FFN half: each layer's output streams, mixed with the torch FFN reference,
    # equal the next layer's input streams.
    ffn = []
    ffn_mix = []
    for index, layer in enumerate(layers):
        hidden = recorded_out[index][1]
        routed = index in fixture["moe_at"]
        site = _stack_mhc_site(layer, _impl().MHC_FFN_SITE, f"FFN half, layer {index}")
        post_mix, comb_mix, layer_input = _stack_mhc_pre(
            site, hidden, f"FFN half, layer {index}"
        )
        half = _stack_ffn_half(
            layer, layer_input, cfg, fixture["mlp_operands"][index], routed=routed
        )
        ffn.append(half)
        ffn_mix.append((site, post_mix, comb_mix, layer_input))
        if routed:
            if half["condition"] > STACK_MAX_CONDITION:
                raise VacuousControlError(
                    f"layer {index}'s bank output has condition "
                    f"{half['condition']:.4f} against a bound of "
                    f"{STACK_MAX_CONDITION}: the output is a small residue of "
                    f"large opposing terms, so every reading here is inflated by "
                    f"a vanishing denominator"
                )
        expected = _stack_mhc_post(site, half["out"], hidden, post_mix, comb_mix)
        if index + 1 < len(layers):
            next_input = recorded_in[index + 1][1][0].float()
            rtol, atol = _stack_peak_band(expected, f"FFN half, layer {index}")
            torch.testing.assert_close(next_input, expected, rtol=rtol, atol=atol)

    # Final norm: the last layer's routed MLP output, mixed, averaged over the
    # streams and normalised, is what the stack returns.
    last = recorded_out[-1][1]
    if not recorded_mlp:
        raise VacuousControlError(
            "the last layer's MLP hook recorded nothing, so the product never "
            "called the routed MLP and there is no product half either to "
            "compare or to add; a skip here would hide a stack that ran no experts"
        )
    half_product = recorded_mlp[-1][3]
    half_ref = ffn[-1]["out"]
    if tuple(half_product.shape) != tuple(half_ref.shape):
        raise ReferenceShapeError(
            f"the routed MLP returned {tuple(half_product.shape)} and the "
            f"reference computed {tuple(half_ref.shape)}; the two are compared "
            f"cell by cell and one of them is added to the residual, so a shape "
            f"disagreement is a refusal and not a skip"
        )
    final_site, final_post_mix, final_comb_mix, _ = ffn_mix[-1]
    final_streams = final_site.mhc_post(
        half_product, last, final_post_mix, final_comb_mix
    )
    if tuple(final_streams.shape) != tuple(last.shape):
        raise ReferenceShapeError(
            f"the last layer's feed-forward site returned "
            f"{tuple(final_streams.shape)} for {tuple(last.shape)} streams; the mix "
            f"returns the streams it was handed, and the mean below reduces that axis"
        )
    final_input = final_streams.mean(dim=1).to(fixture["table"].dtype)
    expected = _ffn_norm(final_input, fixture["final_gain"], float(cfg.rms_norm_eps))
    if tuple(got.shape) != (STACK_TOKENS, STACK_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(STACK_TOKENS, STACK_HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(half_product.float(), half_ref,
                               rtol=MOE_RTOL, atol=MOE_ATOL)
    rtol, atol = _stack_peak_band(expected.float(), "the final norm")
    torch.testing.assert_close(got.float(), expected.float(), rtol=rtol, atol=atol)

    # Each dense layer must read its own weights and its own norm gain.
    first_dense, second_dense = fixture["dense_at"]
    _stack_outside_tolerance(
        f"layer {second_dense}'s FFN recomputed with layer {first_dense}'s weights",
        _stack_ffn_half(
            layers[second_dense], ffn_mix[second_dense][3], cfg,
            fixture["mlp_operands"][first_dense], routed=False,
        )["out"],
        ffn[second_dense]["out"],
    )

    _stack_outside_tolerance(
        f"layer {first_dense}'s FFN normalised with its input gain",
        _stack_ffn_half(
            layers[first_dense], ffn_mix[first_dense][3], cfg,
            fixture["mlp_operands"][first_dense], routed=False,
            gain=layers[first_dense].input_layernorm_weight,
        )["out"],
        ffn[first_dense]["out"],
    )

    # The routed block must be handed the pre-norm states first and their FFN
    # norm second, with the router's gain being this layer's post-attention gain.
    moe_layer = fixture["moe_at"][0]
    c_gain = layers[-1].post_attention_layernorm_weight
    c_eps = float(cfg.rms_norm_eps)
    c_args = recorded_mlp[-1][1]
    c_kwargs = recorded_mlp[-1][2]
    if len(c_args) < 2:
        raise ReferenceShapeError(
            f"the routed MLP was called with {len(c_args)} positional tensors; this "
            f"check reads the two activation tensors the block is handed, so a "
            f"shorter call is a refusal and not a skip"
        )
    c_normed = _ffn_norm(c_args[0], c_gain, c_eps)
    order_holds = torch.equal(c_args[1], c_normed)
    worst_cell = float((c_args[1].float() - c_normed.float()).abs().max())
    c_router_gamma = c_kwargs.get("router_gamma")
    gain_holds = c_router_gamma is c_gain or (
        c_router_gamma is not None and torch.equal(c_router_gamma, c_gain)
    )
    if not (order_holds and gain_holds):
        raise AssertionError(
            f"layer {moe_layer}'s block was handed its arguments in the wrong order: "
            f"the second positional tensor is the FFN norm of the first is "
            f"{order_holds} (worst cell {worst_cell:.6g}) and the router's gain is "
            f"this layer's post-attention gain is {gain_holds}; the fused router "
            f"applies the norm itself, so the pre-norm states go first and the "
            f"normalised tensor second"
        )
    if torch.equal(c_args[0], _ffn_norm(c_args[1], c_gain, c_eps)):
        raise VacuousControlError(
            f"layer {moe_layer}'s argument-order check cannot tell the two apart: "
            f"reading the tensors swapped also holds"
        )

    with pytest.raises(ValueError, match="one mapping per layer"):
        model.forward(
            input_ids,
            layer_carriers=carriers[:-1],
            quant_config=quant_config,
        )

    # A missing embedding table is refused before any kernel dispatches.
    model.embed_tokens_weight = None
    _reset_seam_counters()
    unloaded_before = _read_seam_counters()
    with pytest.raises(ValueError, match="no embed_tokens_weight"):
        model.forward(
            input_ids,
            layer_carriers=carriers,
            quant_config=quant_config,
        )
    unloaded_after = _read_seam_counters()
    moved = {
        seam: unloaded_after[seam][0] - unloaded_before[seam][0]
        for seam in _SEAMS
        if unloaded_after[seam][0] != unloaded_before[seam][0]
    }
    if moved:
        raise VacuousControlError(
            f"the refusal ran after {sorted(moved.items())} dispatched, so it is "
            f"not reached before the stack starts"
        )


def test_tiny_root_forward_matches_the_reference() -> None:
    """The root equals the head projection of the rows it was asked to sample."""
    fixture = _root_fixture()
    root, layers = fixture["root"], fixture["layers"]
    head = fixture["head"]
    selection = _mla_selection_operands(tokens=STACK_TOKENS, pages=STACK_PAGES)
    carriers = _stack_carriers(layers, selection)
    positions = torch.tensor(ROOT_SAMPLING_POSITIONS, dtype=torch.long)

    input_ids = torch.randint(
        0, STACK_VOCAB_SIZE, (STACK_TOKENS,),
        generator=torch.Generator().manual_seed(SEED_STACK_IDS),
        dtype=torch.int64,
    )

    recorded: list = []

    def _record_stack_call(module, args, kwargs, output):
        recorded.append((args, kwargs, output))

    handle = root.model.register_forward_hook(_record_stack_call, with_kwargs=True)
    try:
        _reset_seam_counters()
        before = _read_seam_counters()
        got = root.forward(
            input_ids,
            layer_carriers=carriers,
            sampling_positions=positions,
        )
        after = _read_seam_counters()
    finally:
        handle.remove()

    route_expected = {
        "mla_projection": 9 * STACK_LAYERS,
        "mla_absorb": 2 * STACK_LAYERS,
        "mla_sparse": 1 * STACK_LAYERS,
        "dsa_kpool_hadamard": 2 * STACK_LAYERS,
        "dsa_paged_gather": 1 * STACK_LAYERS,
        "dsa_score_gemm": 1 * STACK_LAYERS,
        "dsa_topk_select": 1 * STACK_LAYERS,
        "dsa_index_expand": 1 * STACK_LAYERS,
        "blockwise_fp8_mm": 3 * STACK_DENSE_LAYERS,
        "moe_fused": 1 * STACK_MOE_LAYERS,
        "noaux_tc_router": 1 * STACK_MOE_LAYERS,
        "mhc_sinkhorn": MHC_SITES_PER_LAYER * STACK_LAYERS,
        "mhc_hyper_connection": MHC_SITES_PER_LAYER * STACK_LAYERS,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("root", route_expected, before, after)

    if len(recorded) != 1:
        raise VacuousControlError(
            f"the root called its stack {len(recorded)} times; the forward under "
            f"test calls it exactly once"
        )
    args, kwargs, hidden = recorded[0]
    expected_keys = {
        "layer_carriers", "quant_config", "block_size", "moe_group", "tp_degree",
        "expert_parallel_rank",
    }
    if len(args) != 1 or args[0] is not input_ids:
        raise VacuousControlError(
            f"the stack was handed {len(args)} positional arguments and the first "
            f"is not the caller's input_ids, so the ids were rebuilt on the way "
            f"down"
        )
    if set(kwargs) != expected_keys:
        raise VacuousControlError(
            f"the stack was handed the keywords {sorted(kwargs)}; expected "
            f"{sorted(expected_keys)}"
        )
    if kwargs["layer_carriers"] is not carriers:
        raise VacuousControlError(
            "the stack received some other object than the caller's carrier "
            "sequence, so the per-layer state is not the caller's"
        )
    defaults = {"block_size": None, "moe_group": None, "tp_degree": 1,
                "expert_parallel_rank": 0}
    wrong = {
        name: kwargs[name] for name, value in defaults.items()
        if kwargs[name] != value
    }
    if wrong:
        raise VacuousControlError(
            f"the root forwarded {sorted(wrong.items())} where its own declared "
            f"defaults are {sorted(defaults.items())}"
        )

    resolved = kwargs["quant_config"]
    pinned = _quant_config()
    if not isinstance(resolved, _impl().Glm5NextQuantConfig):
        raise VacuousControlError(
            f"the root threaded down a {type(resolved).__name__}, not a "
            f"Glm5NextQuantConfig, so the MLPs' route selector is not the "
            f"resolved policy"
        )
    if not resolved.is_block_quantized or resolved.block_shape != pinned.block_shape:
        raise VacuousControlError(
            f"the root resolved is_block_quantized={resolved.is_block_quantized}, "
            f"block_shape={resolved.block_shape}; the pinned checkpoint's policy "
            f"is block-quantised at {pinned.block_shape}"
        )

    if fixture["tied"] or root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "the untied root's head tensor is not lm_head_weight, so the test is "
            "not measuring the untied head"
        )

    rows = len(ROOT_SAMPLING_POSITIONS)
    if tuple(got.shape) != (rows, STACK_VOCAB_SIZE):
        raise ReferenceShapeError(
            f"the root returned {tuple(got.shape)}, expected "
            f"{(rows, STACK_VOCAB_SIZE)}: {rows} requested rows by the "
            f"{STACK_VOCAB_SIZE}-wide vocabulary"
        )
    expected = _root_reference(hidden, head, ROOT_SAMPLING_POSITIONS)

    terms = torch.stack(
        [hidden[int(index)].abs().float() for index in ROOT_SAMPLING_POSITIONS]
    ) @ head.abs().float().t()
    condition = float((terms / expected.abs().clamp_min(1e-12)).max())
    if condition > ROOT_MAX_CONDITION:
        raise VacuousControlError(
            f"the logits' dot products have condition {condition:.4f} against a "
            f"bound of {ROOT_MAX_CONDITION}: each logit is a small residue of "
            f"large opposing terms, so every reading here is inflated by a "
            f"vanishing denominator"
        )
    torch.testing.assert_close(got.float(), expected, rtol=RTOL, atol=ATOL)

    first, second = (
        index for index, value in enumerate(ROOT_SAMPLING_POSITIONS)
        if value == ROOT_SAMPLING_POSITIONS[-1]
    )
    if not torch.equal(got[first], got[second]):
        raise VacuousControlError(
            f"rows {first} and {second} ask for the same position and came back "
            f"different, so the selection is not the index the caller gave"
        )

    # A tied root projects with the embedding table itself.
    tied_root = _impl().Glm5NextForConditionalGeneration(
        _root_config(tie_word_embeddings=True)
    )
    table = _fp8_grid_values(
        SEED_STACK_EMBED, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE
    ).to(torch.bfloat16)
    tied_root.model.embed_tokens_weight = torch.nn.Parameter(
        table, requires_grad=False
    )
    tied_declared = tied_root.declared_parameter_names()
    tied_head = tied_root._head_weight()
    if "lm_head_weight" in tied_declared:
        raise VacuousControlError(
            "the tied root declares lm_head_weight; the weight map adds no "
            "lm_head.weight entry in that case, so there is no checkpoint tensor "
            "to fill it"
        )
    if tied_head is not tied_root.model.embed_tokens_weight:
        raise VacuousControlError(
            "the tied root's head tensor is not the embedding table object, so "
            "the two can drift apart"
        )

    # A missing head is refused before the stack runs.
    saved_head = root.lm_head_weight
    try:
        root.lm_head_weight = None
        _reset_seam_counters()
        unloaded_before = _read_seam_counters()
        with pytest.raises(ValueError, match="no lm_head_weight"):
            root.forward(
                input_ids,
                layer_carriers=carriers,
                sampling_positions=positions,
            )
        unloaded_after = _read_seam_counters()
        moved = {
            seam: unloaded_after[seam][0] - unloaded_before[seam][0]
            for seam in _SEAMS
            if unloaded_after[seam][0] != unloaded_before[seam][0]
        }
        if moved:
            raise VacuousControlError(
                f"the refusal ran after {sorted(moved.items())} dispatched, so the "
                f"head is resolved after the stack instead of before it"
            )
    finally:
        root.lm_head_weight = saved_head
    if root.lm_head_weight is not saved_head:
        raise VacuousControlError(
            "root.lm_head_weight was not put back after the missing-head check, so "
            "every check after it runs against a root whose head the product refuses"
        )
    if root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "root._head_weight() no longer returns root.lm_head_weight after the "
            "missing-head check, so the head the product projects with is not the "
            "tensor the fixture loaded"
        )

    with pytest.raises(TypeError, match="sampling_positions"):
        root.forward(input_ids, layer_carriers=carriers)

    with pytest.raises(TypeError, match="sampling_params"):
        root.forward(
            input_ids,
            layer_carriers=carriers,
            sampling_positions=positions,
            sampling_params=None,
        )

    # Plant a change on the hidden state at the row slot 0 asks for and hand it
    # to the root's tail: slot 0 must move, every other slot must not.
    if root.lm_head_weight is None:
        raise VacuousControlError(
            "root.lm_head_weight is None before anything is planted; the product "
            "would refuse by name instead of selecting the rows this check asks about"
        )
    if root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "root._head_weight() does not return root.lm_head_weight before anything "
            "is planted, so this check cannot say which rows the root selected"
        )
    slot = 0
    plant_position = int(ROOT_SAMPLING_POSITIONS[slot])
    slot_peak = float(expected[slot].abs().max())
    slot_band = ATOL + RTOL * slot_peak
    head_rows = head.float()
    head_row = int(head_rows.norm(dim=1).argmax())
    head_norm2 = float(head_rows[head_row].pow(2).sum())
    if head_norm2 <= 0.0:
        raise VacuousControlError(
            f"head row {head_row} is the widest of {int(head_rows.shape[0])} and "
            f"still has zero norm, so no perturbation of the hidden state can move "
            f"the logit it projects"
        )
    plant_scale = ROOT_PLANT_MULTIPLE * slot_band
    probe_hidden = hidden.clone()
    probe_hidden[plant_position] = (
        hidden[plant_position].float()
        + (plant_scale / head_norm2) * head_rows[head_row]
    ).to(hidden.dtype)
    moved_reference = _root_reference(probe_hidden, head, ROOT_SAMPLING_POSITIONS)
    slot_diff = float((moved_reference[slot] - expected[slot]).abs().max())
    if slot_diff <= 0.0:
        raise VacuousControlError(
            f"the plant of {plant_scale:.6g} along head row {head_row} moved slot "
            f"{slot} of the reference by exactly nothing, so this head masks it"
        )
    planted_stack_calls = []

    def _plant_the_stack_output(_module, _args, _kwargs, _output):
        """Hand the root's own tail the planted state instead of the stack's."""
        planted_stack_calls.append(_output)
        return probe_hidden

    replay_carriers = _stack_carriers(layers, selection)
    plant_handle = root.model.register_forward_hook(
        _plant_the_stack_output, with_kwargs=True
    )
    try:
        got_planted = root.forward(
            input_ids,
            layer_carriers=replay_carriers,
            sampling_positions=positions,
        )
    finally:
        plant_handle.remove()
    if len(planted_stack_calls) != 1:
        raise VacuousControlError(
            f"the planting hook on root.model fired {len(planted_stack_calls)} "
            f"times where this check plants once, so the logits it compares are "
            f"not the product's reading of the planted state"
        )
    if tuple(got_planted.shape) != tuple(got.shape):
        raise ReferenceShapeError(
            f"the root returned {tuple(got_planted.shape)} on the planted state and "
            f"{tuple(got.shape)} on the stack's own, so the two cannot be compared "
            f"slot by slot"
        )
    planted_outside = not torch.allclose(got_planted[slot].float(),
                                         got[slot].float(),
                                         rtol=RTOL, atol=ATOL)
    if not planted_outside:
        raise VacuousControlError(
            f"the root's own logits for slot {slot} stayed inside rtol={RTOL}, "
            f"atol={ATOL} after {plant_scale:.6g} was planted on position "
            f"{plant_position}, the row that slot asks for, so this forward does "
            f"not read that row"
        )
    for other in range(1, len(ROOT_SAMPLING_POSITIONS)):
        other_diff = float(
            (got_planted[other].float() - got[other].float()).abs().max()
        )
        if not torch.equal(got_planted[other], got[other]):
            raise VacuousControlError(
                f"slot {other} asks for position "
                f"{int(ROOT_SAMPLING_POSITIONS[other])}, which this check planted "
                f"nothing on, and the root's own logits for it moved by "
                f"{other_diff:.6g}: this forward is reading a row the caller did not "
                f"ask for, or its projection mixes rows"
            )
    torch.testing.assert_close(got_planted.float(), moved_reference, rtol=RTOL, atol=ATOL)
    _stack_outside_tolerance(
        f"the root's own logits with {plant_scale:.6g} planted on the hidden state "
        f"at position {plant_position}, the row slot {slot} asks for",
        got_planted,
        got,
    )
