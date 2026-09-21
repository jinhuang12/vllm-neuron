"""The blockwise-FP8 dequant path for the four scaled MLA projections.

The prepared weight has to be the dequantised weight, a missing scale grid has to
refuse by name, and the end-to-end result has to match a torch oracle.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
    block_grid_shape,
)

#: The checkpoint's fp8 format: ``"fmt": "e4m3"`` in the config fixture.
_FP8 = torch.float8_e4m3fn

#: Largest finite magnitude of ``float8_e4m3fn``. Read off the dtype rather than
#: written as a literal, so a torch build with a different range moves it.
_FP8_MAX = float(torch.finfo(_FP8).max)

#: The checkpoint's declared block shape
#: (``vllm_neuron/model/glm5_next/quantization.py``). Cited by full path
#: because a bare ``quantization.py`` has two in-repo candidates -- the
#: ``llama3`` module carries one too -- and an ambiguous basename is a
#: mechanical drift hit.
BLOCK = (128, 128)

#: Token count, the same value the call site's acceptance uses.
DECLARED_SEQ = 128

#: The floor fixture. Chosen so all four scaled sites land on a ``[2, 2]`` grid
#: with both axes truncating a partial edge tile — the shape floor check 1
#: declares. Every width stays small enough to run on CPU.
FLOOR_OVERRIDES = {
    "hidden_size": 200,
    "num_attention_heads": 4,
    "q_lora_rank": 160,
    "kv_lora_rank": 130,
    "qk_nope_head_dim": 40,
    "qk_rope_head_dim": 0,
    "v_head_dim": 40,
}

#: The call site's tiny geometry, copied so check 6 runs where that
TINY_OVERRIDES = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
}

#: The standing block-dequant comparator, `### 3L.1`'s for the dense half.
BLOCK_DEQUANT_RTOL, BLOCK_DEQUANT_ATOL = 3e-2, 1e-5

#: The exact-scale arm, in its declared form: one scalar on both
#: terms.
EXACT_RTOL, EXACT_ATOL = 1e-5, 1e-5


def _impl():
    """Import the implementation inside a test body, never at module import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _seam():
    from vllm_neuron.functional.attention import mla_projections

    return mla_projections


def _config_module():
    from vllm_neuron.model.glm5_next import config

    return config


def real_config():
    return _config_module().Glm5NextTextConfig()


def floor_config():
    return dataclasses.replace(real_config(), **FLOOR_OVERRIDES)


def tiny_config():
    return dataclasses.replace(real_config(), **TINY_OVERRIDES)


def closed_form_widths(cfg) -> tuple[tuple[str, int, int], ...]:
    """The five sites as ``(name, in_features, out_features)``, computed here. """
    heads = int(cfg.num_attention_heads)
    qk_head_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)
    return (
        ("q_a_proj", int(cfg.hidden_size), int(cfg.q_lora_rank)),
        ("q_b_proj", int(cfg.q_lora_rank), heads * qk_head_dim),
        (
            "kv_a_proj_with_mqa",
            int(cfg.hidden_size),
            int(cfg.kv_lora_rank) + int(cfg.qk_rope_head_dim),
        ),
        (
            "kv_b_proj",
            int(cfg.kv_lora_rank),
            heads * (int(cfg.qk_nope_head_dim) + int(cfg.v_head_dim)),
        ),
        ("o_proj", heads * int(cfg.v_head_dim), int(cfg.hidden_size)),
    )


def _tiles(rows: int, cols: int):
    """Every tile's ``(grid_row, grid_col, row_slice, col_slice)``, explicitly. """
    block_rows, block_cols = BLOCK
    for gr in range((rows + block_rows - 1) // block_rows):
        for gc in range((cols + block_cols - 1) // block_cols):
            r0, c0 = gr * block_rows, gc * block_cols
            yield gr, gc, slice(r0, min(r0 + block_rows, rows)), slice(
                c0, min(c0 + block_cols, cols)
            )


def quantise_blockwise(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[rows, cols]`` fp32 to fp8 bytes plus a per-tile scale grid. """
    rows, cols = int(dense.shape[0]), int(dense.shape[1])
    grid = torch.zeros(block_grid_shape((rows, cols), BLOCK), dtype=torch.float32)
    out = torch.zeros((rows, cols), dtype=torch.float32)
    for gr, gc, rsl, csl in _tiles(rows, cols):
        tile = dense[rsl, csl]
        amax = float(tile.abs().max())
        scale = (amax / _FP8_MAX) if amax > 0.0 else 1.0
        grid[gr, gc] = scale
        out[rsl, csl] = (tile / scale).clamp(-_FP8_MAX, _FP8_MAX)
    return out.to(_FP8), grid


def oracle_dequantise(
    fp8_bytes: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    """Dequantise tile by tile — this file's own arithmetic, not the fork's. """
    dense = fp8_bytes.to(torch.float32)
    out = torch.empty_like(dense)
    rows, cols = int(dense.shape[0]), int(dense.shape[1])
    for gr, gc, rsl, csl in _tiles(rows, cols):
        out[rsl, csl] = dense[rsl, csl] * float(scale_inv[gr, gc])
    return out


def build_fp8_attention(
    cfg,
    seed: int = 850390,
    *,
    materialise: tuple[str, ...] | None = None,
    unit_scales: bool = False,
):
    """An MLA attention module whose four scaled projections hold fp8 bytes. """
    module = _impl().Glm5NextMLAAttention(cfg)
    gen = torch.Generator().manual_seed(seed)
    scaled = set(DSA_SCALED_PROJECTIONS)
    chosen = scaled if materialise is None else set(materialise)
    reference: dict[str, torch.Tensor] = {}

    for name, idim, odim in closed_form_widths(cfg):
        draw = torch.randn(odim, idim, generator=gen, dtype=torch.float32)
        draw = draw * (float(idim) ** -0.5)
        if name not in scaled:
            setattr(module, f"{name}_weight", torch.nn.Parameter(draw))
            reference[name] = draw.clone()
            continue
        if unit_scales:
            fp8_bytes = draw.to(_FP8)
            grid = torch.ones(
                block_grid_shape((odim, idim), BLOCK), dtype=torch.float32
            )
        else:
            fp8_bytes, grid = quantise_blockwise(draw)
        setattr(
            module,
            f"{name}_weight",
            torch.nn.Parameter(fp8_bytes, requires_grad=False),
        )
        if name in chosen:
            setattr(
                module,
                f"{name}_{FP8_SCALE_SUFFIX}",
                torch.nn.Parameter(grid, requires_grad=False),
            )
        reference[name] = oracle_dequantise(fp8_bytes, grid)

    for name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank) + int(cfg.qk_rope_head_dim)),
    ):
        gain = 1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
        setattr(module, name, torch.nn.Parameter(gain))
    return module, reference


def worst_relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest elementwise ``|got-want| / (|want| + atol)`` as a plain float. """
    return float(
        ((got - want).abs() / (want.abs() + BLOCK_DEQUANT_ATOL)).max().detach()
    )


def _prepared(module) -> dict[str, torch.Tensor]:
    return getattr(module, module.PREPARED_WEIGHTS_ATTR)


# --------------------------------------------------------------------------- #
# check 1
# --------------------------------------------------------------------------- #


def test_scale_grids_resolve_at_five_sites_above_the_floor() -> None:
    """check 1 of 7 — the four scale parameters exist, and at the shape floor. """
    cfg = floor_config()
    module, _reference = build_fp8_attention(cfg)
    declared = set(module.declared_param_names)
    scaled = set(DSA_SCALED_PROJECTIONS)


    sites, partial_axes = 0, 0
    for name, idim, odim in closed_form_widths(cfg):
        scale_name = f"{name}_{FP8_SCALE_SUFFIX}"
        if name not in scaled:
            assert scale_name not in declared, (
                f"{name} carries no scale companion in this checkpoint, so "
                f"{scale_name} must not be declared"
            )
            sites += 1
            continue

        assert scale_name in declared, f"{scale_name} is not declared"
        scale = getattr(module, scale_name)
        expected = block_grid_shape((odim, idim), BLOCK)

        assert tuple(scale.shape) == tuple(expected), (
            f"{scale_name} is {tuple(scale.shape)}; a {(odim, idim)} weight at "
            f"block size {BLOCK} needs {expected}"
        )
        assert expected[0] >= 2 and expected[1] >= 2, (
            f"{name}'s grid {expected} is below the [2, 2] floor, so check "
            f"2 and 4 would not be able to tell a per-block scale from a "
            f"global one"
        )
        on_edge = (odim % BLOCK[0] != 0) + (idim % BLOCK[1] != 0)
        partial_axes += on_edge
        assert on_edge >= 1, (
            f"{name} is {(odim, idim)}, a whole multiple of {BLOCK} on both "
            f"axes, so no partial edge tile is exercised at this site"
        )
        sites += 1

    assert sites == 5, f"five sites must be read; {sites} were"
    assert partial_axes > 0


# --------------------------------------------------------------------------- #
# check 2
# --------------------------------------------------------------------------- #


def test_prepared_weight_is_the_dequantised_weight() -> None:
    """check 2 of 7 — the prepared weight is the dequantised weight. """
    cfg = floor_config()
    module, reference = build_fp8_attention(cfg)

    count = module.prepare_projection_weights()
    prepared = _prepared(module)
    assert count == 5

    scaled_checked, worst_overall = 0, 0.0
    for name, _idim, _odim in closed_form_widths(cfg):
        want = reference[name].t().contiguous()
        got = prepared[name]
        assert got.dtype is torch.float32
        assert tuple(got.shape) == tuple(want.shape)

        if name in set(DSA_SCALED_PROJECTIONS):
            worst = worst_relative_error(got, want)
            worst_overall = max(worst_overall, worst)
            torch.testing.assert_close(
                got, want, rtol=BLOCK_DEQUANT_RTOL, atol=BLOCK_DEQUANT_ATOL
            )
            scaled_checked += 1
        else:
            raw = getattr(module, f"{name}_weight")
            expected = raw.to(torch.float32).t().contiguous()
            identical = torch.equal(got, expected)
            assert identical, (
                f"{name} is not scaled in this checkpoint, so its prepared "
                f"tensor must be exactly what the call site produced"
            )

    assert scaled_checked == 4, f"four scaled sites expected; got {scaled_checked}"


# --------------------------------------------------------------------------- #
# check 3
# --------------------------------------------------------------------------- #


def test_exact_scale_arm_passes_at_single_op_tolerance() -> None:
    """check 3 of 7 — with every scale exactly 1.0, agreement is exact. """
    cfg = floor_config()
    module, reference = build_fp8_attention(cfg, seed=850391, unit_scales=True)
    module.prepare_projection_weights()
    prepared = _prepared(module)

    checked, worst_overall = 0, 0.0
    for name in DSA_SCALED_PROJECTIONS:
        grid = getattr(module, f"{name}_{FP8_SCALE_SUFFIX}")
        assert bool((grid == 1.0).all()), f"{name}'s scales are not all 1.0"
        want = reference[name].t().contiguous()
        got = prepared[name]
        worst = worst_relative_error(got, want)
        worst_overall = max(worst_overall, worst)
        torch.testing.assert_close(got, want, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        checked += 1

    assert checked == 4


# --------------------------------------------------------------------------- #
# check 4
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# check 5
# --------------------------------------------------------------------------- #


def test_missing_scale_raises_a_named_refusal() -> None:
    """check 5 of 7 — the refusal is distinguishable from a skip. """
    cfg = floor_config()
    refused = 0
    for withheld in DSA_SCALED_PROJECTIONS:
        keep = tuple(n for n in DSA_SCALED_PROJECTIONS if n != withheld)
        module, _reference = build_fp8_attention(cfg, materialise=keep)
        scale_name = f"{withheld}_{FP8_SCALE_SUFFIX}"
        assert getattr(module, scale_name, None) is None

        with pytest.raises(ValueError) as caught:
            module.prepare_projection_weights()
        message = str(caught.value)

        assert withheld in message, (
            f"the refusal does not name the site; it said: {message}"
        )
        assert scale_name in message, (
            f"the refusal does not name the missing parameter {scale_name}; "
            f"it said: {message}"
        )
        refused += 1

    assert refused == 4


# --------------------------------------------------------------------------- #
# check 6
# --------------------------------------------------------------------------- #


def test_end_to_end_through_the_seam_matches_a_torch_oracle() -> None:
    """check 6 of 7 — the dequantised weight reaches the kernel. """
    MP = _seam()
    cfg = tiny_config()
    module, reference = build_fp8_attention(cfg, seed=850393)
    module.prepare_projection_weights()


    heads = int(cfg.num_attention_heads)
    gen = torch.Generator().manual_seed(6850)
    hidden = torch.randn(
        DECLARED_SEQ, int(cfg.hidden_size), generator=gen, dtype=torch.float32
    )
    attn_out = torch.randn(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), generator=gen, dtype=torch.float32
    )

    def norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + cfg.rms_norm_eps) * gain

    # The oracle: dequant, then a torch matmul per site. `mla_projection(x, W)`
    # computes `x @ W` on the prepared, already transposed weight, so on the
    # checkpoint orientation held here that is `x @ W.t`.
    q_latent = norm(hidden @ reference["q_a_proj"].t(), module.q_a_layernorm_weight)
    want_query = (q_latent @ reference["q_b_proj"].t()).reshape(
        DECLARED_SEQ, heads, -1
    )
    kv_latent = norm(
        hidden @ reference["kv_a_proj_with_mqa"].t(), module.kv_a_layernorm_weight
    )
    key_value = (kv_latent @ reference["kv_b_proj"].t()).reshape(
        DECLARED_SEQ, heads, int(cfg.qk_nope_head_dim) + int(cfg.v_head_dim)
    )
    want_key = key_value[..., : int(cfg.qk_nope_head_dim)].contiguous()
    want_value = key_value[..., int(cfg.qk_nope_head_dim) :].contiguous()
    want_out = attn_out.reshape(DECLARED_SEQ, -1) @ reference["o_proj"].t()

    MP.reset_mla_projection_dispatch_counters()
    query, key_nope, value = module.project_qkv(hidden)
    projected = module.project_output(attn_out)

    checked = 0
    for label, got, want in (
        ("query", query, want_query),
        ("key_nope", key_nope, want_key),
        ("value", value, want_value),
        ("output", projected, want_out),
    ):
        assert tuple(got.shape) == tuple(want.shape), (
            f"{label} is {tuple(got.shape)}, oracle says {tuple(want.shape)}"
        )
        torch.testing.assert_close(
            got, want, rtol=BLOCK_DEQUANT_RTOL, atol=BLOCK_DEQUANT_ATOL
        )
        checked += 1

    assert checked == 4


# --------------------------------------------------------------------------- #
# check 7
# --------------------------------------------------------------------------- #


def test_route_predicate_r2_five_dispatches_with_a_control() -> None:
    """The dequant path makes five kernel dispatches and no torch fallback."""
    MP = _seam()
    cfg = tiny_config()
    module, _reference = build_fp8_attention(cfg, seed=850394)
    module.prepare_projection_weights()

    heads = int(cfg.num_attention_heads)
    hidden = torch.zeros(DECLARED_SEQ, int(cfg.hidden_size), dtype=torch.float32)
    attn_out = torch.zeros(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), dtype=torch.float32
    )

    gate = MP.can_run_mla_projection(
        hidden, DECLARED_SEQ, int(cfg.hidden_size), int(cfg.q_lora_rank)
    )
    assert gate is True

    MP.reset_mla_projection_dispatch_counters()
    module.project_qkv(hidden)
    after_qkv, _ = MP.mla_projection_dispatch_counters()
    module.project_output(attn_out)
    nki_dispatch, torch_fallback = MP.mla_projection_dispatch_counters()


    assert after_qkv == 4, f"project_qkv must dispatch 4 times; it dispatched {after_qkv}"
    assert nki_dispatch - after_qkv == 1, "project_output must dispatch exactly once"
    assert nki_dispatch == 5, (
        f"R-2 requires 5 dispatches, one per projection site; the seam counted "
        f"{nki_dispatch}"
    )
    assert torch_fallback == 0

    # The control. A decoy that computes the projection without dispatching.
    real = MP.mla_projection
    calls = {"n": 0}

    def decoy(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return x.to(torch.float32) @ weight.to(torch.float32)

    MP.reset_mla_projection_dispatch_counters()
    MP.mla_projection = decoy
    try:
        module.project_qkv(hidden)
        module.project_output(attn_out)
        controlled, _ = MP.mla_projection_dispatch_counters()
    finally:
        MP.mla_projection = real


    assert calls["n"] == 5, (
        f"the decoy must stand in at all five sites, else the control is "
        f"partial; it was called {calls['n']} times"
    )
    assert controlled == 0, (
        f"substituting the seam must drive the dispatch count to 0; it read "
        f"{controlled}, so the counter is not measuring this call path"
    )
    assert MP.mla_projection is real, "the seam was not restored"

    MP.reset_mla_projection_dispatch_counters()
    module.project_qkv(hidden)
    module.project_output(attn_out)
    restored, _ = MP.mla_projection_dispatch_counters()
    assert restored == 5, (
        f"after restoring the seam the count must return to 5; it read "
        f"{restored}, so the 0 above was state that outlived the control"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
