"""`inc-glm53f-085` — the blockwise-FP8 dequant path for the four scaled MLA projections.

THE QUESTION THIS FILE ANSWERS. The checkpoint stores four of the five MLA
projections as blockwise-FP8 bytes with one scale per 128x128 tile.
`inc-glm53f-039b` prepared all five as ``weight.to(torch.float32).t()`` and
never applied those scales, so four projections computed the WRONG FUNCTION at
exactly the right shapes. The shapes are unchanged by the defect, so only a
numeric acceptance can see it.

THE THREE CASES, decided by the weight's dtype and never by a config flag: a
real dtype passes through as `-039b` landed it; fp8 bytes with their scale are
dequantised in the checkpoint's ``[out_features, in_features]`` orientation
BEFORE the transpose; fp8 bytes with NO scale raise, naming site and parameter.

THE ORACLE IS AUTHORED HERE, NOT IMPORTED. The fork's ``dequantise_blockwise``
broadcasts with ``repeat_interleave`` then slices; this file walks tiles and
multiplies each. Two different arithmetics agreeing is evidence; the same one
agreeing with itself is circular.

WHY CONJUNCTS 1-5 NEED THEIR OWN CONFIG. Conjunct 1 needs every grid at least
``[2, 2]`` with one axis truncating a partial edge tile, because a single block
cannot tell a per-block scale from one global scalar. The real geometry cannot
serve: its dimensions are all multiples of 128, so no partial tile exists, and
its weights reach 16384x1536. Hence ``FLOOR_OVERRIDES``, measured to put all
four scaled sites on a ``[2, 2]`` grid with BOTH axes partial.

Conjunct 6 runs at `-039b`'s ``TINY_OVERRIDES``, BELOW that floor, by design:
it certifies plumbing, and the arithmetic's non-vacuity is carried at the floor
fixture by conjuncts 1-4.

SEVEN CONJUNCTS, SEVEN COLLECTED ITEMS, and no parametrized case — one would
collect as several and the declared count would stop meaning what it says.
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

#: The checkpoint's declared block shape (``quantization.py:114``).
BLOCK = (128, 128)

#: Token count, the same value `-039b`'s landed acceptance uses.
DECLARED_SEQ = 128

#: The floor fixture. Chosen so all four scaled sites land on a ``[2, 2]`` grid
#: with BOTH axes truncating a partial edge tile — the shape floor conjunct 1
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

#: `-039b`'s landed tiny geometry, copied so conjunct 6 runs where that
#: increment's own numeric conjunct runs (``test_mla_projections.py:96-104``).
TINY_OVERRIDES = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
}

#: The standing block-dequant comparator, `### 3L.1`'s for `inc-glm53f-026`.
BLOCK_DEQUANT_RTOL, BLOCK_DEQUANT_ATOL = 3e-2, 1e-5

#: The exact-scale arm, in the form B42 elected: one declared scalar on both
#: terms.
EXACT_RTOL, EXACT_ATOL = 1e-5, 1e-5


def say(*parts: object) -> None:
    """Print one counted reading.

    pytest ``-q`` writes a progress dot with no trailing newline after each
    test, so every test after the first has that dot prefixed to its first
    printed line. Match with ``grep -o``, never with a ``^`` anchor.
    """
    print(" ".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation INSIDE a test body, never at module import."""
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
    """The five sites as ``(name, in_features, out_features)``, computed here.

    Re-derived in this file so every expectation below is compared against
    something the module did not produce.
    """
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
    """Every tile's ``(grid_row, grid_col, row_slice, col_slice)``, explicitly.

    A generator over the grid, written as arithmetic on tile indices. The last
    tile on an axis is clipped to the weight's real extent, which is what makes
    a partial edge tile a truncation rather than padding the weight up.
    """
    block_rows, block_cols = BLOCK
    for gr in range((rows + block_rows - 1) // block_rows):
        for gc in range((cols + block_cols - 1) // block_cols):
            r0, c0 = gr * block_rows, gc * block_cols
            yield gr, gc, slice(r0, min(r0 + block_rows, rows)), slice(
                c0, min(c0 + block_cols, cols)
            )


def quantise_blockwise(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[rows, cols]`` fp32 to fp8 bytes plus a per-tile scale grid.

    ``scale_inv`` is the DEQUANT multiplier, matching the checkpoint's ``_inv``
    suffix: dequantising multiplies and never divides.

    Per tile the scale is ``amax / FP8_MAX``, so each tile gets a visibly
    different scale. That tiles DIFFER is the point — a grid of equal entries
    would be a global scale wearing a grid's shape, and conjunct 4's control
    could not tell the two apart.
    """
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
    """Dequantise tile by tile — this file's OWN arithmetic, not the fork's.

    Indexed rather than broadcast on purpose, so agreement with the fork's
    ``repeat_interleave``-then-slice route is two arithmetics agreeing.
    """
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
    """An MLA attention module whose four scaled projections hold fp8 bytes.

    The four ``DSA_SCALED_PROJECTIONS`` leaves hold fp8 bytes plus a scale grid;
    ``kv_b_proj`` holds fp32, because this checkpoint keeps it in BF16 with no
    scale companion. Weights arrive in the checkpoint's own
    ``[out_features, in_features]`` orientation, which leaves the module's
    load-time transpose under test rather than assumed.

    Returns ``(module, reference)``. ``reference[name]`` is what the stored bytes
    and scales MEAN, by this file's own dequant — not the pre-quantisation draw,
    because quantisation is lossy and the draw is not what was stored.

    ``materialise`` names which scale parameters to set (default all four;
    conjunct 5 withholds one). ``unit_scales`` builds the exact-scale arm:
    weights already on the fp8 grid, every scale exactly 1.0, so the round trip
    is lossless.
    """
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
    """Largest elementwise ``|got-want| / (|want| + atol)`` as a plain float.

    Reported as a NUMBER beside every comparison, so a pass records how much
    margin it had and the positive control records how far it missed by.
    """
    return float(
        ((got - want).abs() / (want.abs() + BLOCK_DEQUANT_ATOL)).max().detach()
    )


def _prepared(module) -> dict[str, torch.Tensor]:
    return getattr(module, module.PREPARED_WEIGHTS_ATTR)


# --------------------------------------------------------------------------- #
# CONJUNCT 1
# --------------------------------------------------------------------------- #


def test_conjunct_1_scale_grids_resolve_at_five_sites_above_the_floor() -> None:
    """CONJUNCT 1 of 7 — the four scale parameters exist, and at the shape floor.

    CERTIFYING COMPONENT: the module-level ``_declare_parameters``
    (``model_fp8.py:87-102``, called from ``Glm5NextMLAAttention.__init__``) and
    ``DSA_SCALED_PROJECTIONS``.

    Five sites are read, not four: the fifth reading is that ``kv_b_proj``
    declares NO scale parameter. An absence asserted is a reading; an absence
    left unmentioned is a gap.

    THE FLOOR IS THE POINT. Below ``[2, 2]`` with a partial edge tile,
    conjuncts 2 and 4 lose their force, because one block cannot distinguish a
    per-block scale from one global scalar.
    """
    say("C1_CERTIFYING_COMPONENT=model_fp8._declare_parameters and "
        "DSA_SCALED_PROJECTIONS")
    cfg = floor_config()
    module, _reference = build_fp8_attention(cfg)
    declared = set(module.declared_param_names)
    scaled = set(DSA_SCALED_PROJECTIONS)

    say(f"C1_SCALED_LEAVES={sorted(scaled)}")
    say(f"C1_DECLARED_PARAM_COUNT={len(declared)}")

    sites, partial_axes = 0, 0
    for name, idim, odim in closed_form_widths(cfg):
        scale_name = f"{name}_{FP8_SCALE_SUFFIX}"
        if name not in scaled:
            assert scale_name not in declared, (
                f"{name} carries no scale companion in this checkpoint, so "
                f"{scale_name} must not be declared"
            )
            say(f"C1_SITE {name} DECLARES_NO_SCALE=True")
            sites += 1
            continue

        assert scale_name in declared, f"{scale_name} is not declared"
        scale = getattr(module, scale_name)
        expected = block_grid_shape((odim, idim), BLOCK)
        say(f"C1_SITE {name} weight={[odim, idim]} grid={list(scale.shape)} "
            f"expected={list(expected)}")

        assert tuple(scale.shape) == tuple(expected), (
            f"{scale_name} is {tuple(scale.shape)}; a {(odim, idim)} weight at "
            f"block size {BLOCK} needs {expected}"
        )
        assert expected[0] >= 2 and expected[1] >= 2, (
            f"{name}'s grid {expected} is below the [2, 2] floor, so conjuncts "
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

    say(f"C1_SITES_READ={sites}")
    say(f"C1_PARTIAL_EDGE_AXES={partial_axes}")
    assert sites == 5, f"five sites must be read; {sites} were"
    assert partial_axes > 0


# --------------------------------------------------------------------------- #
# CONJUNCT 2
# --------------------------------------------------------------------------- #


def test_conjunct_2_prepared_weight_is_the_dequantised_weight() -> None:
    """CONJUNCT 2 of 7 — the prepared weight IS the dequantised weight.

    CERTIFYING COMPONENT: ``prepare_projection_weights`` and
    ``_dequantised_projection_weight``.

    THIS IS THE DEFECT'S OWN TEST. Before this increment the prepared tensor was
    raw fp8 bytes cast to fp32 and transposed — right shape, wrong numbers.

    ``kv_b_proj`` is checked BYTE-IDENTICAL to `-039b`'s landed formula, which
    shows the repair is confined to the scaled sites.
    """
    say("C2_CERTIFYING_COMPONENT=prepare_projection_weights and "
        "_dequantised_projection_weight")
    cfg = floor_config()
    module, reference = build_fp8_attention(cfg)

    count = module.prepare_projection_weights()
    prepared = _prepared(module)
    say(f"C2_PREPARED_COUNT={count}")
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
            say(f"C2_SITE {name} WORST_REL_ERROR={worst:.6e}")
            torch.testing.assert_close(
                got, want, rtol=BLOCK_DEQUANT_RTOL, atol=BLOCK_DEQUANT_ATOL
            )
            scaled_checked += 1
        else:
            raw = getattr(module, f"{name}_weight")
            landed = raw.to(torch.float32).t().contiguous()
            identical = torch.equal(got, landed)
            say(f"C2_SITE {name} BYTE_IDENTICAL_TO_039B_TRANSPOSE={identical}")
            assert identical, (
                f"{name} is not scaled in this checkpoint, so its prepared "
                f"tensor must be exactly what inc-glm53f-039b produced"
            )

    say(f"C2_SCALED_SITES_CHECKED={scaled_checked}")
    say(f"C2_WORST_REL_ERROR_ANY_SITE={worst_overall:.6e}")
    assert scaled_checked == 4, f"four scaled sites expected; got {scaled_checked}"


# --------------------------------------------------------------------------- #
# CONJUNCT 3
# --------------------------------------------------------------------------- #


def test_conjunct_3_exact_scale_arm_passes_at_single_op_tolerance() -> None:
    """CONJUNCT 3 of 7 — with every scale exactly 1.0, agreement is exact.

    CERTIFYING COMPONENT: the dequant arithmetic in
    ``_dequantised_projection_weight``, isolated from quantisation error.

    WHY THIS ARM EXISTS. Conjunct 2 runs at ``3e-2``, wide enough to absorb fp8
    quantisation error — and therefore wide enough to absorb a small arithmetic
    mistake too. Here the weights already sit on the fp8 grid and every block
    scale is exactly 1.0, so the round trip is lossless and any error at all is
    the arithmetic's own. Tolerance is the elected single-op form,
    ``assert_close(rtol=1e-5, atol=1e-5)``.
    """
    say("C3_CERTIFYING_COMPONENT=the dequant arithmetic, isolated from "
        "quantisation error")
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
        say(f"C3_SITE {name} GRID={list(grid.shape)} WORST_REL_ERROR={worst:.6e}")
        torch.testing.assert_close(got, want, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        checked += 1

    say(f"C3_SITES_CHECKED={checked}")
    say(f"C3_WORST_REL_ERROR_ANY_SITE={worst_overall:.6e}")
    assert checked == 4


# --------------------------------------------------------------------------- #
# CONJUNCT 4
# --------------------------------------------------------------------------- #


def test_conjunct_4_positive_control_the_unscaled_formula_fails() -> None:
    """CONJUNCT 4 of 7 — THE POSITIVE CONTROL (D1.5).

    CERTIFYING COMPONENT: conjunct 2's own predicate, shown to be capable of
    failing.

    A pass means nothing until the same predicate is shown to FAIL on the defect
    it catches. The identical fixture is run through `-039b`'s landed formula —
    cast and transpose, no scale — and conjunct 2's comparison must reject it,
    with the worst relative error recorded as a number above ``3e-2``.

    A RUN IN WHICH THIS CONTROL PASSES IS A HOLLOW FIXTURE, NEVER A PASS: it
    would mean the scales were all near 1.0, so applying them changed nothing.
    """
    say("C4_CERTIFYING_COMPONENT=conjunct 2's predicate, shown able to fail")
    cfg = floor_config()
    module, reference = build_fp8_attention(cfg, seed=850392)

    failed, worst_overall = 0, 0.0
    for name in DSA_SCALED_PROJECTIONS:
        raw = getattr(module, f"{name}_weight")
        assert raw.dtype is _FP8, f"{name} must hold fp8 bytes for this control"
        # inc-glm53f-039b's landed formula, verbatim: cast and transpose, no scale.
        unscaled = raw.to(torch.float32).t().contiguous()
        want = reference[name].t().contiguous()
        worst = worst_relative_error(unscaled, want)
        worst_overall = max(worst_overall, worst)
        say(f"C4_SITE {name} UNSCALED_WORST_REL_ERROR={worst:.6e} "
            f"THRESHOLD={BLOCK_DEQUANT_RTOL}")
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                unscaled, want, rtol=BLOCK_DEQUANT_RTOL, atol=BLOCK_DEQUANT_ATOL
            )
        assert worst > BLOCK_DEQUANT_RTOL, (
            f"{name}'s unscaled error is {worst:.6e}, inside the {BLOCK_DEQUANT_RTOL} "
            f"comparator, so this fixture cannot tell the repair from the defect"
        )
        failed += 1

    say(f"C4_SITES_THE_CONTROL_REJECTED={failed}")
    say(f"C4_WORST_REL_ERROR_ANY_SITE={worst_overall:.6e}")
    assert failed == 4, "the control must be rejected at all four scaled sites"


# --------------------------------------------------------------------------- #
# CONJUNCT 5
# --------------------------------------------------------------------------- #


def test_conjunct_5_missing_scale_raises_a_named_refusal() -> None:
    """CONJUNCT 5 of 7 — the refusal is distinguishable from a skip.

    CERTIFYING COMPONENT: ``_dequantised_projection_weight``'s missing-scale
    branch.

    Silently treating unscaled fp8 bytes as numbers is the defect being
    repaired, so absence of a scale must be loud. The error has to NAME the site
    and the missing parameter — a bare failure would be indistinguishable from a
    test that never ran, and a caller could not tell which of the four sites is
    short.

    One site is withheld at a time, so all four report their own refusal rather
    than the loop stopping at whichever comes first.
    """
    say("C5_CERTIFYING_COMPONENT=_dequantised_projection_weight's "
        "missing-scale branch")
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
        say(f"C5_SITE {withheld} REFUSAL={message[:150]}")

        assert withheld in message, (
            f"the refusal does not name the site; it said: {message}"
        )
        assert scale_name in message, (
            f"the refusal does not name the missing parameter {scale_name}; "
            f"it said: {message}"
        )
        refused += 1

    say(f"C5_SITES_THAT_REFUSED={refused}")
    assert refused == 4


# --------------------------------------------------------------------------- #
# CONJUNCT 6
# --------------------------------------------------------------------------- #


def test_conjunct_6_end_to_end_through_the_seam_matches_a_torch_oracle() -> None:
    """CONJUNCT 6 of 7 — the dequantised weight reaches the kernel.

    CERTIFYING COMPONENT: the whole projection chain, ``project_qkv`` and
    ``project_output``, with fp8 weights on the four scaled sites.

    Conjuncts 2 and 3 prove the prepared TENSOR is right. This one proves the
    right tensor is what the seam actually multiplies with — a repair that fixed
    the weight and then failed to reach it would satisfy every conjunct above.

    Run at `-039b`'s ``TINY_OVERRIDES``, whose grids are ``[1, 2]``, ``[1, 1]``,
    ``[1, 2]`` and ``[2, 1]`` and so sit BELOW conjunct 1's floor. Permitted, and
    the plan says why: this conjunct certifies plumbing, and the arithmetic's
    non-vacuity is already carried at the floor fixture above.

    The oracle is a plain torch dequant-then-matmul chain written here, latent
    norms included, and it never calls the module.
    """
    say("C6_CERTIFYING_COMPONENT=project_qkv and project_output with fp8 weights")
    MP = _seam()
    cfg = tiny_config()
    module, reference = build_fp8_attention(cfg, seed=850393)
    module.prepare_projection_weights()

    grids = {
        name: list(getattr(module, f"{name}_{FP8_SCALE_SUFFIX}").shape)
        for name in DSA_SCALED_PROJECTIONS
    }
    say(f"C6_GRIDS={grids}")

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
    # checkpoint orientation held here that is `x @ W.t()`.
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
        worst = worst_relative_error(got, want)
        say(f"C6_{label.upper()}_WORST_REL_ERROR={worst:.6e}")
        torch.testing.assert_close(
            got, want, rtol=BLOCK_DEQUANT_RTOL, atol=BLOCK_DEQUANT_ATOL
        )
        checked += 1

    say(f"C6_OUTPUTS_CHECKED={checked}")
    assert checked == 4


# --------------------------------------------------------------------------- #
# CONJUNCT 7
# --------------------------------------------------------------------------- #


def test_conjunct_7_route_predicate_r2_five_dispatches_with_a_control() -> None:
    """CONJUNCT 7 of 7 — the route predicate's counted values (D13 form R-2).

    CERTIFYING COMPONENT: the seam `inc-glm53f-039a` authors in
    ``vllm_neuron/functional/attention/mla_projections.py``. This block authors
    no seam, which is D13's own ownership trigger.

    Needed because a dequant repair that quietly stopped reaching the kernel
    would still satisfy every numeric conjunct above.

    Declared values: 5 dispatches per forward pass — 4 from ``project_qkv``, 1
    from ``project_output`` — and ``can_run_mla_projection()`` True.

    THE CONTROL SUBSTITUTES THE SEAM AND MUST MOVE THE COUNT. A decoy computing
    the projection in torch without dispatching drives it to 0, which is exactly
    the failure this predicate exists to catch.

    WHERE THE PATCH HAS TO GO, measured not assumed: ``model_fp8`` imports
    ``mla_projection`` INSIDE each method body, so rebinding it on ``model_fp8``
    does nothing — the next call re-imports. The patch goes on the SEAM module's
    attribute, which is what that re-import resolves.

    ``torch_fallback`` is recorded but is a STRUCTURAL zero: it is incremented
    nowhere in the seam module, whose docstring says it "can only ever read 0,
    because this module has no torch projection route to increment it". It
    carries no non-vacuity weight; the 5-dispatch count and its control do.
    """
    say("C7_CERTIFYING_COMPONENT=the mla_projections seam authored by "
        "inc-glm53f-039a")
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
    say(f"C7_GATE_CAN_RUN_MLA_PROJECTION={gate}")
    assert gate is True

    MP.reset_mla_projection_dispatch_counters()
    module.project_qkv(hidden)
    after_qkv, _ = MP.mla_projection_dispatch_counters()
    module.project_output(attn_out)
    nki_dispatch, torch_fallback = MP.mla_projection_dispatch_counters()

    say(f"C7_DISPATCHES_FROM_PROJECT_QKV={after_qkv}")
    say(f"C7_DISPATCHES_FROM_PROJECT_OUTPUT={nki_dispatch - after_qkv}")
    say(f"C7_SEAM_NKI_DISPATCH={nki_dispatch}")
    say(f"C7_SEAM_TORCH_FALLBACK={torch_fallback} STRUCTURAL_ZERO=True")

    assert after_qkv == 4, f"project_qkv must dispatch 4 times; it dispatched {after_qkv}"
    assert nki_dispatch - after_qkv == 1, "project_output must dispatch exactly once"
    assert nki_dispatch == 5, (
        f"R-2 requires 5 dispatches, one per projection site; the seam counted "
        f"{nki_dispatch}"
    )
    assert torch_fallback == 0

    # THE CONTROL. A decoy that computes the projection without dispatching.
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

    say(f"C7_CONTROL_DECOY_CALLS={calls['n']}")
    say(f"C7_CONTROL_SEAM_NKI_DISPATCH={controlled}")
    say(f"C7_CONTROL_MOVED_THE_COUNT={controlled != nki_dispatch}")
    say(f"C7_SEAM_RESTORED={MP.mla_projection is real}")

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
    say(f"C7_AFTER_RESTORE_SEAM_NKI_DISPATCH={restored}")
    assert restored == 5, (
        f"after restoring the seam the count must return to 5; it read "
        f"{restored}, so the 0 above was state that outlived the control"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
