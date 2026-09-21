"""The MLA decode path: absorbed projections, the paged latent bank, sparse attention.

A decode step is compared against the matching slice of a prefill of the same
tokens, and against a token-count-independent fp32 oracle.
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------- #
# the declared geometry. Every value is the checkpoint's, and the two that the route
# predicate actually reads are named as such, because a reader needs to know which
# numbers may not be shrunk for speed.
DECLARED_HEADS = 64
LATENT_RANK = 512          # Kv_lora_rank. An exact fit: 512 % 128 == 0 and 512 <= 512,
                           # which is what makes the tiled counter read 0.
NOPE_WIDTH = 256           # qk_nope_head_dim
V_WIDTH = 256              # v_head_dim
ROPE_WIDTH = 0             # qk_rope_head_dim on this checkpoint
CONTEXT_ROWS = 2048        # prior tokens already in the cache
SELECTED_ROWS = 2048       # the production top-k count. > 512, which is what makes
                           # the row-tiled counter read 1.
DECODE_STEPS = 3
BATCH = 1

#: The registered pair for (iv-b), unchanged from the block's original registration.
RTOL = 1e-2
ATOL_IV_B = 1e-5

#: bf16 carries 8 mantissa bits, so one unit at magnitude m is about m * 2**-8. The
#: ruled tolerance for test (b) is two of those units.
BF16_MANTISSA_BITS = 8
BF16_UNITS_ALLOWED = 2


def _model_module():
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def real_config():
    from vllm_neuron.model.glm5_next import config as cfg_mod

    return cfg_mod.Glm5NextTextConfig()


def declared_config():
    """The checkpoint's config, with the five widths this block depends on asserted. """
    cfg = real_config()
    assert int(cfg.num_attention_heads) == DECLARED_HEADS
    assert int(cfg.kv_lora_rank) == LATENT_RANK
    assert int(cfg.qk_nope_head_dim) == NOPE_WIDTH
    assert int(cfg.v_head_dim) == V_WIDTH
    assert int(cfg.qk_rope_head_dim) == ROPE_WIDTH
    return cfg


def build_attention(*, seed: int = 850390, weight_scale: float = 1.0):
    """One MLA attention layer with dense random weights, load-time preps run. """
    model_fp8 = _model_module()
    cfg = declared_config()
    module = model_fp8.Glm5NextMLAAttention(cfg)
    gen = torch.Generator().manual_seed(seed)
    raw: dict[str, torch.Tensor] = {}
    for name, in_features, out_features in module.projection_widths():
        weight = (
            torch.randn(out_features, in_features, generator=gen, dtype=torch.float32)
            * (in_features ** -0.5)
            * weight_scale
        )
        raw[name] = weight
        setattr(module, f"{name}_weight", torch.nn.Parameter(weight))
    gains: dict[str, torch.Tensor] = {}
    for name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
    ):
        gain = 1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
        gains[name] = gain
        setattr(module, name, torch.nn.Parameter(gain))
    module.prepare_projection_weights()
    prepared_absorb = module.prepare_absorb_weights()
    assert prepared_absorb == 2
    return module, raw, gains, gen


def _counter_modules():
    from vllm_neuron.functional.attention import mla_absorb, mla_sparse

    return mla_sparse, mla_absorb


def reset_counters() -> None:
    sparse, absorb = _counter_modules()
    sparse.reset_mla_sparse_dispatch_counters()
    sparse.reset_mla_sparse_tiled_dispatch_counters()
    sparse.reset_mla_sparse_row_tiled_dispatch_counters()
    absorb.reset_mla_absorb_dispatch_counters()


def read_counters() -> dict[str, int]:
    """The five readings the route predicate names, each from its owning module. """
    sparse, absorb = _counter_modules()
    seam = sparse.mla_sparse_dispatch_counters()
    tiled = sparse.mla_sparse_tiled_dispatch_counters()
    row_tiled = sparse.mla_sparse_row_tiled_dispatch_counters()
    absorbed = absorb.mla_absorb_dispatch_counters()
    return {
        "sparse": seam[0],
        "tiled": tiled[0],
        "row_tiled": row_tiled[0],
        "absorb": absorbed[0],
        "torch_fallback": seam[1] + tiled[1] + row_tiled[1] + absorbed[1],
    }


def atol_for(reference: torch.Tensor) -> float:
    """The ruled tolerance: two bf16 units at the reference's own magnitude. """
    peak = float(reference.detach().reshape(-1).to(torch.float32).abs().max())
    return BF16_UNITS_ALLOWED * (2.0 ** -BF16_MANTISSA_BITS) * peak


def report_and_check(label: str, got: torch.Tensor, reference: torch.Tensor) -> None:
    """The decode verdict, with both measured numbers beside it.
    """
    flat_got = got.detach().reshape(-1).to(torch.float32)
    flat_ref = reference.detach().reshape(-1).to(torch.float32)
    atol = atol_for(reference)
    torch.testing.assert_close(
        flat_got, flat_ref, rtol=RTOL, atol=atol,
        msg=lambda m: f"{label}: {m}",
    )


PAGE = 128                 # the bank's block size here: one staging piece per block


def paged_operands(cache, start, tokens, *, page=PAGE):
    """The three paged operands for a bank a request holds every block of, in order. """
    pages = int(cache.shape[0]) // int(page)
    return {
        "block_table_row": torch.arange(pages, dtype=torch.int32).reshape(pages, 1),
        "latent_slots": torch.arange(start, start + tokens, dtype=torch.int64),
        "page_size": int(page),
    }


def seeded_cache(module, gen, *, rows: int = CONTEXT_ROWS, steps: int = DECODE_STEPS):
    """The latent cache, seeded directly. """
    slots = -(-(rows + steps) // PAGE) * PAGE
    cache = torch.zeros(
        (slots, module.NUM_LATENT_KV_HEADS, module.head_size),
        dtype=torch.bfloat16,
    )
    cache[:rows, 0, :] = torch.randn(
        (rows, module.head_size), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    return cache


def decode_inputs(module, gen, *, steps: int = DECODE_STEPS, self_inclusive: bool = True):
    """Hidden states, and a selection of prior rows plus each step's own cache slot. """
    hidden = (
        torch.randn((steps, module.hidden_size), generator=gen, dtype=torch.float32)
        * 0.5
    ).to(torch.bfloat16)
    priors = SELECTED_ROWS - 1 if self_inclusive else SELECTED_ROWS
    order = torch.randperm(CONTEXT_ROWS, generator=gen)[:priors].to(torch.int32)
    if self_inclusive:
        rows = [
            torch.cat([order, torch.tensor([CONTEXT_ROWS + step], dtype=torch.int32)])
            for step in range(steps)
        ]
        selection = torch.stack(rows)
    else:
        selection = order.unsqueeze(0).repeat(steps, 1)
    # Asserted rather than trusted: the width the seam requires, and -- for the
    # self-inclusive form -- that step s carries its own slot exactly once and carries no
    # other step's slot, which is the property that keeps (b-i) causally sound.
    assert tuple(selection.shape) == (steps, SELECTED_ROWS)
    if self_inclusive:
        for step in range(steps):
            own = CONTEXT_ROWS + step
            assert int((selection[step] == own).sum()) == 1
            others = [CONTEXT_ROWS + other for other in range(steps) if other != step]
            for other in others:
                assert int((selection[step] == other).sum()) == 0
        assert int(selection.max()) == CONTEXT_ROWS + steps - 1
    else:
        assert int(selection.max()) < CONTEXT_ROWS
    scale = float(NOPE_WIDTH) ** -0.5
    return hidden, selection, scale


def run_decode_steps(
    module, cache, hidden, selection, scale, *, start=CONTEXT_ROWS
):
    """One token at a time, cache growing between calls. The path under test."""
    out = []
    for step in range(int(hidden.shape[0])):
        out.append(
            module.attend(
                hidden[step : step + 1],
                cache,
                start + step,
                selection[step : step + 1],
                scale,
                batch_size=BATCH,
                **paged_operands(cache, start + step, 1),
            )
        )
    return torch.cat(out, dim=0)


def fp32_oracle(module, raw, gains, cache_rows, hidden, selection, scale):
    """The whole chain in pure fp32, per row, so it cannot depend on token count. """
    from vllm_neuron.functional.attention.mla_absorb import mla_absorb_torch_oracle
    from vllm_neuron.functional.attention.mla_sparse import (
        mla_sparse_attention_torch_oracle,
    )

    def norm(x, gain):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + module.rms_norm_eps) * gain

    x = hidden.to(torch.float32)
    tokens = int(x.shape[0])
    q_latent = norm(x @ raw["q_a_proj"].T, gains["q_a_layernorm_weight"])
    query = (q_latent @ raw["q_b_proj"].T).reshape(tokens, DECLARED_HEADS, NOPE_WIDTH)
    q_lift = mla_absorb_torch_oracle(query, module._absorb_weight("W_UK"))
    attended = mla_sparse_attention_torch_oracle(q_lift, cache_rows, selection, scale)
    reduced = mla_absorb_torch_oracle(
        attended.to(torch.float32), module._absorb_weight("W_UV")
    )
    flat = reduced.reshape(tokens, DECLARED_HEADS * V_WIDTH)
    return flat @ raw["o_proj"].T


def oracle_latent(module, raw, gains, hidden):
    """The normalised KV latent, in fp32, matching what the path writes to the cache."""

    def norm(x, gain):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + module.rms_norm_eps) * gain

    x = hidden.to(torch.float32)
    return norm(x @ raw["kv_a_proj_with_mqa"].T, gains["kv_a_layernorm_weight"])


def test_the_absorb_split_is_exact_for_all_heads() -> None:
    """`W_UK` and `W_UV` are the prepared `kv_b_proj`'s two halves, bit-identical. """
    module, _, _, _ = build_attention()
    prepared = module._prepared_weight("kv_b_proj")
    assert tuple(prepared.shape) == (
        LATENT_RANK,
        DECLARED_HEADS * (NOPE_WIDTH + V_WIDTH),
    )
    w_uk = module._absorb_weight("W_UK")
    w_uv = module._absorb_weight("W_UV")
    assert tuple(w_uk.shape) == (DECLARED_HEADS, NOPE_WIDTH, LATENT_RANK)
    assert tuple(w_uv.shape) == (DECLARED_HEADS, LATENT_RANK, V_WIDTH)

    stride = NOPE_WIDTH + V_WIDTH
    mismatched_uk = 0
    mismatched_uv = 0
    for head in range(DECLARED_HEADS):
        lo = head * stride
        want_uk = prepared[:, lo : lo + NOPE_WIDTH].T
        want_uv = prepared[:, lo + NOPE_WIDTH : lo + stride]
        if not torch.equal(w_uk[head], want_uk):
            mismatched_uk += 1
        if not torch.equal(w_uv[head], want_uv):
            mismatched_uv += 1
    assert mismatched_uk == 0
    assert mismatched_uv == 0

    perturbed = w_uk[0].clone()
    perturbed[0, 0] += 1.0
    control_perturbed_rejected = not torch.equal(
        perturbed, prepared[:, :NOPE_WIDTH].T
    )
    control_untransposed_rejected = not torch.equal(
        w_uk[0], prepared[:, :NOPE_WIDTH]
    )
    assert control_perturbed_rejected
    assert control_untransposed_rejected


# --------------------------------------------------------------------------- #
def test_the_split_reproduces_the_dense_expansion() -> None:
    """Per head, the absorbed operands reproduce the expansion's key and value.
    """
    module, _, _, gen = build_attention()
    tokens = 4

    # The hidden states are fp32 here, deliberately, and the first attempt at this test
    # got it wrong in a way worth recording. Both methods return their results cast to
    # the input's dtype. With bf16 hidden states, the latent this test reads back is a
    # Bf16 round of the latent `project_qkv` expanded internally in fp32 -- so the test
    # compared an expansion of one latent against an expansion of a slightly different
    # one. It failed at 90 of 1,024 elements with a greatest relative difference of
    # 1.33, which is far too large to be reassociation and was the clue: a 0.4 %
    # perturbation of the input, amplified at output elements that cancel.
    #
    # In fp32 both methods share the same first three dispatches on the same input, and
    # those dispatches are deterministic, so the latent this test expands is the latent
    # `project_qkv` expanded. The query check below is what proves that shared prefix
    # rather than assuming it.
    hidden = (
        torch.randn((tokens, module.hidden_size), generator=gen, dtype=torch.float32)
        * 0.5
    )
    query_dense, key_nope, value = (t.detach() for t in module.project_qkv(hidden))

    query_absorbed, latent = (
        t.detach() for t in module.project_query_and_latent(hidden)
    )
    latent32 = latent.to(torch.float32)
    assert tuple(latent32.shape) == (tokens, LATENT_RANK)

    # The shared-prefix proof. The two methods are line-for-line identical from their
    # `x = hidden_states.to(float32)` through `q_a_proj`, the q norm, `q_b_proj`, the
    # reshape, `kv_a_proj_with_mqa` and the kv norm; they diverge only at whether the
    # latent is then expanded. So equal queries and an equal latent follow from the same
    # fact. Reading that off the source is not a reading, so the equality below is the
    # runtime proof: if these dispatches were not deterministic on one input, or if the
    # two chains had drifted apart, this is what would catch it.
    shared_prefix_identical = torch.equal(query_dense, query_absorbed)
    assert shared_prefix_identical

    w_uk = module._absorb_weight("W_UK").detach().to(torch.float32)
    w_uv = module._absorb_weight("W_UV").detach().to(torch.float32)
    worst_key = 0.0
    worst_value = 0.0
    for head in range(DECLARED_HEADS):
        got_key = latent32 @ w_uk[head].T
        got_value = latent32 @ w_uv[head]
        want_key = key_nope[:, head].to(torch.float32)
        want_value = value[:, head].to(torch.float32)
        torch.testing.assert_close(got_key, want_key, rtol=RTOL, atol=ATOL_IV_B)
        torch.testing.assert_close(got_value, want_value, rtol=RTOL, atol=ATOL_IV_B)
        worst_key = max(worst_key, float((got_key - want_key).abs().max()))
        worst_value = max(worst_value, float((got_value - want_value).abs().max()))

    # The comparison must be able to reject, or 128 passing assertions above mean
    # nothing. One element of the expected tensor is moved.
    spoiled = key_nope[:, 0].to(torch.float32).clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        latent32 @ w_uk[0].T, spoiled, rtol=RTOL, atol=ATOL_IV_B
    )
    assert fired


# --------------------------------------------------------------------------- #
def test_decode_matches_the_prefill_slice_for_every_step() -> None:
    """A decode step equals the prefill run's matching row, 3/3, at the tolerance. """
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    prefill = module.attend(
        hidden, cache.clone(), CONTEXT_ROWS, selection, scale, batch_size=BATCH,
        **paged_operands(cache, CONTEXT_ROWS, int(hidden.shape[0])),
    )
    decode = run_decode_steps(module, cache.clone(), hidden, selection, scale)
    assert tuple(decode.shape) == tuple(prefill.shape)

    for step in range(DECODE_STEPS):
        report_and_check(
            f"B_I_STEP_{step}", decode[step : step + 1], prefill[step : step + 1]
        )

    spoiled = prefill.to(torch.float32).clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        decode.to(torch.float32), spoiled, rtol=RTOL, atol=atol_for(prefill)
    )
    assert fired


# --------------------------------------------------------------------------- #
def test_decode_matches_a_token_count_independent_fp32_oracle() -> None:
    """A decode step equals a pure-fp32, per-row torch oracle, 3/3 steps. """
    module, raw, gains, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    decode = run_decode_steps(module, cache.clone(), hidden, selection, scale)

    oracle_cache = cache.clone()
    oracle_cache[CONTEXT_ROWS : CONTEXT_ROWS + DECODE_STEPS, 0, :] = oracle_latent(
        module, raw, gains, hidden
    ).to(torch.bfloat16)
    rows = oracle_cache[: CONTEXT_ROWS + DECODE_STEPS, 0, :].to(torch.float32)
    reference = fp32_oracle(module, raw, gains, rows, hidden, selection, scale)

    for step in range(DECODE_STEPS):
        report_and_check(
            f"B_II_STEP_{step}", decode[step : step + 1], reference[step : step + 1]
        )

    # The oracle must be able to disagree, or it is not an independent reference.
    spoiled = reference.clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        decode.to(torch.float32), spoiled, rtol=RTOL, atol=atol_for(reference)
    )
    assert fired


# --------------------------------------------------------------------------- #
def test_the_written_latent_reads_back_bit_identical() -> None:
    """The latent step ``s`` wrote is in slot ``CONTEXT_ROWS + s``, bit for bit. """
    module, _, _, gen = build_attention()
    # One spare slot past the three the path writes, so control 2 has something untouched
    # to read. The whole window is read -- a length taken from the position would pin a
    # captured graph to one position -- and the spare row is still never gathered:
    # `decode_inputs` selects prior context rows plus each step's own slot and nothing
    # else, so no selection names it. The bank holds whole blocks, so the rows past the
    # spare are untouched too; this reads the first of them, the one the step count
    # names, because that is the row a write running off its slot would land in.
    cache = seeded_cache(module, gen, steps=DECODE_STEPS + 1)
    hidden, selection, scale = decode_inputs(module, gen)
    spare = CONTEXT_ROWS + DECODE_STEPS

    sentinel = torch.full((module.head_size,), -7.0, dtype=torch.bfloat16)
    for step in range(DECODE_STEPS):
        cache[CONTEXT_ROWS + step, 0, :] = sentinel
    planted = sum(
        int(torch.equal(cache[CONTEXT_ROWS + step, 0, :], sentinel))
        for step in range(DECODE_STEPS)
    )
    assert planted == DECODE_STEPS
    assert torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))

    # Not a clone. The whole point is to inspect the tensor the path wrote into.
    run_decode_steps(module, cache, hidden, selection, scale)

    survived = sum(
        int(torch.equal(cache[CONTEXT_ROWS + step, 0, :], sentinel))
        for step in range(DECODE_STEPS)
    )
    assert survived == 0

    agreeing = 0
    for step in range(DECODE_STEPS):
        want = module.project_query_and_latent(hidden[step : step + 1])[1]
        want_bf16 = want.to(cache.dtype)[0]
        got = cache[CONTEXT_ROWS + step, 0, :]
        same = bool(torch.equal(got, want_bf16))
        assert same
        agreeing += 1
    assert agreeing == DECODE_STEPS

    assert torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))

    moved = module.project_query_and_latent(hidden[0:1])[1].to(cache.dtype)[0].clone()
    moved[0] = moved[0] + 1.0
    fired = not torch.equal(cache[CONTEXT_ROWS, 0, :], moved)
    assert fired


# --------------------------------------------------------------------------- #
def test_how_much_the_own_slot_moves_the_output() -> None:
    """A measurement, not a criterion: is the self-inclusive selection enough on its own?
    """
    # One module for both arms. `attend` mutates only the cache passed to it and reads
    # fixed weights, so a second build would cost two load-time prep passes and buy
    # nothing; and the dispatch counters live in the seam modules, not here.
    module, _, _, _ = build_attention()
    # Two generators on one seed, so hidden states and the prior permutation are the same
    # draw on both arms and the only difference is the single swapped column.
    gen_a = torch.Generator().manual_seed(4242)
    gen_b = torch.Generator().manual_seed(4242)
    hidden_a, sel_inclusive, scale = decode_inputs(module, gen_a)
    hidden_b, sel_prior_only, _ = decode_inputs(module, gen_b, self_inclusive=False)

    assert torch.equal(hidden_a, hidden_b)
    differing_columns = int((sel_inclusive != sel_prior_only).sum(dim=1).max())
    assert differing_columns == 1
    assert int(sel_inclusive[0].max()) == CONTEXT_ROWS
    assert int(sel_prior_only[0].max()) < CONTEXT_ROWS

    cache_inclusive = seeded_cache(module, torch.Generator().manual_seed(99001))
    cache_prior_only = seeded_cache(module, torch.Generator().manual_seed(99001))
    assert torch.equal(cache_inclusive, cache_prior_only)

    run_decode_steps(
        module, cache_inclusive, hidden_a, sel_inclusive, scale
    )
    run_decode_steps(
        module, cache_prior_only, hidden_a, sel_prior_only, scale
    )


# --------------------------------------------------------------------------- #
def test_a_larger_batch_raises_a_named_error_first() -> None:
    """`B > 1` raises `Glm5NextMLADecodeError`, and nothing dispatches. """
    model_fp8 = _model_module()
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    reset_counters()
    with pytest.raises(model_fp8.Glm5NextMLADecodeError) as caught:
        module.attend(
            hidden, cache.clone(), CONTEXT_ROWS, selection, scale, batch_size=2,
            **paged_operands(cache, CONTEXT_ROWS, int(hidden.shape[0])),
        )
    message = " ".join(str(caught.value).split())
    assert "batch_size" in message

    after = read_counters()
    for name, value in after.items():
        assert value == 0, f"{name} moved on a refusal: {after}"

    module.attend(hidden[:1], cache.clone(), CONTEXT_ROWS, selection[:1], scale,
                  batch_size=BATCH, **paged_operands(cache, CONTEXT_ROWS, 1))
    admissible = read_counters()
    assert admissible["sparse"] == 1
    assert admissible["absorb"] == 2

    assert issubclass(model_fp8.Glm5NextMLADecodeError, ValueError)


# --------------------------------------------------------------------------- #
def test_route_predicate_counter_readings_on_every_arm() -> None:
    """Which kernels ran, counted, on the decode arm and the prefill reference arm. """
    sparse, _ = _counter_modules()
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    # The predicates are derived from the seam's own constants, so the expected counter
    # values and the kernel actually chosen cannot disagree.
    tiled = LATENT_RANK % sparse.LATENT_TILE != 0 or LATENT_RANK > sparse.MOVING_MAX
    rows_tiled = SELECTED_ROWS > sparse.MOVING_MAX
    assert tiled is False
    assert rows_tiled is True

    reset_counters()
    run_decode_steps(module, cache.clone(), hidden, selection, scale)
    decode_counts = read_counters()
    assert decode_counts == {
        "sparse": DECODE_STEPS * 1,
        "tiled": 0,
        "row_tiled": DECODE_STEPS * 1,
        "absorb": DECODE_STEPS * 2,
        "torch_fallback": 0,
    }

    reset_counters()
    module.attend(hidden, cache.clone(), CONTEXT_ROWS, selection, scale,
                  batch_size=BATCH,
                  **paged_operands(cache, CONTEXT_ROWS, int(hidden.shape[0])))
    prefill_counts = read_counters()
    assert prefill_counts == {
        "sparse": 1,
        "tiled": 0,
        "row_tiled": 1,
        "absorb": 2,
        "torch_fallback": 0,
    }

    reset_counters()
    ragged = LATENT_RANK + sparse.LATENT_TILE // 2
    assert ragged % sparse.LATENT_TILE != 0
    control_topk = sparse.KEY_CHUNK
    assert control_topk % sparse.KEY_CHUNK == 0
    assert control_topk <= sparse.MOVING_MAX
    q_lift = torch.randn((1, 2, ragged), dtype=torch.float32) * 0.1
    c_kv = torch.randn((256, ragged), dtype=torch.float32) * 0.1
    idx = torch.zeros((1, control_topk), dtype=torch.int32)
    sparse.mla_sparse_attention(q_lift, c_kv, idx, 0.1)
    ragged_counts = read_counters()
    assert ragged_counts["tiled"] == 1
    assert ragged_counts["torch_fallback"] == 0

    reset_counters()
    assert sparse.MOVING_MAX % sparse.KEY_CHUNK == 0
    small_idx = torch.zeros((1, sparse.MOVING_MAX), dtype=torch.int32)
    q_small = torch.randn((1, 2, LATENT_RANK), dtype=torch.float32) * 0.1
    c_small = torch.randn((512, LATENT_RANK), dtype=torch.float32) * 0.1
    sparse.mla_sparse_attention(q_small, c_small, small_idx, 0.1)
    small_counts = read_counters()
    assert small_counts["row_tiled"] == 0
    assert small_counts["sparse"] == 1

    for accessor in (
        sparse.mla_sparse_dispatch_counters,
        sparse.mla_sparse_tiled_dispatch_counters,
        sparse.mla_sparse_row_tiled_dispatch_counters,
    ):
        reading = accessor()
        assert len(reading) == 2


PERRANK_WORLD = 2
#: 64 heads over 2 ranks. An exact split, asserted below rather than assumed: a
#: world size that did not divide the head count would floor, and then no rank's
#: heads would sum back to the model's.
PERRANK_HEADS = DECLARED_HEADS // PERRANK_WORLD

#: A world size that does not divide 64, for the flooring control in test (1).
PERRANK_UNEVEN_WORLD = 3


def _patch_world(monkeypatch, world_size: int) -> None:
    """Run the module at a synthetic world size, at the resolver the code reads. """
    monkeypatch.setattr(_model_module(), "_resolve_world_size", lambda: world_size)


def _bare_attention():
    """One MLA layer with no weights installed, at the checkpoint's real geometry. """
    return _model_module().Glm5NextMLAAttention(declared_config())


def _install(module, weights: dict, gains: dict) -> None:
    """Install a given weight set and run the two load-time preparations. """
    for name, weight in weights.items():
        setattr(module, f"{name}_weight", torch.nn.Parameter(weight))
    for name, gain in gains.items():
        setattr(module, name, torch.nn.Parameter(gain))
    module.prepare_projection_weights()
    assert module.prepare_absorb_weights() == 2


def _rank_slices(raw: dict, rank: int, world_size: int) -> dict:
    """One rank's share of a full weight set, cut at the ratified geometry. """
    heads = DECLARED_HEADS // world_size
    q_rows = heads * (NOPE_WIDTH + ROPE_WIDTH)
    kv_rows = heads * (NOPE_WIDTH + V_WIDTH)
    o_cols = heads * V_WIDTH
    return {
        # Replicated: MLA keeps one compressed latent per token, not one per head,
        # so neither latent projection has a head axis to split.
        "q_a_proj": raw["q_a_proj"].clone(),
        "kv_a_proj_with_mqa": raw["kv_a_proj_with_mqa"].clone(),
        "q_b_proj": raw["q_b_proj"][rank * q_rows : (rank + 1) * q_rows, :].clone(),
        "kv_b_proj": raw["kv_b_proj"][rank * kv_rows : (rank + 1) * kv_rows, :].clone(),
        "o_proj": raw["o_proj"][:, rank * o_cols : (rank + 1) * o_cols].clone(),
    }


def _reset_projection_counters() -> None:
    """Reset the projection seam's own dispatch counters. """
    from vllm_neuron.functional.attention import mla_projections

    mla_projections.reset_mla_projection_dispatch_counters()


def _read_projection_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the projection seam since the reset. """
    from vllm_neuron.functional.attention import mla_projections

    return mla_projections.mla_projection_dispatch_counters()


class _CountedTwoRankGroup:
    """The injected tensor-parallel coordinator: it counts, and it really sums. """

    def __init__(self, world_size: int = PERRANK_WORLD) -> None:
        self.world_size = world_size
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []
        self.recording = True
        self._recorded: list[torch.Tensor] = []
        self._replayed = 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(tensor.shape))
        if self.recording:
            self._recorded.append(tensor.detach().clone())
        else:
            tensor.add_(self._recorded[self._replayed])
            self._replayed += 1
        return tensor


def _run_sharded(group, raw, gains, cache, hidden, selection, scale,
                 *, doctor_rank_1_o_proj_by_one_head: bool = False):
    """Both ranks, in order, through the production decode path. """
    outputs = {}
    counters = []
    for rank in range(PERRANK_WORLD):
        group.recording = rank == 0
        module = _bare_attention()
        assert module._heads_per_rank() == PERRANK_HEADS
        weights = _rank_slices(raw, rank, PERRANK_WORLD)
        if doctor_rank_1_o_proj_by_one_head and rank == 1:
            # The doctored slice: the right shape, taken one head too early. This
            # is the defect head partitioning actually risks -- a rank that
            # computes head 32's values and multiplies them by head 31's output
            # columns -- and no shape check can see it, which is why the numeric
            # criterion has to.
            cols = PERRANK_HEADS * V_WIDTH
            low = rank * cols - V_WIDTH
            weights["o_proj"] = raw["o_proj"][:, low : low + cols].clone()
        _install(module, weights, gains)
        _reset_projection_counters()
        outputs[rank] = run_decode_steps(
            module, cache.clone(), hidden, selection, scale
        )
        counters.append(_read_projection_counters())
    return outputs[PERRANK_WORLD - 1], counters


# --------------------------------------------------------------------------- #
def test_perrank_widths_only_the_head_bearing_projections_narrow(monkeypatch) -> None:
    """Conjunct (1): the three head-width projections halve, the other widths do not.
    """
    model_fp8 = _model_module()
    cfg = declared_config()
    module = _bare_attention()
    q_head_width = NOPE_WIDTH + ROPE_WIDTH
    kv_head_width = NOPE_WIDTH + V_WIDTH

    def widths() -> dict[str, tuple[int, int]]:
        return {name: (idim, odim) for name, idim, odim in module.projection_widths()}

    # -- world size 1: byte-identical to what this file measured before this block.
    assert module._heads_per_rank() == DECLARED_HEADS
    at_one = widths()
    expected_one = {
        "q_a_proj": (int(cfg.hidden_size), int(cfg.q_lora_rank)),
        "q_b_proj": (int(cfg.q_lora_rank), DECLARED_HEADS * q_head_width),
        "kv_a_proj_with_mqa": (int(cfg.hidden_size), LATENT_RANK + ROPE_WIDTH),
        "kv_b_proj": (LATENT_RANK, DECLARED_HEADS * kv_head_width),
        "o_proj": (DECLARED_HEADS * V_WIDTH, int(cfg.hidden_size)),
    }
    assert at_one == expected_one

    # -- world size 2: the three head-bearing widths halve and nothing else moves.
    _patch_world(monkeypatch, PERRANK_WORLD)
    assert PERRANK_HEADS * PERRANK_WORLD == DECLARED_HEADS, (
        "the declared head count does not divide by the declared world size, so "
        "no rank's heads would sum back to the model's"
    )
    assert module._heads_per_rank() == PERRANK_HEADS
    at_two = widths()
    expected_two = {
        "q_a_proj": (int(cfg.hidden_size), int(cfg.q_lora_rank)),
        "q_b_proj": (int(cfg.q_lora_rank), PERRANK_HEADS * q_head_width),
        "kv_a_proj_with_mqa": (int(cfg.hidden_size), LATENT_RANK + ROPE_WIDTH),
        "kv_b_proj": (LATENT_RANK, PERRANK_HEADS * kv_head_width),
        "o_proj": (PERRANK_HEADS * V_WIDTH, int(cfg.hidden_size)),
    }
    assert at_two == expected_two

    # Which widths moved, stated as a reading rather than left to the dicts above.
    moved = sorted(name for name in at_one if at_one[name] != at_two[name])
    assert moved == ["kv_b_proj", "o_proj", "q_b_proj"]
    assert at_one["q_a_proj"] == at_two["q_a_proj"]
    assert at_one["kv_a_proj_with_mqa"] == at_two["kv_a_proj_with_mqa"]

    # Rank invariant. A width is a function of the world size alone; if it read the
    # rank, two ranks would expect different shapes from one checkpoint.
    monkeypatch.setattr(model_fp8, "_resolve_rank", lambda: 1)
    assert widths() == at_two

    _patch_world(monkeypatch, PERRANK_UNEVEN_WORLD)
    uneven_heads = DECLARED_HEADS // PERRANK_UNEVEN_WORLD
    at_three = widths()
    assert module._heads_per_rank() == uneven_heads
    assert at_three["q_b_proj"][1] == uneven_heads * q_head_width
    assert at_three["q_b_proj"][1] != DECLARED_HEADS * q_head_width // PERRANK_UNEVEN_WORLD
    assert at_three != at_two

    # Cross-check: the width a loader slices to is the width this class expects.
    # Both floor through ``_per_rank``, so they agree by construction -- asserted
    # rather than assumed, because a divergence would refuse a correct checkpoint.
    _patch_world(monkeypatch, PERRANK_WORLD)
    for leaf, name, axis in (
        ("q_b_proj_weight", "q_b_proj", 1),
        ("kv_b_proj_weight", "kv_b_proj", 1),
        ("o_proj_weight", "o_proj", 0),
    ):
        geometry = model_fp8._shard_geometry_for(module, leaf, PERRANK_WORLD)
        assert geometry.shard_size == at_two[name][axis]


# --------------------------------------------------------------------------- #
def test_perrank_numerics_the_reduced_sharded_decode_equals_the_unsharded_output(
    monkeypatch,
) -> None:
    """Conjunct (2): two ranks' reduced decode output equals the unsharded output. """
    import time

    model_fp8 = _model_module()

    # -- arm 1: the unsharded reference, at world size 1, with the real guard.
    reference_module, raw, gains, gen = build_attention()
    assert reference_module._heads_per_rank() == DECLARED_HEADS
    cache = seeded_cache(reference_module, gen)
    hidden, selection, scale = decode_inputs(reference_module, gen)

    # The guard, read directly: at world size 1 there is no group, so the
    # production line below cannot reduce and no vllm symbol is imported. This is
    # the reading that makes the eight tests above a control rather than a hope.
    assert model_fp8._resolve_tp_group() is None

    _reset_projection_counters()
    reference = run_decode_steps(
        reference_module, cache.clone(), hidden, selection, scale
    )
    whole_counters = _read_projection_counters()

    # -- arm 2: two ranks, each on its own heads, reducing through the real site.
    group = _CountedTwoRankGroup()
    _patch_world(monkeypatch, PERRANK_WORLD)
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)

    reduced, per_rank_counters = _run_sharded(
        group, raw, gains, cache, hidden, selection, scale
    )

    for rank, (nki, fallback) in enumerate(per_rank_counters):
        assert nki == whole_counters[0], (
            f"rank {rank} dispatched {nki} projections against {whole_counters[0]} "
            f"unsharded; partitioning must change widths, not the call count"
        )
        assert fallback == 0
    assert whole_counters[1] == 0

    # The counted reading. Once per decode step per rank, and not once more: a
    # reduction inside the per-head loop, or one left in a helper that runs twice,
    # would not read 6.
    assert group.calls == DECODE_STEPS * PERRANK_WORLD
    assert set(group.shapes) == {(BATCH, int(declared_config().hidden_size))}

    for step in range(DECODE_STEPS):
        report_and_check(
            f"PERRANK_2_STEP_{step}",
            reduced[step : step + 1],
            reference[step : step + 1],
        )

    # -- the doctored control. One rank's ``o_proj`` slice, right shape, taken one
    #    head too early. One step rather than three, because the control's job is
    #    to show the criterion can fail and a second and third step would only pay
    #    for the same demonstration twice.
    doctored_group = _CountedTwoRankGroup()
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: doctored_group)
    doctored, _ = _run_sharded(
        doctored_group, raw, gains, cache,
        hidden[:1], selection[:1], scale,
        doctor_rank_1_o_proj_by_one_head=True,
    )
    one_step_reference = reference[:1]
    fired = not torch.allclose(
        doctored.to(torch.float32),
        one_step_reference.to(torch.float32),
        rtol=RTOL,
        atol=atol_for(one_step_reference),
    )
    assert doctored_group.calls == PERRANK_WORLD
    assert fired


# --------------------------------------------------------------------------- #
def test_perrank_load_the_shard_table_cuts_the_five_mla_families(monkeypatch) -> None:
    """Conjunct (3): the three families carry a declared shard, the two latents none.
    """
    model_fp8 = _model_module()
    cfg = declared_config()
    module = _bare_attention()
    leaves = (
        "q_a_proj_weight",
        "q_b_proj_weight",
        "kv_a_proj_with_mqa_weight",
        "kv_b_proj_weight",
        "o_proj_weight",
    )

    def geometry_at(world_size: int) -> dict:
        return {
            leaf: model_fp8._shard_geometry_for(module, leaf, world_size)
            for leaf in leaves
        }

    # -- world size 1: nothing is sharded, which is the control arm. The same call
    #    distinguishes rather than agrees: five Nones here, three geometries below.
    at_one = geometry_at(1)
    assert list(at_one.values()) == [None] * len(leaves)

    # -- world size 2: three declared cuts, two replicated families.
    at_two = geometry_at(PERRANK_WORLD)
    triples = {
        leaf: None if g is None else (g.shard_dim, g.shard_size, g.num_shards)
        for leaf, g in at_two.items()
    }
    expected = {
        # Replicated: one latent per token, so no head axis to cut.
        "q_a_proj_weight": None,
        "kv_a_proj_with_mqa_weight": None,
        # Column-parallel on the output rows, in checkpoint orientation.
        "q_b_proj_weight": (0, PERRANK_HEADS * (NOPE_WIDTH + ROPE_WIDTH), PERRANK_WORLD),
        "kv_b_proj_weight": (0, PERRANK_HEADS * (NOPE_WIDTH + V_WIDTH), PERRANK_WORLD),
        # Row-parallel: the head width is this projection's input.
        "o_proj_weight": (1, PERRANK_HEADS * V_WIDTH, PERRANK_WORLD),
    }
    assert triples == expected

    # The shards tile the tensor exactly. ``shard_size * num_shards`` is the full
    # extent, so the two ranks' slices cover the checkpoint's dimension with no gap
    # and no overlap -- the property an off-by-one extent breaks and a shape check
    # on one rank alone would not notice.
    for leaf, full_extent in (
        ("q_b_proj_weight", DECLARED_HEADS * (NOPE_WIDTH + ROPE_WIDTH)),
        ("kv_b_proj_weight", DECLARED_HEADS * (NOPE_WIDTH + V_WIDTH)),
        ("o_proj_weight", DECLARED_HEADS * V_WIDTH),
    ):
        geometry = at_two[leaf]
        assert geometry.shard_size * geometry.num_shards == full_extent

    # -- the cut, followed through. A full weight set, sliced at those extents, is
    #    accepted by the production preparation for both ranks.
    _, raw, gains, _ = build_attention()
    _patch_world(monkeypatch, PERRANK_WORLD)
    for rank in range(PERRANK_WORLD):
        sharded = _bare_attention()
        weights = _rank_slices(raw, rank, PERRANK_WORLD)
        _install(sharded, weights, gains)
        assert tuple(sharded._prepared_weight("o_proj").shape) == (
            PERRANK_HEADS * V_WIDTH,
            int(cfg.hidden_size),
        )
        for _name, expected_heads, _contraction, _out_features in sharded.absorb_widths():
            assert expected_heads == PERRANK_HEADS

    wrong = _bare_attention()
    with pytest.raises(ValueError) as refusal:
        _install(wrong, {**_rank_slices(raw, 0, PERRANK_WORLD),
                         "q_b_proj": raw["q_b_proj"].clone()}, gains)
    assert "q_b_proj_weight is" in str(refusal.value)
