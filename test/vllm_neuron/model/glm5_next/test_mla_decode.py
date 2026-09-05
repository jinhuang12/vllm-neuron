"""`inc-glm53f-042` -- the MLA decode path.

WHAT THIS BLOCK BUILT. One method, ``Glm5NextMLAAttention.attend``, that turns hidden
states into this layer's attention output through the ABSORBED chain: project the query
and the KV latent, write the latent to its cache slot, lift the query into the latent
rank with ``inc-glm53f-097``'s absorb seam, run ``inc-glm53f-093``'s row-tiled sparse
attention, bring the result back down to the head width with the same absorb seam, and
project out. The expansion of ``kv_b_proj`` from a 512 latent to 32,768 per token --
which is what absorption exists to avoid -- never happens on this path.

THE ACCEPTANCE ITEMS, and where their wording comes from. All five are the plan block's
(revision 178, L690-704), and item (b) is the RE-REGISTERED form recorded at design
entry ``design-20260905-q`` / ``DECISIONS.md`` §16 after the first attempt measured the
original tolerance unachievable:

  (iv-a)  the weight split is EXACT for all 64 heads, bit-identical.
  (iv-b)  the split reproduces ``-039b``'s landed dense expansion, per head, at
          ``assert_close(rtol=1e-2, atol=1e-5)``.
          -- (iv-a)/(iv-b) per DECISIONS §15a.5 --
  (b-i)   a decode step matches the prefill run's corresponding slice, 3/3 steps.
  (b-ii)  a decode step matches an S-INDEPENDENT pure-fp32 torch oracle, 3/3 steps.
  (c)     ``B > 1`` raises a NAMED error rather than producing wrong numbers.
  (d)     the route predicate: five counter readings per arm.

WHY (b) CARRIES A COMPUTED TOLERANCE RATHER THAN A FIXED ``atol``. The first attempt
registered ``atol=1e-5``, and it could not be met by ANY implementation: the path
returns bf16, and a 16,384-term reduction leaves about ONE bf16 unit of the tensor's
largest element sitting on elements that cancelled to near zero. A fixed ``atol`` also
cannot travel -- rescaling the weights by 4 moved the required value by 2,223x while the
failing fraction rose. So the ruled tolerance is tied to the output's OWN resolution,
``atol_b = 2 * 2**-8 * max|reference|``: two bf16 units at the reference's magnitude.
That is not a tolerance fitted to the residual -- it is the dtype's granularity, and the
worst ratio it leaves is reported beside every verdict so a reviewer can see the
headroom rather than take it on trust.

WHY (b-ii) EXISTS ALONGSIDE (b-i). (b-i) compares two runs of the SAME kernels at
different token counts, so it cannot distinguish a faithful path from a wrong one that
is wrong identically on both sides. (b-ii) compares against pure torch in fp32, built
per row and therefore independent of the token count, which is the property the kernels
lack. Together they say the path is both self-consistent and correct; either alone
leaves one of those open.

WHAT THESE TESTS DO NOT ESTABLISH. Exit status is rung 1. Whether the coverage is
ADEQUATE is rung 2 and belongs to review, which is why every counted zero here owns a
control that fires and every tolerance verdict prints its own headroom.
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------- #
# THE DECLARED GEOMETRY. Every value is the checkpoint's, and the two that the route
# predicate actually reads are named as such, because a reader needs to know which
# numbers may not be shrunk for speed.
DECLARED_HEADS = 64
LATENT_RANK = 512          # kv_lora_rank. An EXACT fit: 512 % 128 == 0 and 512 <= 512,
                           # which is what makes `-041`'s tiled counter read 0.
NOPE_WIDTH = 256           # qk_nope_head_dim
V_WIDTH = 256              # v_head_dim
ROPE_WIDTH = 0             # qk_rope_head_dim on this checkpoint
CONTEXT_ROWS = 2048        # prior tokens already in the cache
SELECTED_ROWS = 2048       # the production top-k count. > 512, which is what makes
                           # `-093`'s row-tiled counter read 1.
DECODE_STEPS = 3
BATCH = 1

#: The registered pair for (iv-b), unchanged from the block's original registration.
RTOL = 1e-2
ATOL_IV_B = 1e-5

#: bf16 carries 8 mantissa bits, so one unit at magnitude m is about m * 2**-8. The
#: ruled tolerance for item (b) is TWO of those units.
BF16_MANTISSA_BITS = 8
BF16_UNITS_ALLOWED = 2

SENT = "MLADEC"


def say(*parts: object) -> None:
    """Print a reading. The suite runs under `-s`, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


def _model_module():
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def real_config():
    from vllm_neuron.model.glm5_next import config as cfg_mod

    return cfg_mod.Glm5NextTextConfig()


def declared_config():
    """The checkpoint's config, with the five widths this block depends on asserted.

    Asserted rather than assumed: if the fixture drifts off the checkpoint, the counter
    readings below stop meaning what their names say, and a silent drift would leave the
    route predicate passing on a geometry the plan never declared.
    """
    cfg = real_config()
    assert int(cfg.num_attention_heads) == DECLARED_HEADS
    assert int(cfg.kv_lora_rank) == LATENT_RANK
    assert int(cfg.qk_nope_head_dim) == NOPE_WIDTH
    assert int(cfg.v_head_dim) == V_WIDTH
    assert int(cfg.qk_rope_head_dim) == ROPE_WIDTH
    return cfg


def build_attention(*, seed: int = 850390, weight_scale: float = 1.0):
    """One MLA attention layer with dense random weights, load-time preps run.

    ``weight_scale`` exists so a test can show a verdict is scale-free rather than an
    artefact of one fixture's magnitudes.
    """
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
    """The five readings the route predicate names, each from its owning module.

    The owners are cross-referenced rather than restated: `-040` owns the seam counter,
    `-041` the tiled one, `-093` the row-tiled one, `-097` the absorb one. The fifth is
    the torch-fallback total, which is what makes "no torch path ran" a measurement
    rather than a claim about the source.
    """
    sparse, absorb = _counter_modules()
    seam = sparse.mla_sparse_dispatch_counters()
    tiled = sparse.mla_sparse_tiled_dispatch_counters()
    row_tiled = sparse.mla_sparse_row_tiled_dispatch_counters()
    absorbed = absorb.mla_absorb_dispatch_counters()
    return {
        "sparse_040": seam[0],
        "tiled_041": tiled[0],
        "row_tiled_093": row_tiled[0],
        "absorb_097": absorbed[0],
        "torch_fallback": seam[1] + tiled[1] + row_tiled[1] + absorbed[1],
    }


def atol_for(reference: torch.Tensor) -> float:
    """The ruled tolerance: two bf16 units at the reference's own magnitude.

    ``2 * 2**-8 * max|reference|``, per the plan block at revision 178. It is computed
    from the reference tensor of THIS comparison, so a test cannot borrow a looser
    magnitude from somewhere else.
    """
    peak = float(reference.detach().reshape(-1).to(torch.float32).abs().max())
    return BF16_UNITS_ALLOWED * (2.0 ** -BF16_MANTISSA_BITS) * peak


def report_and_check(label: str, got: torch.Tensor, reference: torch.Tensor) -> None:
    """Item (b)'s verdict, with the two numbers the ruling requires printed beside it.

    ``assert_close`` is PER ELEMENT -- ``|got - ref| <= atol + rtol*|ref|`` -- so a
    single max-difference figure would not say whether a miss is one cancelled element
    or thousands. The worst per-element ratio and the count of small-magnitude elements
    are therefore printed, and the assertion runs on the same numbers.
    """
    flat_got = got.detach().reshape(-1).to(torch.float32)
    flat_ref = reference.detach().reshape(-1).to(torch.float32)
    atol = atol_for(reference)
    peak = float(flat_ref.abs().max())
    diff = (flat_got - flat_ref).abs()
    allowance = atol + RTOL * flat_ref.abs()
    ratio = diff / allowance
    small = int((flat_ref.abs() < 0.1 * peak).sum())
    say(label, f"elements={flat_ref.numel()}", f"max_abs_diff={diff.max().item():.10g}",
        f"peak_reference={peak:.10g}", f"atol_b={atol:.10g}",
        f"worst_ratio={ratio.max().item():.6g}",
        f"elements_below_a_tenth_of_peak={small}",
        f"failing={int((diff > allowance).sum())}")
    torch.testing.assert_close(
        flat_got, flat_ref, rtol=RTOL, atol=atol,
        msg=lambda m: f"{label}: {m}",
    )


def seeded_cache(module, gen, *, rows: int = CONTEXT_ROWS, steps: int = DECODE_STEPS):
    """The latent cache, seeded directly.

    No prefill run is needed to build it: the cache is a plain tensor at the layer's own
    declared spec -- one KV head of ``head_size`` per token -- so the context rows are
    filled with values and the step slots left empty. That is what lets the acceptance
    use the production 2,048-row context without paying for a 2,048-token forward pass.
    """
    cache = torch.zeros(
        (rows + steps, module.NUM_LATENT_KV_HEADS, module.head_size),
        dtype=torch.bfloat16,
    )
    cache[:rows, 0, :] = torch.randn(
        (rows, module.head_size), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    return cache


def decode_inputs(module, gen, *, steps: int = DECODE_STEPS):
    """Hidden states and a selection whose every row lies BEFORE the new tokens.

    That restriction is what makes (b-i) a comparison of the plumbing rather than of the
    attention mask. A batched call writes all three latents before the seam reads the
    cache, so all three tokens would see each other; a step-by-step run lets token 1 see
    token 0 but not token 2. Selecting only prior rows removes that difference, and then
    a disagreement is the path's and not causality's.
    """
    hidden = (
        torch.randn((steps, module.hidden_size), generator=gen, dtype=torch.float32)
        * 0.5
    ).to(torch.bfloat16)
    order = torch.randperm(CONTEXT_ROWS, generator=gen)[:SELECTED_ROWS]
    selection = order.to(torch.int32).unsqueeze(0).repeat(steps, 1)
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
            )
        )
    return torch.cat(out, dim=0)


def fp32_oracle(module, raw, gains, cache_rows, hidden, selection, scale):
    """The whole chain in pure fp32, PER ROW, so it cannot depend on token count.

    The two seams' OWN torch oracles are used for absorb and for sparse attention rather
    than reimplemented here. That is deliberate: the question this oracle answers is
    whether the decode path composes the chain correctly, and an oracle I invented could
    differ from a seam by a convention -- where the softmax scale is applied, say --
    which
    would then read as a defect in the path instead of a difference in my arithmetic.
    The projections are plain matmuls and an RMSNorm, so those are written out.
    """
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


# --------------------------------------------------------------------------- #
# (iv-a) -- (iv-a)/(iv-b) per DECISIONS §15a.5
def test_item_iv_a_the_absorb_split_is_exact_for_all_heads() -> None:
    """`W_UK` and `W_UV` are the prepared `kv_b_proj`'s two halves, bit-identical.

    -- (iv-a)/(iv-b) per DECISIONS §15a.5 --

    EXACT and not merely close, because this is a reshape, a slice and a permute of a
    weight that already exists: no arithmetic happens, so any difference at all would
    mean the wrong bytes were selected. The orientations are NOT symmetric -- ``W_UK``
    is the key half TRANSPOSED, because absorb-in contracts the head width, while
    ``W_UV`` is the value half as it stands. A test that accepted either orientation
    would pass on an operand that produces silently wrong attention.
    """
    module, _, _, _ = build_attention()
    prepared = module._prepared_weight("kv_b_proj")
    say("IV_A_PREPARED_SHAPE", tuple(prepared.shape), prepared.dtype)
    assert tuple(prepared.shape) == (
        LATENT_RANK,
        DECLARED_HEADS * (NOPE_WIDTH + V_WIDTH),
    )
    w_uk = module._absorb_weight("W_UK")
    w_uv = module._absorb_weight("W_UV")
    say("IV_A_W_UK_SHAPE", tuple(w_uk.shape))
    say("IV_A_W_UV_SHAPE", tuple(w_uv.shape))
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
    say("IV_A_HEADS_CHECKED", DECLARED_HEADS)
    say("IV_A_W_UK_HEADS_NOT_BIT_IDENTICAL", mismatched_uk)
    say("IV_A_W_UV_HEADS_NOT_BIT_IDENTICAL", mismatched_uv)
    assert mismatched_uk == 0
    assert mismatched_uv == 0

    # Two firing controls. Without them the two zeros above would pass on an equality
    # that cannot tell anything apart.
    perturbed = w_uk[0].clone()
    perturbed[0, 0] += 1.0
    control_perturbed_rejected = not torch.equal(
        perturbed, prepared[:, :NOPE_WIDTH].T
    )
    control_untransposed_rejected = not torch.equal(
        w_uk[0], prepared[:, :NOPE_WIDTH]
    )
    say("IV_A_CONTROL_REJECTS_A_PERTURBED_HEAD", control_perturbed_rejected)
    say("IV_A_CONTROL_REJECTS_AN_UNTRANSPOSED_OPERAND", control_untransposed_rejected)
    assert control_perturbed_rejected
    assert control_untransposed_rejected


# --------------------------------------------------------------------------- #
def test_item_iv_b_the_split_reproduces_the_landed_dense_expansion() -> None:
    """Per head, the absorbed operands reproduce `-039b`'s expanded key and value.

    -- (iv-a)/(iv-b) per DECISIONS §15a.5 --

    This is the item that ties the split to something already landed rather than to my
    own arithmetic. ``project_qkv`` expands the latent densely into ``key_nope`` and
    ``value``; the absorbed path never performs that expansion, so the only way to know
    the two halves were cut on the right boundary is to reproduce its output from them.

    Registered at ``assert_close(rtol=1e-2, atol=1e-5)`` and NOT at bit-identity: the
    first attempt registered bit-identity here and measurement refuted it -- 0 of 64
    heads agreed bit-for-bit, because two matmul implementations reassociate
    differently. This test prints its own worst absolute difference for both operands
    rather than quoting an earlier probe's figure, because that probe ran at a different
    input dtype and its number is not true of this run.
    """
    module, _, _, gen = build_attention()
    tokens = 4

    # THE HIDDEN STATES ARE fp32 HERE, DELIBERATELY, and the first attempt at this test
    # got it wrong in a way worth recording. Both methods return their results cast to
    # the INPUT's dtype. With bf16 hidden states, the latent this test reads back is a
    # bf16 ROUND of the latent `project_qkv` expanded internally in fp32 -- so the test
    # compared an expansion of one latent against an expansion of a slightly different
    # one. It failed at 90 of 1,024 elements with a greatest RELATIVE difference of
    # 1.33, which is far too large to be reassociation and was the clue: a 0.4 %
    # perturbation of the input, amplified at output elements that cancel.
    #
    # In fp32 both methods share the same first three dispatches on the same input, and
    # those dispatches are deterministic, so the latent this test expands IS the latent
    # `project_qkv` expanded. The query check below is what proves that shared prefix
    # rather than assuming it.
    hidden = (
        torch.randn((tokens, module.hidden_size), generator=gen, dtype=torch.float32)
        * 0.5
    )
    query_dense, key_nope, value = (t.detach() for t in module.project_qkv(hidden))
    say("IV_B_KEY_NOPE_SHAPE", tuple(key_nope.shape))
    say("IV_B_VALUE_SHAPE", tuple(value.shape))
    say("IV_B_KEY_NOPE_DTYPE", key_nope.dtype)

    query_absorbed, latent = (
        t.detach() for t in module.project_query_and_latent(hidden)
    )
    latent32 = latent.to(torch.float32)
    say("IV_B_LATENT_SHAPE", tuple(latent32.shape), latent32.dtype)
    assert tuple(latent32.shape) == (tokens, LATENT_RANK)

    # THE SHARED-PREFIX PROOF. The two methods are line-for-line identical from their
    # `x = hidden_states.to(float32)` through `q_a_proj`, the q norm, `q_b_proj`, the
    # reshape, `kv_a_proj_with_mqa` and the kv norm; they diverge only at whether the
    # latent is then expanded. So equal queries and an equal latent follow from the same
    # fact. Reading that off the source is not a reading, so the equality below is the
    # runtime proof: if these dispatches were not deterministic on one input, or if the
    # two chains had drifted apart, this is what would catch it.
    shared_prefix_identical = torch.equal(query_dense, query_absorbed)
    say("IV_B_THE_TWO_METHODS_SHARE_A_BIT_IDENTICAL_QUERY", shared_prefix_identical)
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
    say("IV_B_HEADS_CHECKED", DECLARED_HEADS)
    say("IV_B_WORST_ABS_DIFF_KEY", f"{worst_key:.10g}")
    say("IV_B_WORST_ABS_DIFF_VALUE", f"{worst_value:.10g}")
    say("IV_B_REFERENCE_MAGNITUDE",
        f"{float(key_nope.to(torch.float32).abs().max()):.6g}")

    # The comparison must be able to reject, or 128 passing assertions above mean
    # nothing. One element of the expected tensor is moved.
    spoiled = key_nope[:, 0].to(torch.float32).clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        latent32 @ w_uk[0].T, spoiled, rtol=RTOL, atol=ATOL_IV_B
    )
    say("IV_B_CONTROL_REJECTS_A_MOVED_ELEMENT", fired)
    assert fired


# --------------------------------------------------------------------------- #
def test_item_b_i_decode_matches_the_prefill_slice_for_every_step() -> None:
    """A decode step equals the prefill run's matching row, 3/3, at the tolerance.

    The prefill reference arm is ONE call carrying all three tokens; the decode arm is
    three calls of one token each with the cache growing between them. Both run the same
    method -- there is no separate prefill implementation to drift from -- so this item
    asks whether the cache write, the cache read and the two absorb call sites behave
    the same way regardless of how many tokens arrive at once.
    """
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    prefill = module.attend(
        hidden, cache.clone(), CONTEXT_ROWS, selection, scale, batch_size=BATCH
    )
    decode = run_decode_steps(module, cache.clone(), hidden, selection, scale)
    say("B_I_PREFILL_SHAPE", tuple(prefill.shape), prefill.dtype)
    say("B_I_DECODE_SHAPE", tuple(decode.shape), decode.dtype)
    assert tuple(decode.shape) == tuple(prefill.shape)

    for step in range(DECODE_STEPS):
        report_and_check(
            f"B_I_STEP_{step}", decode[step : step + 1], prefill[step : step + 1]
        )
    say("B_I_STEPS_AGREEING", f"{DECODE_STEPS}/{DECODE_STEPS}")

    spoiled = prefill.to(torch.float32).clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        decode.to(torch.float32), spoiled, rtol=RTOL, atol=atol_for(prefill)
    )
    say("B_I_CONTROL_REJECTS_A_MOVED_ELEMENT", fired)
    assert fired


# --------------------------------------------------------------------------- #
def test_item_b_ii_decode_matches_a_token_count_independent_fp32_oracle() -> None:
    """A decode step equals a pure-fp32, per-row torch oracle, 3/3 steps.

    This is the item (b-i) cannot cover. (b-i) compares two runs of the same kernels, so
    an implementation that is wrong identically on both sides satisfies it. The oracle
    here shares no kernel with the path and is built per row, so it is independent of
    the
    token count -- which is exactly the property the kernels do not have.
    """
    module, raw, gains, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    decode = run_decode_steps(module, cache.clone(), hidden, selection, scale)

    # The oracle reads the same cache the path wrote, including the three new latents,
    # so the two sides attend over identical rows.
    oracle_cache = cache.clone()
    oracle_cache[CONTEXT_ROWS : CONTEXT_ROWS + DECODE_STEPS, 0, :] = oracle_latent(
        module, raw, gains, hidden
    ).to(torch.bfloat16)
    rows = oracle_cache[: CONTEXT_ROWS + DECODE_STEPS, 0, :].to(torch.float32)
    reference = fp32_oracle(module, raw, gains, rows, hidden, selection, scale)
    say("B_II_ORACLE_SHAPE", tuple(reference.shape), reference.dtype)

    for step in range(DECODE_STEPS):
        report_and_check(
            f"B_II_STEP_{step}", decode[step : step + 1], reference[step : step + 1]
        )
    say("B_II_STEPS_AGREEING", f"{DECODE_STEPS}/{DECODE_STEPS}")

    # The oracle must be able to disagree, or it is not an independent reference.
    spoiled = reference.clone()
    spoiled[0, 0] += 1.0
    fired = not torch.allclose(
        decode.to(torch.float32), spoiled, rtol=RTOL, atol=atol_for(reference)
    )
    say("B_II_CONTROL_REJECTS_A_MOVED_ELEMENT", fired)
    assert fired


# --------------------------------------------------------------------------- #
def test_item_c_a_larger_batch_raises_a_named_error_first() -> None:
    """`B > 1` raises `Glm5NextMLADecodeError`, and nothing dispatches.

    A DISTINCT type rather than a bare ``ValueError``, and the test asserts the type by
    name: the serving constraint has to be distinguishable from a shape typo, and a test
    catching ``ValueError`` would pass on either -- so the constraint would be asserted
    without being measured.

    The counters are read AFTER the refusal to separate "raised before dispatching" from
    "dispatched and then raised". Those are different defects and only one is
    acceptable.
    """
    model_fp8 = _model_module()
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    reset_counters()
    with pytest.raises(model_fp8.Glm5NextMLADecodeError) as caught:
        module.attend(
            hidden, cache.clone(), CONTEXT_ROWS, selection, scale, batch_size=2
        )
    message = " ".join(str(caught.value).split())
    say("C_RAISED_TYPE", type(caught.value).__name__)
    say("C_MESSAGE", message[:170])
    assert "batch_size" in message

    after = read_counters()
    say("C_COUNTERS_AFTER_REFUSAL", after)
    for name, value in after.items():
        assert value == 0, f"{name} moved on a refusal: {after}"

    # The five zeros above need a control, or a counter that never increments at all
    # would produce the same reading as a refusal that correctly precedes dispatch.
    # This runs AFTER they are asserted, so it cannot influence them.
    module.attend(hidden[:1], cache.clone(), CONTEXT_ROWS, selection[:1], scale,
                  batch_size=BATCH)
    admissible = read_counters()
    say("C_CONTROL_ONE_ADMISSIBLE_CALL", admissible)
    assert admissible["sparse_040"] == 1
    assert admissible["absorb_097"] == 2

    say("C_CONTROL_A_BARE_VALUEERROR_WOULD_NOT_DISTINGUISH_IT",
        issubclass(model_fp8.Glm5NextMLADecodeError, ValueError))
    assert issubclass(model_fp8.Glm5NextMLADecodeError, ValueError)


# --------------------------------------------------------------------------- #
def test_item_d_route_predicate_five_counter_readings_on_every_arm() -> None:
    """Which kernels ran, counted, on the decode arm and the prefill reference arm.

    WHY A NUMERIC ITEM IS NOT ENOUGH, which is the whole reason this item exists: (b-i)
    compares a decode step against a prefill slice of the SAME implementation, and a
    torch MLA would satisfy both sides equally. Comparing a path against itself cannot
    establish which path it is. These counters can.

    The values are the plan block's, cross-referenced to their owning increments rather
    than redefined here: `-040`'s seam counter fires on every dispatch, `-041`'s tiled
    counter only on a tiled latent body, `-093`'s row-tiled counter only when the
    selected-row count exceeds the moving maximum, `-097`'s absorb counter once per
    absorb call site. The two zeros are load-bearing and each owns a firing control.
    """
    sparse, _ = _counter_modules()
    module, _, _, gen = build_attention()
    cache = seeded_cache(module, gen)
    hidden, selection, scale = decode_inputs(module, gen)

    # The predicates are DERIVED from the seam's own constants, so the expected counter
    # values and the kernel actually chosen cannot disagree.
    tiled = LATENT_RANK % sparse.LATENT_TILE != 0 or LATENT_RANK > sparse.MOVING_MAX
    rows_tiled = SELECTED_ROWS > sparse.MOVING_MAX
    say("D_SEAM_LATENT_TILE", sparse.LATENT_TILE)
    say("D_SEAM_MOVING_MAX", sparse.MOVING_MAX)
    say("D_PREDICATE_tiled", f"{LATENT_RANK} -> {tiled}")
    say("D_PREDICATE_rows_tiled", f"{SELECTED_ROWS} -> {rows_tiled}")
    assert tiled is False
    assert rows_tiled is True

    reset_counters()
    run_decode_steps(module, cache.clone(), hidden, selection, scale)
    decode_counts = read_counters()
    say("D_DECODE_ARM_OVER_3_STEPS", decode_counts)
    assert decode_counts == {
        "sparse_040": DECODE_STEPS * 1,
        "tiled_041": 0,
        "row_tiled_093": DECODE_STEPS * 1,
        "absorb_097": DECODE_STEPS * 2,
        "torch_fallback": 0,
    }

    reset_counters()
    module.attend(hidden, cache.clone(), CONTEXT_ROWS, selection, scale,
                  batch_size=BATCH)
    prefill_counts = read_counters()
    say("D_PREFILL_REFERENCE_ARM_ONE_CALL", prefill_counts)
    assert prefill_counts == {
        "sparse_040": 1,
        "tiled_041": 0,
        "row_tiled_093": 1,
        "absorb_097": 2,
        "torch_fallback": 0,
    }

    # THE FIRING CONTROL FOR `-041`'s COUNTED ZERO. A latent that is NOT an exact fit
    # must make the tiled counter read 1. Without this, a tiled counter that never
    # increments for any reason would produce the same zero as a correct exact-fit
    # route. Run directly on the seam at a small geometry, because the point is the
    # predicate and not the model.
    # THE CONTROL'S OWN GEOMETRY HAS TO BE ADMISSIBLE, and the first attempt's was not.
    # Three of the seam's rules bind here and all three are read off its own validator:
    #   * the selected-row count must be a positive multiple of KEY_CHUNK, so topk=4
    #     (the first attempt's value) is refused outright -- 128 is the smallest legal
    #     value and is used;
    #   * the latent rank has NO multiple-of rule, only a positivity one, so a latent
    #     that fails the exact-fit test is legal on its own;
    #   * tiling the row axis and the latent axis in the SAME call is explicitly
    #     refused, so this control must keep topk at or below MOVING_MAX while the
    #     latent is ragged. topk=128 satisfies that.
    # The raggedness used is the MODULUS clause: 576 % 128 = 64, so it is not an exact
    # fit for the partition tile and the tiled body is the one that must run.
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
    say("D_CONTROL_A_RAGGED_LATENT_MAKES_THE_TILED_COUNTER_FIRE",
        f"latent={ragged}", f"topk={control_topk}", ragged_counts)
    assert ragged_counts["tiled_041"] == 1
    assert ragged_counts["torch_fallback"] == 0

    # THE FIRING CONTROL FOR `-093`'s READING, the other direction: a selection at or
    # below the moving maximum must leave the row-tiled counter at 0, so the 1 above is
    # a response to the production row count and not a constant.
    reset_counters()
    assert sparse.MOVING_MAX % sparse.KEY_CHUNK == 0
    small_idx = torch.zeros((1, sparse.MOVING_MAX), dtype=torch.int32)
    q_small = torch.randn((1, 2, LATENT_RANK), dtype=torch.float32) * 0.1
    c_small = torch.randn((512, LATENT_RANK), dtype=torch.float32) * 0.1
    sparse.mla_sparse_attention(q_small, c_small, small_idx, 0.1)
    small_counts = read_counters()
    say("D_CONTROL_A_SMALL_SELECTION_LEAVES_ROW_TILED_AT_ZERO",
        f"topk={sparse.MOVING_MAX}", small_counts)
    assert small_counts["row_tiled_093"] == 0
    assert small_counts["sparse_040"] == 1

    # THE TORCH-FALLBACK ZERO IS THE ONE READING THAT CANNOT OWN A FIRING CONTROL, and
    # saying so is better than staging a control that only looks like one. Neither seam
    # module contains a torch path at all -- `mla_absorb`'s own accessor docstring says
    # its `torch_fallback` "can only ever read 0, because this module has no torch path"
    # -- so no input makes this counter increment, and a control that cannot fire is
    # exactly what the rest of this file refuses to write.
    #
    # What IS checkable is that the zero is a real reading rather than a missing field:
    # it comes from each module's own two-member accessor, and it is summed across all
    # four accessors in `read_counters`, so a fallback added to ANY of them later would
    # surface here instead of being silently ignored.
    for accessor in (
        sparse.mla_sparse_dispatch_counters,
        sparse.mla_sparse_tiled_dispatch_counters,
        sparse.mla_sparse_row_tiled_dispatch_counters,
    ):
        reading = accessor()
        say("D_FALLBACK_MEMBER_PRESENT", accessor.__name__, reading)
        assert len(reading) == 2
    say("D_TORCH_FALLBACK_CANNOT_BE_MADE_TO_FIRE",
        "no torch path exists in either seam module; disclosed, not staged")
