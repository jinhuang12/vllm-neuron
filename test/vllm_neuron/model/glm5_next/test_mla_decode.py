"""`inc-glm53f-042` -- the MLA decode path.

WHAT THIS BLOCK BUILT. One method, ``Glm5NextMLAAttention.attend``, that turns hidden
states into this layer's attention output through the ABSORBED chain: project the query
and the KV latent, write the latent to its cache slot, lift the query into the latent
rank with ``inc-glm53f-097``'s absorb seam, run ``inc-glm53f-093``'s row-tiled sparse
attention, bring the result back down to the head width with the same absorb seam, and
project out. The expansion of ``kv_b_proj`` from a 512 latent to 32,768 per token --
which is what absorption exists to avoid -- never happens on this path.

THE ACCEPTANCE ITEMS, and where their wording comes from. All are the plan block
``#### `inc-glm53f-042``'s, and item (b) is the RE-REGISTERED form recorded at design
entry ``design-20260905-q`` / ``DECISIONS.md`` §16 after the first attempt measured the
original tolerance unachievable. The block is cited BY ANCHOR and not by line number,
per D-18 and review item B72-N6: the line span moved twice while this file was being
written, and a number folded today is stale at the next lap.

  (iv-a)  the weight split is EXACT for all 64 heads, bit-identical.
  (iv-b)  the split reproduces ``-039b``'s landed dense expansion, per head, at
          ``assert_close(rtol=1e-2, atol=1e-5)``.
          -- (iv-a)/(iv-b) per DECISIONS §15a.5 --
  (b-i)   a decode step matches the prefill run's corresponding slice, 3/3 steps.
  (b-ii)  a decode step matches an S-INDEPENDENT pure-fp32 torch oracle, 3/3 steps.
  (b-iii) the latent this step WROTE reads back bit-identical from its cache slot.
  (b-iv)  a measurement, not a criterion: how much the own slot moves the output.
  (c)     ``B > 1`` raises a NAMED error rather than producing wrong numbers.
  (d)     the route predicate: five counter readings per arm.

WHY (b-iii) AND (b-iv) EXIST -- review finding B72-M1, repaired under this same id. The
first landing of this file could not fail if the cache write were deleted, misplaced or
wrong. Every selected row was a PRIOR row (``selection.max() < CONTEXT_ROWS``), the seam
and both oracles only ever gather selected rows, and every arm passed ``cache.clone()``
and then discarded it -- so no reading in the file depended on the write at all, while
two of its own sentences claimed otherwise. The repair is test-only: the selection now
includes each step's OWN slot, the two false sentences are corrected below, and (b-iii)
reads the written latent back with NO tolerance to hide in.

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


def decode_inputs(module, gen, *, steps: int = DECODE_STEPS, self_inclusive: bool = True):
    """Hidden states, and a selection of prior rows PLUS each step's own cache slot.

    WHAT CHANGED AND WHY (review finding B72-M1). This helper used to select only rows
    BEFORE the new tokens and assert ``selection.max() < CONTEXT_ROWS``. That made every
    reading in the file blind to the cache write: the sparse seam and both oracles only
    ever gather SELECTED rows, so a slot no one selects cannot influence any output, and
    the write could have been deleted with six of six items still passing.

    Step ``s`` now selects its OWN slot ``CONTEXT_ROWS + s`` and no other new slot. That
    is the one form that keeps (b-i) valid. The reason the old code gave for excluding
    new rows is still true as far as it goes -- a batched call writes all three latents
    before its single read, so a token selecting a LATER token would see it in the
    batched arm and not in the step-by-step arm -- but a token selecting only ITSELF is
    causally identical in both arms: the batched arm has already written slot
    ``CONTEXT_ROWS + s`` before it reads, and the step-by-step arm writes that same slot
    at the start of step ``s``. So the two arms still gather the same rows, and a
    disagreement is still the path's rather than causality's.

    The selected count does not move: ``SELECTED_ROWS`` prior rows becomes
    ``SELECTED_ROWS - 1`` prior rows plus the own slot, so the total stays 2,048 and
    stays a multiple of the seam's ``KEY_CHUNK``.

    ``self_inclusive=False`` reproduces the OLD selection exactly. It exists for (b-iv),
    which measures how much the own slot actually moves the output, and it is not used by
    any acceptance item.
    """
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
    # OTHER step's slot, which is the property that keeps (b-i) causally sound.
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
    asks whether the cache read and the two absorb call sites behave the same way
    regardless of how many tokens arrive at once.

    WHAT THIS ITEM DOES NOT ASK, corrected per review finding B72-N1. The sentence here
    used to claim this item asks about "the cache write". It cannot, and it still cannot
    after the M1 repair: both arms run the same write, so agreeing tells us they are
    CONSISTENT and says nothing about whether either wrote the right value to the right
    slot. That question has no tolerance in it and belongs to (b-iii), which reads the
    slot back bit-identically. What the repair does buy this item is that the written
    row is now GATHERED, so a write that landed in the wrong slot in one arm and not the
    other would now show up here.
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

    # The oracle attends over CORRESPONDING rows, not over the same tensor -- corrected
    # per review finding B72-N1. The sentence here used to say the oracle "reads the same
    # cache the path wrote, including the three new latents". It does not: it reads its
    # OWN clone, whose three new slots are filled below with the ORACLE's fp32 latents,
    # while the path reads its own clone holding the bf16 latents IT wrote. Before the M1
    # repair the claim was doubly false, because nothing gathered those three rows on
    # either side. Now step s selects slot CONTEXT_ROWS + s, so both sides do gather the
    # row -- each its own version of it -- and the comparison covers the written value at
    # the §16 pair. That is a correspondence between two independent computations, which
    # is what an oracle is for, and it is stated that way rather than as shared state.
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
def test_item_b_iii_the_written_latent_reads_back_bit_identical() -> None:
    """The latent step ``s`` wrote IS in slot ``CONTEXT_ROWS + s``, bit for bit.

    THIS IS THE ITEM THAT CLOSES B72-M1, and it closes it because it has no tolerance to
    hide in. ``torch.equal`` on bf16 is exact, so a deleted write, a write to the wrong
    slot, a write of the wrong value and a write of the right value in the wrong dtype
    are four different failures here and none of them can pass.

    THE CONTROLS COME FIRST, and each asserts its own plant landed before it reads
    anything. That order matters: a control that never planted would report the same
    clean result as a control that planted and saw the effect, which is the vacuity class
    this campaign has hit twice.

      1. Every step slot is PRE-POISONED with a sentinel no latent can equal. If the
         write line were absent, the sentinel would still be there afterwards -- so
         "0 slots still hold the sentinel" is a positive reading about the write having
         happened, not an absence of evidence.
      2. One slot BEYOND the steps is never written and must still read zero, which is
         what would fire on a write that ran off its slot.
      3. The comparison itself is shown able to reject: one element of the expected
         latent is moved and ``torch.equal`` must return False.

    The expectation is ``project_query_and_latent(hidden[s : s + 1])[1]`` cast to the
    cache dtype -- the same three deterministic dispatches the path itself runs, whose
    determinism on one input is already proven by (iv-b)'s bit-identical shared-prefix
    reading. So this is the (iv-a) form applied to the cache: exact, not approximate.
    """
    module, _, _, gen = build_attention()
    # One SPARE slot past the three the path writes, so control 2 has something untouched
    # to read. attend() reads [:start + tokens], so a trailing row is never gathered.
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
    say("B_III_CONTROL_1_THE_SENTINEL_PLANT_LANDED", f"{planted}/{DECODE_STEPS}")
    assert planted == DECODE_STEPS
    say("B_III_CONTROL_2_THE_SPARE_SLOT_STARTS_ZERO",
        bool(torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))))
    assert torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))

    # NOT a clone. The whole point is to inspect the tensor the path wrote into.
    decode = run_decode_steps(module, cache, hidden, selection, scale)
    say("B_III_DECODE_SHAPE", tuple(decode.shape), decode.dtype)

    survived = sum(
        int(torch.equal(cache[CONTEXT_ROWS + step, 0, :], sentinel))
        for step in range(DECODE_STEPS)
    )
    say("B_III_SENTINEL_SURVIVED_SLOTS", f"{survived}/{DECODE_STEPS}")
    assert survived == 0

    agreeing = 0
    for step in range(DECODE_STEPS):
        want = module.project_query_and_latent(hidden[step : step + 1])[1]
        want_bf16 = want.to(cache.dtype)[0]
        got = cache[CONTEXT_ROWS + step, 0, :]
        same = bool(torch.equal(got, want_bf16))
        say(f"B_III_READBACK_STEP_{step}", same, tuple(got.shape), got.dtype,
            f"max_abs_diff={float((got.to(torch.float32) - want_bf16.to(torch.float32)).abs().max()):.10g}")
        assert same
        agreeing += 1
    say("B_III_SLOTS_AGREEING_BIT_FOR_BIT", f"{agreeing}/{DECODE_STEPS}")
    assert agreeing == DECODE_STEPS

    say("B_III_THE_SPARE_SLOT_IS_STILL_ZERO",
        bool(torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))))
    assert torch.equal(cache[spare, 0, :], torch.zeros_like(cache[spare, 0, :]))

    moved = module.project_query_and_latent(hidden[0:1])[1].to(cache.dtype)[0].clone()
    moved[0] = moved[0] + 1.0
    fired = not torch.equal(cache[CONTEXT_ROWS, 0, :], moved)
    say("B_III_CONTROL_3_REJECTS_A_MOVED_ELEMENT", fired)
    assert fired


# --------------------------------------------------------------------------- #
def test_item_b_iv_how_much_the_own_slot_moves_the_output() -> None:
    """A MEASUREMENT, not a criterion: is the self-inclusive selection enough on its own?

    Review finding B72-M1 offered two repairs and judged that "(a) is sufficient alone",
    (a) being the self-inclusive selection. I predicted before running that it is NOT,
    and this item measures it instead of arguing it. The arithmetic behind the prediction
    (``predictions-042-repair.txt`` R3): the own slot is 1 of 2,048 gathered rows, so its
    softmax weight is about 1/2048 = 4.9e-4, while the registered allowance is
    ``atol_b = 2 * 2**-8 * max|reference|`` -- about 1.8e-3 at the measured peak of 0.23,
    before ``rtol * |ref|`` is even added. If that is right, a DELETED write perturbs the
    output by less than the tolerance permits and (b-i)/(b-ii) would still pass, which
    would leave (b-iii) as the item actually carrying the repair.

    SO THIS ITEM ASSERTS ONLY WHAT MUST HOLD REGARDLESS -- that the two arms differ in
    exactly one selected column and share their hidden states -- and REPORTS the influence
    ratio. It deliberately does not assert the ratio in either direction: asserting
    ``>= 1`` would redden the suite for a true fact that is not a defect, and asserting
    ``< 1`` would freeze my own prediction into a criterion. The number is printed and
    the reviewer reads it.
    """
    # ONE module for both arms. `attend` mutates only the cache passed to it and reads
    # fixed weights, so a second build would cost two load-time prep passes and buy
    # nothing; and the dispatch counters live in the seam modules, not here.
    module, _, _, _ = build_attention()
    # Two generators on ONE seed, so hidden states and the prior permutation are the same
    # draw on both arms and the only difference is the single swapped column.
    gen_a = torch.Generator().manual_seed(4242)
    gen_b = torch.Generator().manual_seed(4242)
    hidden_a, sel_inclusive, scale = decode_inputs(module, gen_a)
    hidden_b, sel_prior_only, _ = decode_inputs(module, gen_b, self_inclusive=False)

    say("B_IV_THE_TWO_ARMS_SHARE_HIDDEN_STATES", bool(torch.equal(hidden_a, hidden_b)))
    assert torch.equal(hidden_a, hidden_b)
    differing_columns = int((sel_inclusive != sel_prior_only).sum(dim=1).max())
    say("B_IV_SELECTIONS_DIFFER_IN_COLUMNS", differing_columns)
    assert differing_columns == 1
    say("B_IV_INCLUSIVE_OWN_SLOT", int(sel_inclusive[0].max()),
        "PRIOR_ONLY_MAX", int(sel_prior_only[0].max()))
    assert int(sel_inclusive[0].max()) == CONTEXT_ROWS
    assert int(sel_prior_only[0].max()) < CONTEXT_ROWS

    # Both caches are seeded from generators on ONE seed, so the 2,048 prior rows are
    # byte-identical across the arms. Without that the two runs would differ because the
    # CONTEXT differed, and the reading would say nothing about the own slot -- the exact
    # "value produced under one condition, applied to another" mistake this increment has
    # made three times. So it is asserted, not assumed.
    cache_inclusive = seeded_cache(module, torch.Generator().manual_seed(99001))
    cache_prior_only = seeded_cache(module, torch.Generator().manual_seed(99001))
    say("B_IV_THE_TWO_ARMS_SHARE_THEIR_PRIOR_CONTEXT",
        bool(torch.equal(cache_inclusive, cache_prior_only)))
    assert torch.equal(cache_inclusive, cache_prior_only)

    inclusive = run_decode_steps(
        module, cache_inclusive, hidden_a, sel_inclusive, scale
    )
    prior_only = run_decode_steps(
        module, cache_prior_only, hidden_a, sel_prior_only, scale
    )

    diff = (inclusive.to(torch.float32) - prior_only.to(torch.float32)).abs()
    allowance = atol_for(prior_only) + RTOL * prior_only.to(torch.float32).abs()
    ratio = float((diff / allowance).max())
    say("B_IV_MAX_ABS_DIFF", f"{float(diff.max()):.10g}",
        "ATOL_B", f"{atol_for(prior_only):.10g}",
        "WORST_RATIO", f"{ratio:.6g}",
        "ELEMENTS_DIFFERING", int((diff > 0).sum()),
        "OF", diff.numel())
    say("B_IV_IS_THE_OWN_SLOT_DETECTABLE_AT_THE_REGISTERED_TOLERANCE",
        "YES" if ratio >= 1.0 else "NO")
    say("B_IV_SO_IS_REPAIR_A_SUFFICIENT_ALONE",
        "YES" if ratio >= 1.0 else "NO -- (b-iii) carries the repair")


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


# =========================================================================== #
# `inc-glm53f-100` -- MLA PER-RANK HEAD PARTITIONING. Three items, one per
# conjunct of the plan block, added to THIS file because all three read the same
# geometry, the same module builder and the same registered comparison pair the
# items above already use.
#
# WHAT THE INCREMENT CHANGED, in one sentence: `Glm5NextMLAAttention` now computes
# on THIS RANK's heads, the three projections whose width carries the head count
# load sharded, and `project_output` -- the row-parallel site -- sums its partial
# across the tensor-parallel group.
#
# THE COMPARISON PAIR IS NOT RE-AUTHORED HERE. Item (2) below reuses
# `report_and_check`, and therefore `RTOL` and `atol_for`, exactly as item (b-ii)
# does. That pair is registered and frozen; this increment does not choose one.
#
# WHY WORLD SIZE 1 IS THE CONTROL ARM EVERYWHERE. `_shard_geometry_for` returns
# `None` for every family at world size 1 and `_resolve_tp_group` returns `None`
# there too, so a single-rank run takes exactly the path it took before this
# increment -- which is what makes the eight items above the control that the
# sharded path is what changed.
# =========================================================================== #

PERRANK_WORLD = 2
#: 64 heads over 2 ranks. An EXACT split, asserted below rather than assumed: a
#: world size that did not divide the head count would floor, and then no rank's
#: heads would sum back to the model's.
PERRANK_HEADS = DECLARED_HEADS // PERRANK_WORLD

#: A world size that does NOT divide 64, for the flooring control in item (1).
PERRANK_UNEVEN_WORLD = 3


def _patch_world(monkeypatch, world_size: int) -> None:
    """Run the module at a synthetic world size, at the resolver the code reads.

    The same injection point `-094`'s load items use
    (``test_load_weights.py:3180``), so this file adds no second way to say
    "pretend there are two ranks".
    """
    monkeypatch.setattr(_model_module(), "_resolve_world_size", lambda: world_size)


def _bare_attention():
    """One MLA layer with NO weights installed, at the checkpoint's real geometry.

    ``build_attention`` above generates weights AT ``projection_widths()``, which is
    the right thing for the items above and the wrong thing for item (2): a sharded
    arm needs each rank to hold a SLICE of one full weight set, not fresh random
    numbers of the per-rank shape. So the two builders are separate, and this one
    installs nothing.
    """
    return _model_module().Glm5NextMLAAttention(declared_config())


def _install(module, weights: dict, gains: dict) -> None:
    """Install a given weight set and run the two load-time preparations.

    ``prepare_projection_weights`` checks every installed weight against
    ``projection_widths()`` and refuses a mismatch with the site named, so this
    helper is also the production width check: a slice at the wrong geometry stops
    here rather than computing the wrong function at plausible shapes.
    """
    for name, weight in weights.items():
        setattr(module, f"{name}_weight", torch.nn.Parameter(weight))
    for name, gain in gains.items():
        setattr(module, name, torch.nn.Parameter(gain))
    module.prepare_projection_weights()
    assert module.prepare_absorb_weights() == 2


def _rank_slices(raw: dict, rank: int, world_size: int) -> dict:
    """One rank's share of a FULL weight set, cut at the ratified geometry.

    The cuts are the ratified shard table's (``increments/shard-table-094.md``
    Part 5, Group B), written out here as this file's own statement of them rather
    than read from ``_SHARD_GEOMETRY`` -- so the table and the code CAN disagree,
    and if they do item (3) goes red.

      ``q_b_proj``   dim 0, ``heads * (nope + rope)`` rows per rank
      ``kv_b_proj``  dim 0, ``heads * (nope + v)``    rows per rank
      ``o_proj``     dim 1, ``heads * v``             columns per rank

    ALL THREE CUTS MUST NAME THE SAME HEADS, and that is the property the
    arithmetic below guarantees: every offset is ``rank * (this rank's heads) *
    (that family's per-head width)``, so rank 0 is heads 0..31 in all three. A
    rank holding one family's heads and another family's would mix heads at
    exactly the right total width, which no shape check could see.

    Each slice is CLONED rather than kept as a view. Item (2)'s doctored arm
    mutates one rank's slice, and a view would write that through into the
    reference's own weights.
    """
    heads = DECLARED_HEADS // world_size
    q_rows = heads * (NOPE_WIDTH + ROPE_WIDTH)
    kv_rows = heads * (NOPE_WIDTH + V_WIDTH)
    o_cols = heads * V_WIDTH
    return {
        # Replicated: MLA keeps ONE compressed latent per token, not one per head,
        # so neither latent projection has a head axis to split.
        "q_a_proj": raw["q_a_proj"].clone(),
        "kv_a_proj_with_mqa": raw["kv_a_proj_with_mqa"].clone(),
        "q_b_proj": raw["q_b_proj"][rank * q_rows : (rank + 1) * q_rows, :].clone(),
        "kv_b_proj": raw["kv_b_proj"][rank * kv_rows : (rank + 1) * kv_rows, :].clone(),
        "o_proj": raw["o_proj"][:, rank * o_cols : (rank + 1) * o_cols].clone(),
    }


def _reset_projection_counters() -> None:
    """Reset the projection seam's own dispatch counters.

    THIS FILE DID NOT READ THEM BEFORE. `read_counters` above covers the sparse and
    absorb seams, which is what items (b) and (d) need; the PROJECTION seam's pair is
    what this increment's route predicate names, so it is taken explicitly here rather
    than inherited. That gap is the one recorded against `-051` as §87 M1.
    """
    from vllm_neuron.functional.attention import mla_projections

    mla_projections.reset_mla_projection_dispatch_counters()


def _read_projection_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the projection seam since the reset.

    The fallback member can only ever read 0 -- the module has no torch path, by its
    own accessor's docstring -- so it is reported as a reading whose zero is
    structural, never staged as a control that could fire.
    """
    from vllm_neuron.functional.attention import mla_projections

    return mla_projections.mla_projection_dispatch_counters()


class _CountedTwoRankGroup:
    """The injected tensor-parallel coordinator: it COUNTS, and it really sums.

    WHAT IT STANDS FOR. In production ``project_output`` calls ``all_reduce`` on
    ``get_tp_group()``'s coordinator and the two ranks' partials meet inside the
    collective. A pytest item has one process, so the two ranks run one after the
    other and this object is what makes their partials meet: on the FIRST pass it
    records each partial and leaves the tensor alone, and on the SECOND pass it
    adds the recorded partial back in place. Rank 1's returned value is therefore
    the fully reduced one, which is exactly what rank 1 would return on hardware.
    Rank 0's is not, and is not compared against anything.

    IT HONOURS THE CONTRACT THE PRODUCTION LINE ASSUMES. ``project_output`` calls
    ``all_reduce`` as a statement and discards the return, the form all 18 shipped
    row-parallel sites use, so it depends on the sum being written THROUGH the
    argument. ``add_`` does that, and the tensor is also returned, so the
    production line would be correct under either reading of vLLM's contract.

    ``world_size`` is a plain attribute because that is all the production code
    reads off the group besides ``all_reduce``.
    """

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
    """Both ranks, in order, through the PRODUCTION decode path. Returns rank 1's.

    Rank 0 first so its partials are on record before rank 1 reduces. Each rank
    gets its own clone of the cache: the KV latent is replicated, so both ranks
    write the same latent to the same slot, and a shared cache would only hide a
    write that differed.

    Returns ``(rank 1's output, [(nki, fallback) per rank])`` -- the second value is
    the route predicate's reading, taken per rank because that is what the predicate
    compares.
    """
    outputs = {}
    counters = []
    for rank in range(PERRANK_WORLD):
        group.recording = rank == 0
        module = _bare_attention()
        assert module._heads_per_rank() == PERRANK_HEADS
        weights = _rank_slices(raw, rank, PERRANK_WORLD)
        if doctor_rank_1_o_proj_by_one_head and rank == 1:
            # THE DOCTORED SLICE: the right shape, taken ONE HEAD too early. This
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

    THE EXPECTATION IS COMPUTED FROM THE DECLARED CONSTANTS, never asked of the
    code: the three head-bearing widths are ``PERRANK_HEADS`` times their own
    per-head width, and the remaining widths are the config's own scalars. Reading
    them from ``projection_widths()`` and dividing would have passed whatever that
    method said.
    """
    model_fp8 = _model_module()
    cfg = declared_config()
    module = _bare_attention()
    q_head_width = NOPE_WIDTH + ROPE_WIDTH
    kv_head_width = NOPE_WIDTH + V_WIDTH

    def widths() -> dict[str, tuple[int, int]]:
        return {name: (idim, odim) for name, idim, odim in module.projection_widths()}

    # -- world size 1: byte-identical to what this file measured before -100.
    assert module._heads_per_rank() == DECLARED_HEADS
    at_one = widths()
    expected_one = {
        "q_a_proj": (int(cfg.hidden_size), int(cfg.q_lora_rank)),
        "q_b_proj": (int(cfg.q_lora_rank), DECLARED_HEADS * q_head_width),
        "kv_a_proj_with_mqa": (int(cfg.hidden_size), LATENT_RANK + ROPE_WIDTH),
        "kv_b_proj": (LATENT_RANK, DECLARED_HEADS * kv_head_width),
        "o_proj": (DECLARED_HEADS * V_WIDTH, int(cfg.hidden_size)),
    }
    say("PERRANK_1_WIDTHS_AT_WORLD_1", at_one)
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
    say("PERRANK_1_WIDTHS_AT_WORLD_2", at_two)
    assert at_two == expected_two

    # Which widths moved, stated as a reading rather than left to the dicts above.
    moved = sorted(name for name in at_one if at_one[name] != at_two[name])
    say("PERRANK_1_WIDTHS_THAT_MOVED", moved)
    assert moved == ["kv_b_proj", "o_proj", "q_b_proj"]
    assert at_one["q_a_proj"] == at_two["q_a_proj"]
    assert at_one["kv_a_proj_with_mqa"] == at_two["kv_a_proj_with_mqa"]

    # RANK INVARIANT. A width is a function of the world size alone; if it read the
    # rank, two ranks would expect different shapes from one checkpoint.
    monkeypatch.setattr(model_fp8, "_resolve_rank", lambda: 1)
    say("PERRANK_1_WIDTHS_AT_RANK_1", widths() == at_two)
    assert widths() == at_two

    # FIRING CONTROL: the halving is not a hard-coded halving. At a world size that
    # does not divide the head count the widths FLOOR, and the floored answer is
    # not the divided one -- so an implementation that simply halved would be
    # caught here.
    _patch_world(monkeypatch, PERRANK_UNEVEN_WORLD)
    uneven_heads = DECLARED_HEADS // PERRANK_UNEVEN_WORLD
    at_three = widths()
    say("PERRANK_1_UNEVEN_WORLD", PERRANK_UNEVEN_WORLD, uneven_heads, at_three)
    assert module._heads_per_rank() == uneven_heads
    assert at_three["q_b_proj"][1] == uneven_heads * q_head_width
    assert at_three["q_b_proj"][1] != DECLARED_HEADS * q_head_width // PERRANK_UNEVEN_WORLD
    assert at_three != at_two

    # CROSS-CHECK: the width a LOADER slices to is the width this class expects.
    # Both floor through ``_per_rank``, so they agree by construction -- asserted
    # rather than assumed, because a divergence would refuse a correct checkpoint.
    _patch_world(monkeypatch, PERRANK_WORLD)
    for leaf, name, axis in (
        ("q_b_proj_weight", "q_b_proj", 1),
        ("kv_b_proj_weight", "kv_b_proj", 1),
        ("o_proj_weight", "o_proj", 0),
    ):
        geometry = model_fp8._shard_geometry_for(module, leaf, PERRANK_WORLD)
        say("PERRANK_1_LOADER_AGREES_WITH_ACCESSOR", leaf,
            geometry.shard_size, at_two[name][axis])
        assert geometry.shard_size == at_two[name][axis]


# --------------------------------------------------------------------------- #
def test_perrank_numerics_the_reduced_sharded_decode_equals_the_unsharded_output(
    monkeypatch,
) -> None:
    """Conjunct (2): two ranks' reduced decode output equals the unsharded output.

    THE PRODUCTION REDUCTION SITE IS WHAT RUNS. Nothing in ``project_output`` is
    replaced: the group it resolves is injected, its ``all_reduce`` is counted, and
    the assertion is on the value that call wrote. So this item measures the
    collective's PLACE in the chain -- after the projection, before the cast back
    to the model dtype -- and not just that a sum was available somewhere.

    WHY THE REFERENCE IS THE RIGHT ONE. Each rank contracts its own heads against
    its own slice of ``o_proj``'s columns and produces a partial sum at the full
    output width. Their sum is the same 16,384-term contraction the unsharded path
    performs, reassociated -- so the two agree to the registered pair, and a
    partitioning that mixed heads does not.

    THE PAIR IS THE REGISTERED ONE, reused through ``report_and_check`` exactly as
    item (b-ii) does. This increment registers no tolerance of its own.
    """
    import time

    model_fp8 = _model_module()

    # -- ARM 1: the unsharded reference, at world size 1, with the REAL guard.
    reference_module, raw, gains, gen = build_attention()
    assert reference_module._heads_per_rank() == DECLARED_HEADS
    cache = seeded_cache(reference_module, gen)
    hidden, selection, scale = decode_inputs(reference_module, gen)

    # The guard, read directly: at world size 1 there is no group, so the
    # production line below cannot reduce and no vllm symbol is imported. This is
    # the reading that makes the eight items above a control rather than a hope.
    say("PERRANK_2_GUARD_AT_WORLD_1", model_fp8._resolve_tp_group())
    assert model_fp8._resolve_tp_group() is None

    started = time.perf_counter()
    _reset_projection_counters()
    reference = run_decode_steps(
        reference_module, cache.clone(), hidden, selection, scale
    )
    whole_counters = _read_projection_counters()
    say("PERRANK_2_REFERENCE_SECONDS", f"{time.perf_counter() - started:.3f}",
        "shape", tuple(reference.shape), reference.dtype)
    say("PERRANK_2_R2_PROJECTION_COUNTERS_AT_WORLD_1", whole_counters)

    # -- ARM 2: two ranks, each on its own heads, reducing through the real site.
    group = _CountedTwoRankGroup()
    _patch_world(monkeypatch, PERRANK_WORLD)
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)

    started = time.perf_counter()
    reduced, per_rank_counters = _run_sharded(
        group, raw, gains, cache, hidden, selection, scale
    )
    say("PERRANK_2_SHARDED_SECONDS", f"{time.perf_counter() - started:.3f}")

    # THE ROUTE PREDICATE (D13 form R-2), which this block declares and which no item
    # in this file read before: partitioning changes WIDTHS, never the number of
    # projection dispatches, so each rank's count equals the unsharded count. The
    # number itself is printed rather than typed -- an expectation typed here would be
    # a transcription of whatever the chain happens to do.
    say("PERRANK_2_R2_PROJECTION_COUNTERS_PER_RANK", per_rank_counters)
    for rank, (nki, fallback) in enumerate(per_rank_counters):
        assert nki == whole_counters[0], (
            f"rank {rank} dispatched {nki} projections against {whole_counters[0]} "
            f"unsharded; partitioning must change widths, not the call count"
        )
        assert fallback == 0
    assert whole_counters[1] == 0
    say("PERRANK_2_R2_TORCH_FALLBACK_ZERO_IS_STRUCTURAL",
        "the projection module has no torch path; its accessor's own docstring says "
        "this counter can only ever read 0, so it is reported, not staged")

    # THE COUNTED READING. Once per decode step per rank, and not once more: a
    # reduction inside the per-head loop, or one left in a helper that runs twice,
    # would not read 6.
    say("PERRANK_2_REDUCE_CALLS", group.calls,
        "expected", DECODE_STEPS * PERRANK_WORLD, "shapes", set(group.shapes))
    assert group.calls == DECODE_STEPS * PERRANK_WORLD
    assert set(group.shapes) == {(BATCH, int(declared_config().hidden_size))}

    for step in range(DECODE_STEPS):
        report_and_check(
            f"PERRANK_2_STEP_{step}",
            reduced[step : step + 1],
            reference[step : step + 1],
        )
    say("PERRANK_2_STEPS_AGREEING", f"{DECODE_STEPS}/{DECODE_STEPS}")

    # -- THE DOCTORED CONTROL. One rank's ``o_proj`` slice, right shape, taken one
    #    head too early. ONE step rather than three, because the control's job is
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
    gap = (
        doctored.to(torch.float32) - one_step_reference.to(torch.float32)
    ).abs().max().item()
    fired = not torch.allclose(
        doctored.to(torch.float32),
        one_step_reference.to(torch.float32),
        rtol=RTOL,
        atol=atol_for(one_step_reference),
    )
    say("PERRANK_2_CONTROL_REJECTS_A_ONE_HEAD_SHIFT", fired,
        f"max_abs_diff={gap:.10g}",
        f"allowance_atol={atol_for(one_step_reference):.10g}",
        "reduce_calls", doctored_group.calls)
    assert doctored_group.calls == PERRANK_WORLD
    assert fired


# --------------------------------------------------------------------------- #
def test_perrank_load_the_shard_table_cuts_the_five_mla_families(monkeypatch) -> None:
    """Conjunct (3): the three families carry a declared shard, the two latents none.

    THE ATTACHMENT SITE IS WHAT IS ASKED. ``_shard_geometry_for`` is the one reader
    of the shard table and is what ``_materialise_declared_parameters`` calls for
    every declared parameter, so asking it -- with a real module instance and a
    real world size -- is asking the code path the load takes, not the table.

    AND THE CUT IS FOLLOWED THROUGH. Declaring a geometry proves nothing on its own,
    so the second half of this item slices a full weight set at the declared extents
    and hands the slices to the production preparation, which checks every one
    against ``projection_widths()`` and refuses a mismatch. The wrong-slice arm
    shows that check can refuse.
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

    # -- world size 1: nothing is sharded, which is the control arm. The SAME call
    #    distinguishes rather than agrees: five Nones here, three geometries below.
    at_one = geometry_at(1)
    say("PERRANK_3_GEOMETRY_AT_WORLD_1", {k: v for k, v in at_one.items()})
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
        # Row-parallel: the head width is this projection's INPUT.
        "o_proj_weight": (1, PERRANK_HEADS * V_WIDTH, PERRANK_WORLD),
    }
    say("PERRANK_3_GEOMETRY_AT_WORLD_2", triples)
    assert triples == expected
    say("PERRANK_3_SHARDED_FAMILIES",
        sorted(k for k, v in triples.items() if v is not None),
        "replicated",
        sorted(k for k, v in triples.items() if v is None))

    # THE SHARDS TILE THE TENSOR EXACTLY. ``shard_size * num_shards`` is the full
    # extent, so the two ranks' slices cover the checkpoint's dimension with no gap
    # and no overlap -- the property an off-by-one extent breaks and a shape check
    # on one rank alone would not notice.
    for leaf, full_extent in (
        ("q_b_proj_weight", DECLARED_HEADS * (NOPE_WIDTH + ROPE_WIDTH)),
        ("kv_b_proj_weight", DECLARED_HEADS * (NOPE_WIDTH + V_WIDTH)),
        ("o_proj_weight", DECLARED_HEADS * V_WIDTH),
    ):
        geometry = at_two[leaf]
        say("PERRANK_3_TILES", leaf, geometry.shard_size, geometry.num_shards,
            full_extent)
        assert geometry.shard_size * geometry.num_shards == full_extent

    # -- THE CUT, FOLLOWED THROUGH. A full weight set, sliced at those extents, is
    #    accepted by the production preparation for both ranks.
    _, raw, gains, _ = build_attention()
    _patch_world(monkeypatch, PERRANK_WORLD)
    for rank in range(PERRANK_WORLD):
        sharded = _bare_attention()
        weights = _rank_slices(raw, rank, PERRANK_WORLD)
        shapes = {name: tuple(w.shape) for name, w in weights.items()}
        say("PERRANK_3_INSTALLED_SHAPES", rank, shapes)
        _install(sharded, weights, gains)
        assert tuple(sharded._prepared_weight("o_proj").shape) == (
            PERRANK_HEADS * V_WIDTH,
            int(cfg.hidden_size),
        )
        for name, expected_heads, contraction, out_features in sharded.absorb_widths():
            say("PERRANK_3_ABSORB_WIDTH", rank, name, expected_heads, contraction,
                out_features)
            assert expected_heads == PERRANK_HEADS

    # FIRING CONTROL: the preparation can refuse. A rank handed the FULL weight --
    # the shape it would have had before this increment -- is rejected by name.
    wrong = _bare_attention()
    with pytest.raises(ValueError) as refusal:
        _install(wrong, {**_rank_slices(raw, 0, PERRANK_WORLD),
                         "q_b_proj": raw["q_b_proj"].clone()}, gains)
    say("PERRANK_3_CONTROL_REFUSES_A_FULL_WIDTH_SLICE", str(refusal.value)[:120])
    assert "q_b_proj_weight is" in str(refusal.value)
