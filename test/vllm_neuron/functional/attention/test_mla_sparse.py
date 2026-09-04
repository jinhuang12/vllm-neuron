# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-040` and `inc-glm53f-041` -- the sparse MLA latent
attention kernel and its tiling path.

FOURTEEN tests and NO `parametrize` decorator in this file. Four carry `geometry` in
their name and are `-040`'s DECLARED acceptance selection; eight carry `width` and are
`-041`'s; two carry neither, and screen the module rather than measure the kernel. Each
test prints its counted value and names the component whose behaviour it certifies.

The declared command, and the only one whose result the plan block quotes::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_sparse.py \
        -k geometry -v -s -p no:cacheprovider

THE DECLARED CASE IS ONE CASE, AND IT IS AT R == 0. The plan block says so in as many
words: the substrate asserts `0 < R <= 128`, removing that is the increment's whole
content, and "a run at R > 0 would prove nothing about this target". So `DECLARED_CASE`
below is a single tiny shape at this checkpoint's own `L = 512` and `H = 64` with no
RoPE half at all, and the route predicate is read over that one case.

WHAT THIS FILE ANSWERS AND WHAT IT DOES NOT. It answers "does the kernel compute
sparse latent attention at a geometry the substrate refuses". Whether the decode path
is wired to it is `inc-glm53f-042`'s question, in its own file. Whether an arbitrary
latent width tiles correctly is `inc-glm53f-041`'s, also in this file but as its own
items -- the `width` selection at the end of this file, added by that increment.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as MS

#: THE ONE DECLARED CASE. `latent` and `heads` are this checkpoint's own
#: `kv_lora_rank` and `num_attention_heads`; `rope` is its `qk_rope_head_dim`, which
#: is the value 0 and not a placeholder. `topk` is the smallest admissible selected-row
#: count and `s_kv` is deliberately larger than it, so the selection is a real subset
#: of the cache rather than the whole cache in some order.
#:
#: Line numbers are DELIBERATELY ABSENT from the config citations, on the landed
#: `test_mla_projections_kernel.py` precedent, which records why: the plan's own
#: `config.py` line numbers are stale, so copying them here would put a false citation
#: on this increment's added lines. The widths themselves are confirmed against the
#: published checkpoint, which is the durable authority and needs no line number.
DECLARED_CASE = dict(seq=1, heads=64, latent=512, topk=128, s_kv=256, rope=0)

#: Section 3's bf16 module-comparison threshold. READ FROM THE PLAN, NOT AUTHORED
#: HERE: this increment authors no criterion, tolerance or threshold, so these two
#: numbers are quoted and never adjusted to reach green.
RTOL = 1e-2
ATOL = 1e-5

#: The refused substrate member's own symbol and vendor module path, named HERE and
#: nowhere in the shipped module -- which is exactly what the source screen counts.
VENDOR_SOURCE_TERMS = (
    "mla_sparse_attention_cte_kernel",
    "mla_vupmx_oproj_cte_kernel",
    "nkilib.experimental",
    "mla_common_cte",
)

#: The substrate member the geometry tests interrogate live. Imported by string so
#: this file states the dependency once and a missing substrate reads as a skip with a
#: reason rather than a collection error.
VENDOR_MODULE = "nkilib.experimental.mla.deepseek.mla_sparse_attention_cte"
VENDOR_KERNEL = "mla_sparse_attention_cte_kernel"

#: The member's parameter validator, which is where its RoPE-width bound is written --
#: a different module from the kernel above, and the one
#: `probe-040-substrate-r4.out` read the bound out of.
VENDOR_VALIDATOR_MODULE = "nkilib.experimental.mla.deepseek.mla_validate_params"
VENDOR_VALIDATOR_FUNCTION = "_validate_mla_attention_inputs"

#: The excluded region of the P13 screen, named explicitly and not by heuristic.
#: Section 4 allows a `functional/` module a torch path that is (a) the test oracle OR
#: (b) the constraint-violation fallback the pin's dispatchers carry. Only (a) is
#: present, and (b)'s absence is asserted rather than assumed.
ORACLE_NAME = "mla_sparse_attention_torch_oracle"

#: torch entry points that would constitute an attention route if they appeared
#: outside the oracle. `softmax` is the giveaway: a fallback needs it and the kernel
#: path does not.
TORCH_ATTENTION_ATTRS = frozenset(
    {"softmax", "scaled_dot_product_attention", "matmul", "einsum", "bmm"}
)


def say(*parts: object) -> None:
    """Print one counted reading. `-s` keeps it in the transcript.

    Note for whoever greps the transcript: pytest writes a progress marker with NO
    trailing newline after each test, so the first line printed by every test after
    the first can be prefixed by it. Match with `grep -o`, never with a `^` anchor --
    an anchored pattern drops those lines silently, a defect this campaign has already
    been bitten by once.
    """
    print(" ".join(str(p) for p in parts), flush=True)


def module_source() -> str:
    return inspect.getsource(MS)


def make_case(seq: int, heads: int, latent: int, topk: int, s_kv: int, rope: int,
              seed: int = 40):
    """Inputs for one case, seeded from the shape so a failure reproduces from it.

    The selected rows are a PERMUTATION-derived subset, distinct per query, so a
    gather that ignored its index tensor could not agree with the oracle.
    """
    gen = torch.Generator().manual_seed(seed)
    q_lift = torch.randn(seq, heads, latent, generator=gen, dtype=torch.float32) * 0.05
    c_kv = torch.randn(s_kv, latent, generator=gen, dtype=torch.float32) * 0.05
    idx = torch.stack(
        [torch.randperm(s_kv, generator=gen)[:topk] for _ in range(seq)]
    ).to(torch.int32)
    q_pe = k_pe = None
    if rope > 0:
        q_pe = torch.randn(seq, heads, rope, generator=gen, dtype=torch.float32) * 0.05
        k_pe = torch.randn(s_kv, rope, generator=gen, dtype=torch.float32) * 0.05
    return q_lift, c_kv, idx, q_pe, k_pe


def sparse_mla_torch_reference(q_lift, c_kv, idx, softmax_scale, q_pe=None, k_pe=None):
    """An INDEPENDENTLY WRITTEN reference, not the module's oracle.

    Written here rather than imported from the module under test: an oracle the module
    supplies could agree with the kernel through a mistake the two share, and the
    question is whether the KERNEL matches torch -- not whether the module agrees with
    itself. Three things are deliberately done differently from the module's oracle:
    the scores go through `torch.einsum` with the contraction named explicitly, the
    softmax is spelled out as a shifted exponential over an explicit sum rather than
    delegated to `torch.softmax`, and the value pass is a second `einsum`. So a wrong
    contraction, a missing max-shift or a wrong denominator would have to be made
    twice, in two different notations, to hide.
    """
    q = q_lift.to(torch.float64)
    cache = c_kv.to(torch.float64)
    rows = idx.to(torch.int64)
    seq = q.shape[0]
    out = []
    for s in range(seq):
        gathered = cache[rows[s]]                                  # [K, L]
        scores = torch.einsum("hl,kl->hk", q[s], gathered)
        if q_pe is not None:
            rope_rows = k_pe.to(torch.float64)[rows[s]]             # [K, R]
            scores = scores + torch.einsum(
                "hr,kr->hk", q_pe.to(torch.float64)[s], rope_rows
            )
        scaled = scores * softmax_scale
        shifted = scaled - scaled.max(dim=-1, keepdim=True).values
        expd = torch.exp(shifted)
        weights = expd / expd.sum(dim=-1, keepdim=True)
        out.append(torch.einsum("hk,kl->hl", weights, gathered))
    return torch.stack(out).to(torch.float32)


def declared_scale() -> float:
    """The softmax scale for the declared case: the latent rank's inverse square root.

    DERIVED from the case's own latent rank rather than typed, so the two cannot drift
    apart. This is not a registered comparator value -- it is an input to a numeric
    agreement test, and the plan block confirms this increment touches no tolerance,
    threshold or registered value.
    """
    return float(DECLARED_CASE["latent"]) ** -0.5


# --------------------------------------------------------------------------- #
# The four `geometry` items -- the increment's declared acceptance selection.
# --------------------------------------------------------------------------- #
def test_geometry_the_no_rope_declared_case_matches_the_torch_oracle() -> None:
    """GEOMETRY 1 of 4 -- the kernel computes sparse latent attention at R == 0.

    CERTIFYING COMPONENT: the `wrap_nki` seam `MS.mla_sparse_attention` and, behind
    it, the authored kernel `mla_sparse_attention_nope_kernel`.

    This is the plan block's 1/1 tiny case, at `L == 512` and `R == 0`. THE ROUTE
    PREDICATE IS READ HERE (D13 form R-1) because this increment authors the seam it
    counts: the counter is reset at the start of the case and read at its end, so the
    reading is one dispatch for one declared case and cannot pick up another item's.

    THE COMPARISON OWES A CONTROL, and getting that control right took two rounds.
    A tolerance test against an oracle fed THE SAME inputs cannot see whether the
    kernel gathered the rows it was asked for -- a kernel that ignored its index
    tensor and read the first K cache rows would still produce a valid-looking softmax
    over cache rows. Round 1's control built its wrong answer by ROLLING each index
    row, and it could not fire: it read exactly the same error as the real comparison,
    `7.451e-09` (`investigation-040.md`, FOUND 1). The reason is a property of the
    computation and not a bug -- a softmax-weighted sum over a SET of rows does not
    depend on the order the rows are listed in, which is why the substrate's own gather
    documents itself as order-invariant. So this item now takes TWO readings:

      * the permutation reading, which is expected to AGREE and is stated as the
        property it is rather than mistaken for a control; and
      * the control proper, which changes the SET -- one selected row is replaced by a
        cache row that is not in the selection -- and must DISAGREE outside tolerance.

    THE ROUTE-PREDICATE READINGS ARE PRINTED BEFORE THE NUMERIC COMPARISON, and that
    ordering is load-bearing rather than tidy. The acceptance harness mutates the
    kernel and requires that the numeric item go red WHILE the dispatch count still
    reads 1/1 -- that is how it tells "the mutation broke the arithmetic" from "the
    mutation stopped the kernel being reached". In round 1 these readings were printed
    after the numeric assert, so a mutated run took them with it and the harness had
    nothing to check.
    """
    case = dict(DECLARED_CASE)
    scale = declared_scale()
    say("G1_CERTIFYING_COMPONENT=mla_sparse_attention seam + "
        "mla_sparse_attention_nope_kernel in "
        "vllm_neuron/functional/attention/mla_sparse.py")
    say(f"G1_TOLERANCE rtol={RTOL} atol={ATOL} (plan section 3, quoted not authored)")
    say("G1_DECLARED_CASE " + " ".join(f"{k}={v}" for k, v in case.items()))
    say(f"G1_SOFTMAX_SCALE={scale:.6f} (derived from the case's latent rank)")

    q_lift, c_kv, idx, q_pe, k_pe = make_case(**case)
    assert q_pe is None and k_pe is None, (
        "the declared case must have NO RoPE half at all -- a zero-width RoPE tensor "
        "would be a different claim from the absence the checkpoint declares"
    )
    say(f"G1_ROPE_TENSORS_PRESENT={int(q_pe is not None)}  "
        f"(the checkpoint's qk_rope_head_dim is {MS.TARGET_ROPE_WIDTH})")

    # The gate must be OPEN, or the dispatch count below would read 0 for a reason
    # that has nothing to do with this kernel.
    gate = MS.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    )
    say(f"G1_CAN_RUN_KERNEL={gate}")
    assert gate, (
        "can_run_mla_sparse_attention read False, so the NKI route is closed and no "
        "dispatch count below would mean anything. Under the Tier N harness this "
        "means NKI_SIMULATOR=1 is not set in the invocation"
    )

    MS.reset_mla_sparse_dispatch_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    nki_dispatch, torch_fallback = MS.mla_sparse_dispatch_counters()

    expected_shape = (case["seq"], case["heads"], case["latent"])
    assert tuple(got.shape) == expected_shape, (
        f"kernel returned {tuple(got.shape)}, expected {expected_shape}"
    )
    assert bool(torch.isfinite(got).all()), (
        "the result holds a non-finite value, which is what a tile the kernel never "
        "wrote looks like"
    )

    # D13 form R-1, read over the one declared case -- and read HERE, before the
    # numeric comparison, so a mutated kernel still leaves these lines in the
    # transcript for the harness's mutation rows to check.
    say(f"G1_NKI_DISPATCH={nki_dispatch}/1")
    say(f"G1_TORCH_FALLBACK={torch_fallback}")
    assert nki_dispatch == 1, (
        f"the declared case must reach the NKI seam exactly once; got {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"this module has no torch attention route, so the fallback counter must read "
        f"0; got {torch_fallback}. A nonzero reading means a fallback was added"
    )

    ref = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    err = float((got - ref).abs().max())
    say(f"G1_MAXABS_VS_INDEPENDENT_ORACLE={err:.3e}")
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)
    say("G1_CASES_AGREED=1/1")

    # READING: the selection is a SET, so reordering it must not move the answer. This
    # is expected to AGREE. It is stated because round 1 mistook it for a control.
    rolled = torch.roll(idx, shifts=1, dims=1)
    assert not torch.equal(rolled, idx), (
        "the rolled index equals the original, so this reading measures nothing"
    )
    permuted = sparse_mla_torch_reference(q_lift, c_kv, rolled, scale)
    perm_err = float((got - permuted).abs().max())
    say(f"G1_PERMUTATION_INVARIANCE_MAXABS={perm_err:.3e}  (expected to AGREE)")
    torch.testing.assert_close(got, permuted, rtol=RTOL, atol=ATOL)

    # THE CONTROL. Change the SET, not its order: replace one selected row with a cache
    # row the selection does not hold. This must DISAGREE, or the comparison above
    # cannot see WHICH rows were gathered.
    substituted = idx.clone()
    for s in range(case["seq"]):
        selected = set(int(v) for v in idx[s].tolist())
        unused = [r for r in range(case["s_kv"]) if r not in selected]
        assert unused, (
            f"query {s} selects every cache row, so no substitution can change the "
            f"set; s_kv must exceed topk for this control to exist"
        )
        substituted[s, 0] = unused[0]
    changed = int((substituted != idx).sum())
    say(f"G1_CONTROL_SELECTED_ROWS_CHANGED={changed}/{case['seq']}")
    assert changed == case["seq"], "the substitution did not change the selection"
    wrong = sparse_mla_torch_reference(q_lift, c_kv, substituted, scale)
    control_err = float((got - wrong).abs().max())
    say(f"G1_CONTROL_MAXABS_AGAINST_A_DIFFERENT_SELECTION={control_err:.3e}")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, wrong, rtol=RTOL, atol=ATOL)
    say("G1_CONTROL_FIRES=1")
    assert control_err > ATOL, (
        f"the control's disagreement is {control_err:.3e}, which is inside the "
        f"tolerance -- so the comparison above cannot see WHICH rows were gathered "
        f"and this item would pass on a kernel that ignored its index tensor"
    )


def test_geometry_the_substrate_refuses_this_checkpoint_on_two_counts() -> None:
    """GEOMETRY 2 of 4 -- the refusals this increment exists to remove, ATTRIBUTED.

    CERTIFYING COMPONENT: the substrate's sparse-MLA member as installed, exercised at
    two RoPE widths, plus its validator's own source line; then this module's
    admissibility check on the same geometry.

    THIS IS THE ITEM THAT MAKES THE INCREMENT NON-VACUOUS. Adapting a substrate member
    is only justified if the member refuses, and "it refuses" is a claim about someone
    else's installed code -- so it is re-measured here every run, with the image's own
    messages quoted into the transcript, rather than cited from a plan bullet.

    ONE REFUSAL IS NOT ENOUGH, and round 1 of this acceptance is why. It recorded the
    R == 0 refusal and claimed it came from the member's `0 < R` validator assert. What
    the host actually said was `shape must be a tuple of positive integers, got:
    (1, 1, 64, 0)` -- the framework declining a zero-extent tensor one layer earlier
    (`investigation-040.md`, FOUND 2). The refusal was real; the reason recorded was
    not the world's reason. A single reading cannot tell those apart, so this item takes
    the DIFFERENTIAL: the same widths at R == 0 and at an admissible R == 64. Both
    refuse, on DIFFERENT messages, and that is what attributes each refusal to its own
    cause -- the first to the RoPE width, the second to the gather's hardware
    generation. The validator's own line is then read out of the installed source, so
    the `0 < R` bound is quoted rather than recalled.

    If a later substrate release lifted EITHER count, this item goes red and the
    increment's justification is re-opened -- which is the correct outcome and not a
    nuisance.
    """
    say("G2_CERTIFYING_COMPONENT=the installed substrate sparse-MLA member, at two "
        "RoPE widths, plus its validator source")
    vendor = pytest.importorskip(
        VENDOR_MODULE,
        reason="the substrate sparse-MLA member is installed only on the Neuron host",
    )
    kernel = getattr(vendor, VENDOR_KERNEL)
    say(f"G2_SUBSTRATE_MEMBER={VENDOR_MODULE}.{VENDOR_KERNEL}")

    case = dict(DECLARED_CASE)
    b, s, h, ell = 1, case["seq"], case["heads"], case["latent"]
    s_kv, k = case["s_kv"], case["topk"]

    def attempt(rope: int) -> str:
        """Call the member at this RoPE width and return its refusal, or "" if none.

        bf16 throughout, because the member's validator requires it -- so a refusal
        that fires is about a width and not about a dtype.
        """
        args = (
            torch.zeros(b, s, h, ell, dtype=torch.bfloat16),
            torch.zeros(b, s, h, rope, dtype=torch.bfloat16),
            torch.zeros(b, s_kv, ell, dtype=torch.bfloat16),
            torch.zeros(b, s_kv, rope, dtype=torch.bfloat16),
            torch.zeros(b, s, k, dtype=torch.int32),
        )
        try:
            wrap_nki(kernel)(*args, float(ell) ** -0.5)
        except BaseException as exc:  # noqa: BLE001 -- the refusal IS the reading
            return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
        return ""

    at_zero = attempt(0)
    at_positive = attempt(64)
    say(f"G2_SUBSTRATE_REFUSED_AT_R_EQUALS_ZERO={int(bool(at_zero))}")
    say(f"G2_MESSAGE_AT_R_ZERO_VERBATIM={at_zero}")
    say(f"G2_SUBSTRATE_REFUSED_AT_R_EQUALS_64={int(bool(at_positive))}")
    say(f"G2_MESSAGE_AT_R_64_VERBATIM={at_positive}")

    assert at_zero, (
        "the substrate accepted this checkpoint's R == 0. That is one of the two "
        "constraints this increment exists to remove, so if it is gone the increment's "
        "justification must be re-derived -- report evidence_contradicts_design"
    )
    # The R == 0 refusal must be about the zero width, and the transcript must be able
    # to show which word carried it.
    assert "(1, 1, 64, 0)" in at_zero or "R must be in" in at_zero, (
        f"the member refused at R == 0, but not visibly because of the zero width, so "
        f"this item has not attributed the refusal: {at_zero}"
    )
    assert at_positive, (
        "the substrate ran at R == 64 on this gen3 target. Then its gather is not the "
        "second blocker this module's docstring claims, and that paragraph must be "
        "re-derived -- report evidence_contradicts_design"
    )
    assert "NeuronCore-v4" in at_positive or "gen4" in at_positive, (
        f"the member refused at R == 64, but not for the hardware-generation reason "
        f"this item attributes it to: {at_positive}"
    )
    differ = at_zero != at_positive
    say(f"G2_THE_TWO_REFUSALS_ARE_DIFFERENT={int(differ)}")
    assert differ, (
        "both widths produced the same message, so neither refusal is attributed to "
        "its own cause and this item measures one fact rather than two"
    )

    # The validator's own bound, READ from the installed source rather than recalled.
    # It lives in the member's parameter-validation module, which is where
    # `probe-040-substrate-r4.out` read it from -- not in the kernel module above.
    validator = pytest.importorskip(
        VENDOR_VALIDATOR_MODULE,
        reason="the substrate validator ships with the member, on the Neuron host",
    )
    say(f"G2_VALIDATOR_MODULE={VENDOR_VALIDATOR_MODULE}."
        f"{VENDOR_VALIDATOR_FUNCTION}")
    validator_src = inspect.getsource(
        getattr(validator, VENDOR_VALIDATOR_FUNCTION)
    )
    bound_lines = [
        " ".join(line.split()) for line in validator_src.splitlines()
        if "0 < R" in line or "R must be in" in line
    ]
    say(f"G2_VALIDATOR_ROPE_BOUND_LINES={len(bound_lines)}")
    for line in bound_lines:
        say(f"G2_VALIDATOR_SOURCE_VERBATIM={line}")
    assert bound_lines, (
        "the member's validator no longer carries a RoPE-width bound at all; the "
        "increment's first justification must be re-derived"
    )

    # And the same geometry, through this module: admissible, by the reading rather
    # than by assertion of intent.
    accepted = MS.can_run_mla_sparse_attention(
        torch.zeros(1), s, h, ell, 0, k, s_kv, float(ell) ** -0.5
    )
    say(f"G2_THIS_MODULE_ACCEPTS_THE_SAME_GEOMETRY={accepted}")
    assert accepted, (
        "this module refused the geometry the increment was written to serve"
    )


def test_geometry_the_gen3_gather_bound_is_measured_not_assumed() -> None:
    """GEOMETRY 3 of 4 -- the substrate's gather instruction is gen4, this target gen3.

    CERTIFYING COMPONENT: this image's Tensor Indirection validator, read at trn2.

    THE SECOND REASON THE MEMBER CANNOT BE CALLED, and the reason this module's gather
    is `nisa.nc_n_gather` rather than the indirection view the substrate uses. It is
    measured with the substrate's OWN index dtype: a first attempt with a uint32 index
    produced a dtype complaint instead of the version bound, which would have been a
    misleading reading to record. The transcript of that first attempt is kept in this
    campaign's `probe-040-indirect-r3.out`.

    WHY THIS IS AN ITEM AND NOT A COMMENT. "The substrate's gather is gen4" is the
    load-bearing reason a re-derivation was needed at all rather than a wrapper, and a
    reason nobody re-checks is a reason that quietly stops being true.
    """
    say("G3_CERTIFYING_COMPONENT=nki's tensor-indirection validator at "
        "NEURON_PLATFORM_TARGET_OVERRIDE=trn2")
    p_max = nl.tile_size.pmax

    @nki.jit
    def _indirection_gather(data_hbm, idx_hbm, out_hbm):
        s_kv = data_hbm.shape[2]
        k = idx_hbm.shape[2]
        data_sb = nl.ndarray((p_max, s_kv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=data_sb, src=nl.load(data_hbm[0], dtype=nl.float32))
        idx_sb = nl.ndarray((p_max, k), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_sb, src=nl.load(idx_hbm[0], dtype=nl.uint16))
        out_sb = nl.ndarray((p_max, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sb, src=data_sb.indirect(idx_sb, num_elem=k))
        nl.store(out_hbm[0], value=out_sb)
        return out_hbm

    s_kv, k = 32, 16
    data = torch.arange(p_max * s_kv, dtype=torch.float32).reshape(1, p_max, s_kv)
    idx = torch.zeros(1, p_max, k, dtype=torch.int16)
    out = torch.zeros(1, p_max, k, dtype=torch.float32)

    refused = False
    message = ""
    try:
        wrap_nki(_indirection_gather)(data, idx, out)
    except BaseException as exc:  # noqa: BLE001 -- the refusal IS the reading
        refused = True
        message = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    say(f"G3_INDIRECTION_GATHER_REFUSED_ON_THIS_TARGET={int(refused)}")
    say(f"G3_MESSAGE_VERBATIM={message}")
    assert refused, (
        "the indirection gather ran on this target. If it is available at gen3 then "
        "the substrate's gather is not the blocker this module's docstring says it "
        "is, and that paragraph must be re-derived -- report "
        "evidence_contradicts_design rather than editing the docstring to suit"
    )
    assert "NeuronCore-v4" in message or "gen4" in message, (
        f"the gather was refused, but not for the version reason this item claims: "
        f"{message}"
    )

    # THE CONTROL. The gen3 primitive this module actually uses must run on the same
    # target in the same process -- otherwise the refusal above could be a broken
    # harness rather than a version bound.
    @nki.jit
    def _gpsimd_gather(data_hbm, idx_hbm, out_hbm):
        s_kv = data_hbm.shape[2]
        k = idx_hbm.shape[2]
        data_sb = nl.ndarray((p_max, s_kv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=data_sb, src=nl.load(data_hbm[0], dtype=nl.float32))
        idx_sb = nl.ndarray((p_max, k), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_sb, src=nl.load(idx_hbm[0], dtype=nl.uint32))
        out_sb = nl.ndarray((p_max, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=out_sb, data=data_sb, indices=idx_sb)
        nl.store(out_hbm[0], value=out_sb)
        return out_hbm

    want_cols = [5, 3, 31] + [0] * (k - 3)
    idx32 = torch.tensor(want_cols, dtype=torch.int32).repeat(p_max, 1).unsqueeze(0)
    out32 = torch.zeros(1, p_max, k, dtype=torch.float32)
    wrap_nki(_gpsimd_gather)(data, idx32, out32)
    # data[0, r, j] == r * s_kv + j, so a correct gather on partition r returns
    # r * s_kv + the requested column. Checked on two different partitions, because a
    # gather that serviced only partition 0 would satisfy one of them.
    for row in (0, 7):
        expected = [row * s_kv + c for c in want_cols[:3]]
        got_row = [int(v) for v in out32[0, row, :3].tolist()]
        say(f"G3_GEN3_GATHER_PARTITION_{row}={got_row} EXPECTED={expected}")
        assert got_row == expected, (
            f"the gen3 gather returned {got_row} on partition {row}, expected "
            f"{expected}; the control for the refusal above does not hold"
        )
    say("G3_CONTROL_FIRES=1  (the gen3 primitive runs where the gen4 one refuses)")


def test_geometry_every_bound_comes_from_the_image_and_every_refusal_fires() -> None:
    """GEOMETRY 4 of 4 -- the module's four tile bounds are this image's, and its
    admissibility check refuses each inadmissible axis BY NAME.

    CERTIFYING COMPONENT: the module's tile constants and
    `_require_admissible`, the only place a geometry is refused.

    WHY THE BOUNDS ARE READ AND NOT RESTATED. Each of the four constants names a
    hardware axis extent. Asserting them against `nl.tile_size` means an image change
    reddens this item instead of silently loosening the kernel -- and it also means the
    numbers in the module are not this increment's invention, which is the property the
    plan asks a kernel increment to be able to show.

    WHY THE REFUSALS ARE COUNTED. A geometry check that raised for everything would
    also pass a test that only asked "did it raise". So each refusal is matched against
    a fragment of ITS OWN message, and the admissible declared case is run through the
    same check to show it does NOT raise -- the control for the five refusals.
    """
    say("G4_CERTIFYING_COMPONENT=MS.{LATENT_TILE,KEY_CHUNK,HEAD_MAX,MOVING_MAX} and "
        "MS._require_admissible")
    bounds = (
        ("LATENT_TILE", MS.LATENT_TILE, nl.tile_size.pmax, "nl.tile_size.pmax"),
        ("KEY_CHUNK", MS.KEY_CHUNK, nl.tile_size.pmax, "nl.tile_size.pmax"),
        ("HEAD_MAX", MS.HEAD_MAX, nl.tile_size.gemm_stationary_fmax,
         "nl.tile_size.gemm_stationary_fmax"),
        ("MOVING_MAX", MS.MOVING_MAX, nl.tile_size.gemm_moving_fmax,
         "nl.tile_size.gemm_moving_fmax"),
    )
    for name, declared, from_image, provenance in bounds:
        say(f"G4_BOUND {name}={declared} IMAGE_{provenance}={from_image}")
        assert declared == from_image, (
            f"{name} is {declared} but this image says {provenance} is {from_image}; "
            f"the kernel's tiling no longer matches the hardware it tiles for"
        )
    say(f"G4_BOUNDS_AGREED={len(bounds)}/{len(bounds)}")

    case = dict(DECLARED_CASE)
    scale = declared_scale()

    # THE CONTROL FOR THE FIVE REFUSALS: the declared case must pass this same check.
    MS._require_admissible(case["seq"], case["heads"], case["latent"], case["rope"],
                          case["topk"], case["s_kv"], scale)
    say("G4_THE_DECLARED_CASE_IS_ADMISSIBLE=1  (the control: the check is not a "
        "blanket refusal)")

    # And R == 0 specifically, since that is the bound the increment removes.
    MS._require_admissible(case["seq"], case["heads"], case["latent"], 0,
                           case["topk"], case["s_kv"], scale)
    say("G4_ROPE_WIDTH_ZERO_IS_ADMISSIBLE=1")

    inadmissible = (
        ("heads_past_the_stationary_axis",
         dict(heads=MS.HEAD_MAX + 1), "stationary free axis"),
        ("latent_not_positive",
         dict(latent=0), "latent rank must be positive"),
        ("topk_not_a_multiple_of_the_key_chunk",
         dict(topk=MS.KEY_CHUNK + 2), "multiple of it"),
        ("non_positive_softmax_scale",
         dict(softmax_scale=0.0), "must be positive"),
        ("negative_rope_width",
         dict(rope=-1), "Zero IS admissible"),
    )
    fired = 0
    for name, override, fragment in inadmissible:
        args = dict(seq=case["seq"], heads=case["heads"], latent=case["latent"],
                    rope=case["rope"], topk=case["topk"], s_kv=case["s_kv"],
                    softmax_scale=scale)
        args.update(override)
        with pytest.raises(MS.MlaSparseAttentionError) as caught:
            MS._require_admissible(**args)
        message = " ".join(str(caught.value).split())
        assert fragment in message, (
            f"{name} raised, but not with its own message: expected a mention of "
            f"{fragment!r}, got {message!r}"
        )
        fired += 1
        say(f"G4_REFUSAL {name} FIRED=1 MESSAGE_NAMES_ITS_OWN_AXIS=1")
    say(f"G4_REFUSALS_FIRED={fired}/{len(inadmissible)}")
    assert fired == len(inadmissible)


# --------------------------------------------------------------------------- #
# The two screening items. They carry no `geometry` in their name, so the declared
# selection does not collect them -- they screen the SOURCE and measure no kernel.
# --------------------------------------------------------------------------- #
def test_the_module_carries_no_torch_attention_route_and_no_vendor_symbol() -> None:
    """P13 and the anti-vendoring screen, over the shipped module's own source.

    CERTIFYING COMPONENT: `vllm_neuron/functional/attention/mla_sparse.py` as shipped.

    TWO COUNTED ZEROS, EACH WITH ITS POPULATION AND EACH WITH A CONTROL THAT FIRES.
    (a) No torch attention route outside the named oracle -- a fallback for
    kernel-class work is the P13 defect. (b) No symbol or module path of the refused
    substrate member, so nobody reaches for it later. Both zeros are printed after the
    population they were counted over, and both screens are then run against a string
    that DOES contain a hit, because a zero from a broken screen looks exactly like a
    zero from clean source.
    """
    source = module_source()
    tree = ast.parse(source)
    total_lines = len(source.splitlines())
    say(f"S1_POPULATION_MODULE_LINES={total_lines}")
    say(f"S1_POPULATION_AST_NODES={sum(1 for _ in ast.walk(tree))}")
    assert total_lines > 0

    # (a) torch attention attributes reached OUTSIDE the oracle function.
    oracle = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == ORACLE_NAME), None
    )
    assert oracle is not None, (
        f"the module no longer defines {ORACLE_NAME}, so this screen's exclusion no "
        f"longer names anything real"
    )
    oracle_nodes = {id(n) for n in ast.walk(oracle)}
    hits = []
    for node in ast.walk(tree):
        if id(node) in oracle_nodes:
            continue
        if isinstance(node, ast.Attribute) and node.attr in TORCH_ATTENTION_ATTRS:
            hits.append((getattr(node, "lineno", -1), node.attr))
    say(f"S1_TORCH_ATTENTION_ATTRS_SCREENED={sorted(TORCH_ATTENTION_ATTRS)}")
    say(f"S1_TORCH_ATTENTION_ROUTE_HITS_OUTSIDE_THE_ORACLE={len(hits)}")
    for line, attr in hits:
        say(f"    S1_HIT line={line} attr={attr}")
    assert not hits, (
        f"the module reaches a torch attention entry point outside {ORACLE_NAME} at "
        f"{hits}. A torch fallback for kernel-class work is a P13 defect"
    )
    # The control: the same screen over a function that DOES have one.
    control_tree = ast.parse("def f(x):\n    return torch.softmax(x, dim=-1)\n")
    control_hits = [
        n.attr for n in ast.walk(control_tree)
        if isinstance(n, ast.Attribute) and n.attr in TORCH_ATTENTION_ATTRS
    ]
    say(f"S1_CONTROL_HITS={len(control_hits)} {control_hits}")
    assert len(control_hits) == 1, "the torch-attention screen does not detect a hit"

    # (b) the refused member's symbol and vendor module path.
    vendor_hits = {term: source.count(term) for term in VENDOR_SOURCE_TERMS}
    say(f"S1_VENDOR_TERMS_SCREENED={list(VENDOR_SOURCE_TERMS)}")
    say(f"S1_VENDOR_SYMBOL_HITS={vendor_hits} TOTAL={sum(vendor_hits.values())}")
    assert sum(vendor_hits.values()) == 0, (
        f"the shipped module names the refused substrate member: {vendor_hits}"
    )
    control_text = f"# see {VENDOR_SOURCE_TERMS[0]} for the original"
    say(f"S1_VENDOR_CONTROL_HITS={control_text.count(VENDOR_SOURCE_TERMS[0])}")
    assert control_text.count(VENDOR_SOURCE_TERMS[0]) == 1, (
        "the vendor-symbol screen does not detect a hit"
    )

    # P4, on this module's own source: no NxDI import anywhere.
    #
    # THE TOKEN IS ASSEMBLED AND NEVER WRITTEN WHOLE. The campaign's diff-scoped P4
    # scanner reads added lines, so a screen that spelled the token out would report
    # its own screen as a hit -- it did, on this file's first run
    # (`p4diff-040-branch.out`, attempt 1, TIER2 = 1 on this very line). Splitting it
    # is the same discipline the landed host harnesses use for the compile-cache
    # screen, which bracket-breaks every alternative so it cannot match itself.
    nxdi_token = "neuronx_" + "distributed"
    nxdi = source.count(nxdi_token)
    say(f"S1_NXDI_IMPORTS_IN_THIS_MODULE={nxdi}")
    assert nxdi == 0, "P4: ported code must not import the NxDI stack"
    # And the screen's own control, so the zero above is not a broken search.
    say(f"S1_NXDI_CONTROL_HITS={('import ' + nxdi_token).count(nxdi_token)}")
    assert ("import " + nxdi_token).count(nxdi_token) == 1, (
        "the NxDI screen does not detect a hit"
    )


def test_the_kernel_entry_points_are_authored_here_and_not_imported() -> None:
    """The kernels under test are this module's, unwrapped before being read.

    CERTIFYING COMPONENT: `MS.mla_sparse_kernel_identity`, and through it the four
    `@nki.jit` entry points.

    THE UNWRAP IS THE WHOLE READING. `nki.jit` returns a wrapper whose own
    `__module__` is the decorator's, so reading the attribute off the decorated object
    reports the same answer for an authored kernel and an imported one alike. This
    item therefore also states what the un-unwrapped read WOULD have said, so the
    reader can see that the two differ and that the unwrap is doing work.
    """
    identities = MS.mla_sparse_kernel_identity()
    say(f"K1_ENTRY_POINTS={len(identities)}")
    for module_name, qualname in identities:
        say(f"K1_IDENTITY module={module_name} qualname={qualname}")
        assert module_name == MS.__name__, (
            f"{qualname} reports module {module_name}, not {MS.__name__}: the kernel "
            f"under test is not authored in this module"
        )
    # THE COUNT MOVED, AND `inc-glm53f-093` IS THE WRITER THAT MOVED IT. This is the one
    # second-writer touch of `-040`'s items that block makes, and its plan block's
    # Surface bullet authorises it by name: "the enumerator/K1 count if it authors entry
    # points (disclosed)". `-093` authors the row-tiled pair, so the enumerator returns
    # six and this number follows it. Nothing else in this item changes.
    assert len(identities) == 6, (
        f"this module authors six entry points -- the NoPE one, the RoPE one, "
        f"inc-glm53f-041's tiled pair and inc-glm53f-093's row-tiled pair; the identity "
        f"reading returned {len(identities)}"
    )
    naive = MS.mla_sparse_attention_nope_kernel.__module__
    say(f"K1_WITHOUT_THE_UNWRAP_IT_WOULD_READ={naive}")
    say(f"K1_THE_UNWRAP_CHANGES_THE_ANSWER={int(naive != MS.__name__)}")


# --------------------------------------------------------------------------- #
# `inc-glm53f-041` -- the eight `width` items. THE SELECTION IS PARTITIONED BY NAME:
# every item below carries `width` and none carries `geometry`, so `-k width` runs
# this increment's acceptance and `-k geometry` still runs `-040`'s four unchanged.
# The constants live here rather than beside `-040`'s so this increment reads as one
# block, the same reason its kernel code is contiguous in the module.
# --------------------------------------------------------------------------- #

#: THIS INCREMENT'S DECLARED CASE. The width 2,051 is the plan block's own number,
#: quoted and not chosen here: it violates `% 128` (2,051 = 16 x 128 + 3) and `% 16`
#: alike, so it exercises a ragged tail on BOTH axes the kernel now tiles. Every other
#: field is `-040`'s declared case unchanged, so the only moving part is the width.
WIDTH_CASE = dict(seq=1, heads=64, latent=2051, topk=128, s_kv=256, rope=0)

#: The exact-fit width the bit-identity claim compares against: 17 x 128 = 2,176, the
#: next multiple of the partition tile above 2,051. It is NOT a width `-040` can serve
#: -- its landed seam refuses it for exceeding one MM2 moving tile -- so this is an
#: internal-consistency claim between two widths on THIS increment's path, and the
#: item's own docstring says so, in case a reader takes it for a comparison with `-040`.
REFERENCE_WIDTH = 2176

#: The exact-fit control geometry, declared by the design lap out of `-040`'s review:
#: `seq=2` and `topk=512` walk the per-query loop and the four-chunk MM2 loop that
#: `-040`'s own 1/1 case never entered, at a latent that takes the UNTILED body.
EXACT_FIT_CONTROL_CASE = dict(seq=2, heads=64, latent=512, topk=512, s_kv=1024, rope=0)

#: A minimal tiled geometry, used where an item needs a tiled dispatch cheaply. Its
#: width is `LATENT_TILE + 1`, which is the EXACT value `-040`'s G4 table used to
#: assert was refused -- so its serving is the evidence for that row's deletion.
MINIMAL_TILED_CASE = dict(seq=1, heads=8, latent=129, topk=128, s_kv=256, rope=0)

#: The floor W8's own non-vacuity guard must clear: how much the 3-wide tail tile has to
#: move the FLOAT64 REFERENCE before the item is entitled to claim it tests the tail.
#: Set three orders above the plan's `ATOL` so a tail whose signal was quietly weakened
#: fails the guard long before it could pass the numeric compare vacuously. Not a
#: comparator: it bounds this item's own inputs, not the kernel's agreement.
TAIL_SIGNAL_FLOOR = 1e-2


def case_scale(case: dict) -> float:
    """The softmax scale for a case: its own latent rank's inverse square root.

    Derived per case for `declared_scale`'s reason -- so the scale and the width
    cannot drift apart. Not a registered comparator value; this increment authors no
    tolerance and quotes `RTOL`/`ATOL` from the plan like `-040` does.
    """
    return float(case["latent"]) ** -0.5


def both_counters() -> tuple[int, int, int, int]:
    """`(seam_nki, seam_fallback, tiled_nki, tiled_fallback)` after a reset of both.

    Read as one tuple because the two counters COMPOSE rather than partition: the
    seam counter counts every dispatch whichever body ran, and the tiled counter
    additionally counts the tiled ones. `-042` reads both, so both are stated together.
    """
    seam_nki, seam_fallback = MS.mla_sparse_dispatch_counters()
    tiled_nki, tiled_fallback = MS.mla_sparse_tiled_dispatch_counters()
    return seam_nki, seam_fallback, tiled_nki, tiled_fallback


def reset_both_counters() -> None:
    """Zero both counter objects. Each increment owns its own reset; a case calls both."""
    MS.reset_mla_sparse_dispatch_counters()
    MS.reset_mla_sparse_tiled_dispatch_counters()


def test_width_the_ragged_latent_matches_the_torch_oracle() -> None:
    """WIDTH 1 of 8 -- the tiled kernel computes sparse latent attention at 2,051.

    CERTIFYING COMPONENT: `MS._attention_body_tiled`, reached through the seam's
    width branch.

    THE ROUTE READINGS ARE PRINTED BEFORE THE NUMERIC COMPARE, because `-040`'s round
    one learned that the other order loses them on a red run: a failing assertion took
    the dispatch lines with it and the mutation rows could not tell "the arithmetic
    broke" from "the kernel was never reached".
    """
    say("W1_CERTIFYING_COMPONENT=MS._attention_body_tiled via MS.mla_sparse_attention")
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    say("W1_CASE=" + " ".join(f"{k}={v}" for k, v in case.items()) + f" scale={scale:.6e}")

    q_lift, c_kv, idx, q_pe, k_pe = make_case(**case, seed=41)
    assert q_pe is None and k_pe is None, "the declared case carries no RoPE half"

    can_run = MS.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    )
    say(f"W1_CAN_RUN_KERNEL={can_run}")
    assert can_run, "the tiled geometry must be admissible after this increment"

    reset_both_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    seam_nki, seam_fallback, tiled_nki, tiled_fallback = both_counters()
    say(f"W1_SEAM_NKI_DISPATCH={seam_nki}/1 W1_TILED_NKI_DISPATCH={tiled_nki}/1")
    say(f"W1_SEAM_TORCH_FALLBACK={seam_fallback} W1_TILED_TORCH_FALLBACK={tiled_fallback}")
    say(f"W1_OUTPUT_SHAPE={tuple(got.shape)}")
    assert seam_nki == 1 and tiled_nki == 1
    assert seam_fallback == 0 and tiled_fallback == 0
    assert tuple(got.shape) == (case["seq"], case["heads"], case["latent"])

    want = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    err = float((got - want).abs().max())
    say(f"W1_MAXABS_VS_INDEPENDENT_ORACLE={err:.3e}  RTOL={RTOL} ATOL={ATOL}")
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    say("W1_CASES_AGREED=1/1")

    # THE CONTROL, and it is `-040`'s -- the one already measured able to fire. One
    # selected row out of `topk` is replaced by a cache row the selection does not
    # hold, so the SET changes rather than its order, and the comparison must fail.
    substituted = idx.clone()
    for s in range(case["seq"]):
        selected = {int(v) for v in idx[s].tolist()}
        unused = [r for r in range(case["s_kv"]) if r not in selected]
        assert unused, "the selection covers the whole cache; no substitution exists"
        substituted[s, 0] = unused[0]
    wrong = sparse_mla_torch_reference(q_lift, c_kv, substituted, scale)
    control_err = float((got - wrong).abs().max())
    say(f"W1_CONTROL_MAXABS_AGAINST_A_DIFFERENT_SELECTION={control_err:.3e}")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, wrong, rtol=RTOL, atol=ATOL)
    say("W1_CONTROL_FIRES=1")
    assert control_err > ATOL


def test_width_the_zero_extended_reference_is_bit_identical() -> None:
    """WIDTH 2 of 8 -- 2,051 columns agree EXACTLY with the same data padded to 2,176.

    CERTIFYING COMPONENT: `MS._attention_body_tiled`'s ragged tail tiles, on both the
    partition axis and MM2's moving axis.

    WHAT THE TWO SIDES ARE, because the plan block's wording can be read as a
    comparison against `-040` and it cannot be one. BOTH sides run on THIS
    increment's tiled path: `-040`'s landed seam refuses 2,051 (not a multiple of 128)
    and refuses 2,176 as well (wider than one MM2 moving tile), so no width exists
    that both bodies serve. The unpadded side is the real 2,051. The padded side is
    the SAME inputs zero-extended BY THE CALLER to 17 x 128, sliced back to the real
    columns. So this is an internal-consistency claim on one path, and it is a real one:
    it is what would break if a ragged tail tile read or wrote outside its extent.

    WHY EXACT AND NOT `assert_close`. The extension contributes only exact zeros to
    MM1's contraction, and `x + 0.0 == x` in fp32 for finite x. That was measured on
    the primitive before this was written -- 125 exact zeros added to a 3-deep
    contraction came back bit-identical (`probe-041-ragged-tiles.out`, reading r5) --
    so `max abs diff == 0.0` is a claim the arithmetic supports rather than a hope.

    THE SCALE IS HELD FIXED ACROSS BOTH RUNS. It is an input, not something the kernel
    derives from the width, and deriving it per width would have changed the scores and
    made the comparison meaningless.
    """
    say("W2_CERTIFYING_COMPONENT=MS._attention_body_tiled ragged tail tiles")
    case = dict(WIDTH_CASE)
    latent = case["latent"]
    scale = case_scale(case)
    say(f"W2_REAL_WIDTH={latent} W2_PADDED_WIDTH={REFERENCE_WIDTH} "
        f"W2_PAD_COLUMNS={REFERENCE_WIDTH - latent} scale={scale:.6e}  (one scale, both runs)")

    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    q_ext = torch.zeros(case["seq"], case["heads"], REFERENCE_WIDTH,
                        dtype=torch.float32)
    q_ext[:, :, :latent] = q_lift
    c_ext = torch.zeros(case["s_kv"], REFERENCE_WIDTH, dtype=torch.float32)
    c_ext[:, :latent] = c_kv
    say(f"W2_THE_EXTENSION_IS_EXACTLY_ZERO="
        f"{int(bool((q_ext[:, :, latent:] == 0).all() and (c_ext[:, latent:] == 0).all()))}")

    reset_both_counters()
    got_real = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    got_pad = MS.mla_sparse_attention(q_ext, c_ext, idx, scale)
    seam_nki, _, tiled_nki, _ = both_counters()
    say(f"W2_SEAM_NKI_DISPATCH={seam_nki}/2 W2_TILED_NKI_DISPATCH={tiled_nki}/2  "
        f"(both widths take the tiled body)")
    assert seam_nki == 2 and tiled_nki == 2

    compared = got_real.numel()
    say(f"W2_POPULATION_ELEMENTS_COMPARED={compared}")
    diff = float((got_real - got_pad[:, :, :latent]).abs().max())
    say(f"W2_MAXABS_REAL_VS_PADDED_OVER_THE_REAL_COLUMNS={diff:.3e}")
    say(f"W2_BIT_IDENTICAL={int(diff == 0.0)}")
    assert compared == case["seq"] * case["heads"] * latent
    assert diff == 0.0, (
        f"the padded run disagrees with the ragged run by {diff} over the real "
        f"{latent} columns; a tail tile is reading or writing outside its extent"
    )

    tail = got_pad[:, :, latent:]
    say(f"W2_PADDED_TAIL_OUTPUT_MAXABS={float(tail.abs().max()):.3e}  "
        f"(weights times zero rows, so exactly zero)")
    assert float(tail.abs().max()) == 0.0

    # THREE MORE READINGS, IN INCREASING STRENGTH, and every one of them was measured
    # on the host before this item was written to assert it (`probe-041-tiled-
    # shakedown.out`, reading s4). The order matters, because the FIRST CONTROL I WROTE
    # COULD NOT FIRE and the probe is what caught it.
    #
    # (a) A nonzero CACHE extension alone cannot move the real columns: MM1 contracts
    #     it against the exactly-zero half of q, so those products are exact zeros.
    #     Measured at exactly 0.000e+00, which is a second bit-identity reading by a
    #     different mechanism from the one above.
    c_nz = c_ext.clone()
    c_nz[:, latent:] = 0.7
    got_cache_only = MS.mla_sparse_attention(q_ext, c_nz, idx, scale)
    cache_only = float((got_real - got_cache_only[:, :, :latent]).abs().max())
    say(f"W2_NONZERO_CACHE_EXTENSION_ONLY_MAXABS={cache_only:.3e}  (expected to AGREE)")

    # (b) A CONSTANT nonzero extension on BOTH operands is MATHEMATICALLY INERT, and
    #     this is the defect the probe found in my first attempt at a control: it adds
    #     the same 0.3 x 0.7 x 125 to every score in a row, and softmax is invariant to
    #     a constant shift of all its logits. So the weights cannot move. It is kept as
    #     a READING rather than deleted, because it is the strongest available proof
    #     that the tail reaches the arithmetic AT ALL: it comes back near 6e-9 and NOT
    #     bit-identical, where the zero extension came back exactly 0.0. A comparison
    #     blind to the extension would have read 0.0 for both.
    q_nz = q_ext.clone()
    q_nz[:, :, latent:] = 0.3
    got_const = MS.mla_sparse_attention(q_nz, c_nz, idx, scale)
    constant = float((got_real - got_const[:, :, :latent]).abs().max())
    say(f"W2_CONSTANT_EXTENSION_BOTH_OPERANDS_MAXABS={constant:.3e}  "
        f"(inert by softmax shift-invariance, so expected to AGREE)")
    say(f"W2_BUT_IT_IS_NOT_BIT_IDENTICAL={int(constant != 0.0)}  "
        f"(so the tail does reach the arithmetic)")
    assert constant <= ATOL, (
        f"a constant extension moved the answer by {constant}, which softmax's "
        f"shift-invariance says it cannot; the kernel is not computing a softmax"
    )

    # (c) THE CONTROL THAT FIRES. The extension is made ROW-VARYING, so each selected
    #     cache row gains a DIFFERENT amount and the weights genuinely move. This is
    #     the perturbation the property under test is not invariant to, and finding
    #     which one that is took a measurement rather than an argument.
    c_rv = c_ext.clone()
    c_rv[:, latent:] = (
        torch.arange(case["s_kv"], dtype=torch.float32).unsqueeze(1) * 0.01
    )
    got_rowvarying = MS.mla_sparse_attention(q_nz, c_rv, idx, scale)
    control = float((got_real - got_rowvarying[:, :, :latent]).abs().max())
    say(f"W2_CONTROL_MAXABS_WITH_A_ROW_VARYING_EXTENSION={control:.3e}")
    say(f"W2_CONTROL_FIRES={int(control > ATOL)}")
    assert control > ATOL, (
        f"a row-varying extension moved the answer by only {control}, at or below the "
        f"{ATOL} tolerance, so this item cannot see the tail it asserts is inert"
    )


def test_width_the_tiled_seam_counts_its_own_dispatch() -> None:
    """WIDTH 3 of 8 -- the route predicate, D13 form R-1, on the tiled path.

    CERTIFYING COMPONENT: the width branch in `MS.mla_sparse_attention` and
    `MS._MLA_SPARSE_TILED_COUNTERS`.

    THE TWO COUNTERS COMPOSE AND THE ITEM MEASURES BOTH. `-040`'s seam counter counts
    every dispatch through the seam whichever body runs; this increment's counter
    additionally counts the tiled ones. So a tiled call reads 1 and 1. That is the
    reading `-042`'s decode predicate cites, and this block is its authority: it is
    stated here, per call, rather than inferred there.

    A pure-torch implementation reads 0 on both and cannot pass.
    """
    say("W3_CERTIFYING_COMPONENT=MS.mla_sparse_attention width branch + "
        "MS._MLA_SPARSE_TILED_COUNTERS")
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    reset_both_counters()
    say("W3_AFTER_RESET=" + str(both_counters()))
    assert both_counters() == (0, 0, 0, 0)

    MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    seam_nki, seam_fallback, tiled_nki, tiled_fallback = both_counters()
    say(f"W3_SEAM_NKI_DISPATCH={seam_nki}/1")
    say(f"W3_TILED_NKI_DISPATCH={tiled_nki}/1")
    say(f"W3_SEAM_TORCH_FALLBACK={seam_fallback} W3_TILED_TORCH_FALLBACK={tiled_fallback}")
    say(f"W3_COUNTER_OBJECTS_ARE_DISTINCT="
        f"{int(MS._MLA_SPARSE_COUNTERS is not MS._MLA_SPARSE_TILED_COUNTERS)}")
    assert (seam_nki, tiled_nki) == (1, 1)
    assert (seam_fallback, tiled_fallback) == (0, 0)
    assert MS._MLA_SPARSE_COUNTERS is not MS._MLA_SPARSE_TILED_COUNTERS

    # Each increment's reset owns its own object, which is the other half of "neither
    # reads the other's": resetting the tiled counter must not clear the seam counter.
    MS.reset_mla_sparse_tiled_dispatch_counters()
    after = both_counters()
    say(f"W3_THE_TILED_RESET_LEAVES_THE_SEAM_COUNTER_ALONE={after}")
    assert after == (1, 0, 0, 0)


def test_width_the_tile_arithmetic_comes_from_the_image() -> None:
    """WIDTH 4 of 8 -- the two tilings are derived from `nl.tile_size`, not typed.

    CERTIFYING COMPONENT: `MS._latent_tiles` and `MS._output_tiles`.

    WHY BOTH TILINGS EXIST. The latent rides the PARTITION axis in MM1 and MM2's
    MOVING free axis, and on this image those two extents differ (128 against 512), so
    one width needs two different tilings. The expected tile counts are computed from
    the image's own numbers here rather than written down, so an image change reddens
    this item instead of silently mis-tiling.

    THE PROPERTY, and it is stronger than the counts: the tiles must PARTITION the
    width -- contiguous, non-overlapping, summing to it exactly, each within its axis
    bound. A tiling that dropped or double-counted a column would satisfy a count and
    fail this.
    """
    say("W4_CERTIFYING_COMPONENT=MS._latent_tiles and MS._output_tiles")
    latent = WIDTH_CASE["latent"]
    pmax = nl.tile_size.pmax
    moving = nl.tile_size.gemm_moving_fmax
    say(f"W4_IMAGE_pmax={pmax} IMAGE_gemm_moving_fmax={moving} WIDTH={latent}")

    checks = 0
    for label, tiles, bound in (
        ("PARTITION", MS._latent_tiles(latent), pmax),
        ("MOVING", MS._output_tiles(latent), moving),
    ):
        expected_count = -(-latent // bound)
        expected_tail = latent - (expected_count - 1) * bound
        say(f"W4_{label}_TILES={len(tiles)} EXPECTED={expected_count} "
            f"TAIL={tiles[-1][1]} EXPECTED_TAIL={expected_tail}")
        assert len(tiles) == expected_count
        assert tiles[-1][1] == expected_tail
        assert sum(extent for _, extent in tiles) == latent, (
            f"the {label} tiling does not sum to the width"
        )
        offset = 0
        for tile_offset, extent in tiles:
            assert tile_offset == offset, f"the {label} tiling is not contiguous"
            assert 1 <= extent <= bound, f"a {label} tile exceeds its axis bound"
            offset += extent
        assert offset == latent
        say(f"W4_{label}_TILES_PARTITION_THE_WIDTH=1")
        checks += 1
    say(f"W4_TILINGS_CHECKED={checks}/2")
    assert checks == 2

    # The tail is what the increment is for, so it is stated rather than left implied.
    say(f"W4_THE_TAIL_IS_RAGGED_ON_BOTH_AXES="
        f"{int(MS._latent_tiles(latent)[-1][1] % pmax != 0 and MS._output_tiles(latent)[-1][1] % moving != 0)}")


def test_width_an_exact_fit_call_leaves_the_tiled_counter_at_zero() -> None:
    """WIDTH 5 of 8 -- the exact-fit control: the tiled counter stays 0, the seam's reads 1.

    CERTIFYING COMPONENT: the width branch's FALSE arm -- an exact-fit latent must keep
    `-040`'s untiled body.

    THIS IS THE COUNTED ZERO THAT MAKES "ITS OWN COUNTED VALUE" A MEASUREMENT. Item 3
    shows both counters read 1 on a tiled call; this one shows the tiled counter reads
    0 when the width fits, and then fires it to 1 on a minimal tiled call in the same
    process, so the zero is a discrimination and not a counter that never moves.

    THE GEOMETRY IS THE ONE `-040`'S REVIEW ASKED FOR: `seq=2` walks the per-query
    loop and `topk=512` walks MM2's four-chunk loop, neither of which `-040`'s single
    1/1 case entered. So this item also carries `-040`'s body into a corner its own
    acceptance left unmeasured, at the plan's registered tolerance pair.
    """
    say("W5_CERTIFYING_COMPONENT=MS.mla_sparse_attention width branch, FALSE arm")
    case = dict(EXACT_FIT_CONTROL_CASE)
    scale = case_scale(case)
    say("W5_CASE=" + " ".join(f"{k}={v}" for k, v in case.items()))
    say(f"W5_THIS_WIDTH_FITS_ONE_TILE_SET={int(case['latent'] % MS.LATENT_TILE == 0 and case['latent'] <= MS.MOVING_MAX)}")
    say(f"W5_MM2_CHUNKS={case['topk'] // MS.KEY_CHUNK} W5_QUERIES={case['seq']}")

    q_lift, c_kv, idx, _, _ = make_case(**case, seed=42)
    reset_both_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    seam_nki, seam_fallback, tiled_nki, tiled_fallback = both_counters()
    say(f"W5_SEAM_NKI_DISPATCH={seam_nki}/1")
    say(f"W5_TILED_NKI_DISPATCH={tiled_nki}  (the counted zero)")
    say(f"W5_SEAM_TORCH_FALLBACK={seam_fallback} W5_TILED_TORCH_FALLBACK={tiled_fallback}")
    assert seam_nki == 1 and tiled_nki == 0
    assert seam_fallback == 0 and tiled_fallback == 0

    want = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    err = float((got - want).abs().max())
    say(f"W5_MAXABS_VS_INDEPENDENT_ORACLE={err:.3e}  RTOL={RTOL} ATOL={ATOL}")
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    say(f"W5_CASES_AGREED=1/1 over queries={case['seq']}")

    # THE CONTROL FOR THE ZERO: a minimal tiled call in the same process must move the
    # tiled counter off 0. Its width is 129, which is exactly the value `-040`'s G4
    # table used to assert was REFUSED.
    tiled_case = dict(MINIMAL_TILED_CASE)
    tq, tc, tidx, _, _ = make_case(**tiled_case, seed=43)
    MS.mla_sparse_attention(tq, tc, tidx, case_scale(tiled_case))
    seam_after, _, tiled_after, _ = both_counters()
    say(f"W5_CONTROL_A_TILED_CALL_MOVES_IT tiled={tiled_after} seam={seam_after}")
    say(f"W5_CONTROL_FIRES={int(tiled_after == 1)}")
    assert (seam_after, tiled_after) == (2, 1)


def test_width_the_shipped_oracle_agrees_with_the_independent_reference() -> None:
    """WIDTH 6 of 8 -- the module's own torch oracle agrees with this file's reference.

    CERTIFYING COMPONENT: `MS.mla_sparse_attention_torch_oracle`.

    WHY THIS EXISTS, and it is a review finding on `-040` rather than a new idea. The
    module ships that oracle as section 4's clause (a) -- the only torch arithmetic the
    module is allowed -- and its presence is what `-040`'s P13 screen excludes BY NAME.
    But no test called it, so the exclusion named a function nobody had checked. One
    agreement assert fixes that: the shipped oracle and this file's independently
    written float64 reference must agree at the tolerance pair the plan registers.

    IT MEASURES NO KERNEL. Both sides are torch, so this item says nothing about the
    NKI path and claims nothing about it -- it says the excluded region is correct.
    """
    say("W6_CERTIFYING_COMPONENT=MS." + ORACLE_NAME)
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    reset_both_counters()
    shipped = MS.mla_sparse_attention_torch_oracle(q_lift, c_kv, idx, scale)
    independent = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    err = float((shipped - independent).abs().max())
    say(f"W6_MAXABS_SHIPPED_ORACLE_VS_INDEPENDENT_REFERENCE={err:.3e} "
        f"RTOL={RTOL} ATOL={ATOL}")
    torch.testing.assert_close(shipped, independent, rtol=RTOL, atol=ATOL)
    say("W6_ORACLE_AGREED=1/1")

    # And it dispatched nothing: the oracle is not a fallback and nothing routes to it.
    say(f"W6_NO_DISPATCH_HAPPENED={both_counters()}")
    assert both_counters() == (0, 0, 0, 0)


def test_width_the_two_widths_g4_used_to_refuse_are_now_served() -> None:
    """WIDTH 7 of 8 -- the gate this increment removes is removed, and its floor is kept.

    CERTIFYING COMPONENT: the two relaxed latent bounds in `MS._require_admissible`.

    THIS ITEM IS THE EVIDENCE FOR AN EDIT IN `-040`'S OWN ACCEPTANCE. `-040`'s G4 table
    asserted refusals at `LATENT_TILE + 1` and `MOVING_MAX + LATENT_TILE`; this
    increment serves both widths, so those two rows were removed from G4 under the
    lead's ruling and their positive counterparts are asserted HERE instead. The two
    expressions are written exactly as G4 wrote them, so a reader can match them line
    for line against the deleted rows.

    THE CONTROL: `latent=0` must still refuse, with its own message. Without it this
    item would pass just as well against a check that had stopped refusing anything.
    """
    say("W7_CERTIFYING_COMPONENT=MS._require_admissible, the two relaxed latent bounds")
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    served = 0
    for label, latent in (
        ("latent_not_a_multiple_of_the_partition_tile", MS.LATENT_TILE + 1),
        ("latent_past_the_moving_axis", MS.MOVING_MAX + MS.LATENT_TILE),
        ("the_declared_width", case["latent"]),
        ("the_zero_extended_reference_width", REFERENCE_WIDTH),
    ):
        MS._require_admissible(case["seq"], case["heads"], latent, case["rope"],
                               case["topk"], case["s_kv"], scale)
        say(f"W7_SERVED {label} latent={latent} REFUSAL=none")
        served += 1
    say(f"W7_WIDTHS_NOW_SERVED={served}/4")
    assert served == 4

    with pytest.raises(MS.MlaSparseAttentionError) as caught:
        MS._require_admissible(case["seq"], case["heads"], 0, case["rope"],
                               case["topk"], case["s_kv"], scale)
    message = " ".join(str(caught.value).split())
    say(f"W7_CONTROL_LATENT_ZERO_STILL_REFUSES_VERBATIM={message}")
    assert "latent rank must be positive" in message
    say("W7_CONTROL_FIRES=1")


def test_width_the_ragged_tail_tile_carries_signal_the_output_depends_on() -> None:
    """WIDTH 8 of 8 -- at the declared width, the 3-wide tail tile changes the answer.

    CERTIFYING COMPONENT: `MS._attention_body_tiled`'s tail tile on BOTH axes, at the
    declared width 2,051.

    WHY THIS ITEM EXISTS, and it is a measured gap rather than an idea. Round 1's
    mutation row F deleted the ragged tail tile from MM1 -- a real defect in exactly the
    arithmetic this increment adds -- and W1 PASSED: with `-040`'s random inputs the 3
    tail components move the output by 8.996e-06, under the plan's registered `atol` of
    1e-5. Only W2 caught it, and only by its control going inert. An acceptance that
    catches a deleted tail tile solely through an inert control is not certifying this
    arithmetic, so this item makes the tail LOAD-BEARING in the inputs instead. The
    tolerance is untouched -- it is the plan's; what changes is the data.

    HOW THE SIGNAL IS PLACED, and it is in BOTH operands deliberately. The tail's
    contribution to key `k`'s logit is `scale * sum_over_tail(q[h, l] * c[k, l])`. A ramp
    on the cache alone leaves that sum proportional to `sum_over_tail(q[h, l])`, which is
    a random near-zero number per HEAD, so the signal would vanish for whichever heads
    happen to cancel and the failure margin would be data-dependent. Fixing the query's
    tail to 1.0 makes the sum equal to the ramp for every head, so the margin is
    predictable and the same for all 64 heads. The ramp varies per cache ROW because
    softmax is invariant to a constant shift of a row's logits -- the defect this
    campaign has now hit twice.

    WHAT THIS ITEM ALSO CATCHES, disclosed because it widens a mutation row. The ramp
    leaves the softmax denominator at about 3.5 rather than about 1 (the derivation is at
    the ramp below), so this item ALSO fails when the tiled body drops that denominator --
    the round's mutation G. It is not a tail-only item. G's signature therefore blames two
    items rather than one, and F and G are still told apart because they blame DIFFERENT
    SETS: F blames this item and the exactness item, G blames this item and the oracle
    item.

    THE NON-VACUITY GUARD IS INTERNAL TO THE ITEM. Before any kernel runs, the item
    measures how much the tail contributes to the FLOAT64 REFERENCE by zeroing the tail
    in a copy of the inputs, and requires that to clear a named floor. That reading uses
    no kernel at all, so if a future edit weakened the scaling this item would fail on
    its own guard rather than pass vacuously.
    """
    say("W8_CERTIFYING_COMPONENT=MS._attention_body_tiled tail tile at the declared width")
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    latent = case["latent"]
    tail = latent - (latent % MS.LATENT_TILE)          # 2048: where the tail tile starts
    tail_width = latent - tail                        # 3
    say(f"W8_TAIL_TILE_STARTS_AT={tail} W8_TAIL_WIDTH={tail_width}")
    assert tail_width == 3, "the declared width's tail tile is 3 wide"

    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)
    # The signal: a per-row ramp on the cache's tail columns, and a fixed 1.0 on the
    # query's, so every head sees it. STEP is chosen so the TAIL DECIDES THE SELECTION.
    # Two different spans matter here and confusing them gives the wrong answer:
    #   * ACROSS the whole cache the tail spreads the logits by scale*3*step*255 = 42
    #     nats, against a base spread of about 2.5e-3 -- so the tail, not the random
    #     part, orders the keys.
    #   * BETWEEN ADJACENT SELECTED rows the gap is only scale*3*step = 0.166 nats per
    #     row index, and the 128 selected rows are a random half of 256, so adjacent
    #     selected logits differ by about 0.33 nats.
    # The weights are therefore a GEOMETRIC ramp of ratio about e**-0.33 = 0.72, not a
    # one-hot pick: the effective support is 1/(1-0.72) = about 3.5 rows and the softmax
    # denominator is about 3.5, not about 1. So this item still exercises a graded
    # weighted sum, and it is NOT blind to the denominator. Delete the tail tile and the
    # weights collapse to a near-uniform average of all 128 rows, a change of order 1e-1
    # in every real column. A gentler ramp moves only the weights' shape and lands near
    # 5e-3, close enough to the band to make the item's margin a matter of luck.
    step = 2.5
    ramp = torch.arange(case["s_kv"], dtype=torch.float32).unsqueeze(1) * step
    c_kv = c_kv.clone()
    c_kv[:, tail:] = ramp
    q_lift = q_lift.clone()
    q_lift[:, :, tail:] = 1.0
    say(f"W8_RAMP_STEP={step} W8_MAX_TAIL_LOGIT_TERM="
        f"{scale * tail_width * step * (case['s_kv'] - 1):.3f}")

    # THE GUARD, computed from the inputs and the float64 reference only, no kernel.
    want = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    c_no_tail = c_kv.clone()
    c_no_tail[:, tail:] = 0.0
    q_no_tail = q_lift.clone()
    q_no_tail[:, :, tail:] = 0.0
    want_no_tail = sparse_mla_torch_reference(q_no_tail, c_no_tail, idx, scale)
    # MEASURED OVER THE REAL COLUMNS ONLY, and that restriction is the whole guard.
    # Zeroing the cache's tail trivially zeroes the output's tail columns, so a maximum
    # taken over the full width would be satisfied by arithmetic that never changed a
    # single weight. Restricted to columns [0, 2048) the reading can only move if the
    # tail changed the SOFTMAX -- which is exactly what mutation F destroys.
    contribution = float((want[:, :, :tail] - want_no_tail[:, :, :tail]).abs().max())
    trivial = float((want[:, :, tail:] - want_no_tail[:, :, tail:]).abs().max())
    say(f"W8_TAIL_CONTRIBUTION_OVER_THE_REAL_COLUMNS={contribution:.3e}  "
        f"FLOOR={TAIL_SIGNAL_FLOOR}")
    say(f"W8_TAIL_COLUMNS_MOVE_TRIVIALLY_BY={trivial:.3e}  (not the guard; excluded)")
    assert contribution > TAIL_SIGNAL_FLOOR, (
        f"the tail moves the reference's REAL columns by only {contribution}, at or "
        f"below the {TAIL_SIGNAL_FLOOR} floor, so this item could pass without the tail "
        f"tile ever being computed -- the inputs no longer put signal where it claims"
    )

    reset_both_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    seam_nki, seam_fallback, tiled_nki, tiled_fallback = both_counters()
    say(f"W8_SEAM_NKI_DISPATCH={seam_nki}/1 W8_TILED_NKI_DISPATCH={tiled_nki}/1")
    say(f"W8_SEAM_TORCH_FALLBACK={seam_fallback} W8_TILED_TORCH_FALLBACK={tiled_fallback}")
    assert seam_nki == 1 and tiled_nki == 1
    assert seam_fallback == 0 and tiled_fallback == 0

    # THREE READINGS, because one absolute maximum would misreport this item. The ramp
    # makes the output's TAIL columns about 6e2 in magnitude while its real columns stay
    # about 1e-1, and `assert_close` is elementwise -- it allows `atol + rtol*|want|`,
    # which is about 6 on a tail column and about 1e-3 on a real one. A single absolute
    # maximum is dominated by float32 error on the big numbers and would read like a
    # near-miss against `ATOL` when the assert is nowhere near its limit. So the item
    # reports the two absolute errors on their own scales AND the SLACK RATIO, which is
    # the quantity the assert actually tests: below 1.0 passes, at or above 1.0 fails.
    err = float((got - want).abs().max())
    err_real = float((got[:, :, :tail] - want[:, :, :tail]).abs().max())
    slack = float(((got - want).abs() / (ATOL + RTOL * want.abs())).max())
    say(f"W8_MAXABS_VS_INDEPENDENT_ORACLE={err:.3e}  RTOL={RTOL} ATOL={ATOL}")
    say(f"W8_MAXABS_OVER_THE_REAL_COLUMNS={err_real:.3e}")
    say(f"W8_SLACK_RATIO_AGAINST_THE_REGISTERED_BAND={slack:.3e}  (1.0 is the limit)")
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    say("W8_CASES_AGREED=1/1")
    # The margin a tail-tile defect must clear, both readings taken over the REAL columns
    # so the ratio compares like with like. Quoted by the mutation rows.
    say(f"W8_MARGIN_A_TAIL_DEFECT_MUST_CLEAR={contribution / max(err_real, 1e-12):.3e}x")


# --------------------------------------------------------------------------- #
# `inc-glm53f-093` -- the four `rows` items. THE SELECTION IS PARTITIONED BY NAME, on
# `-041`'s form: every item below carries `rows` and no item above does, so `-k rows`
# runs this increment's acceptance, `-k width` still runs `-041`'s eight and
# `-k geometry` still runs `-040`'s four. At the parent commit `-k rows` collects NOTHING
# -- that is the selector's population control and it is read in the driver, not here.
# --------------------------------------------------------------------------- #

#: THIS INCREMENT'S DECLARED CASE, every field the plan block's own: the production
#: selected-row count 2,048 (`index_topk` in this checkpoint's config) on the
#: checkpoint's own latent rank, head count and RoPE width. `s_kv` is twice the row count
#: so the selection is a real subset.
ROWS_CASE = dict(seq=2, heads=64, latent=512, topk=2048, s_kv=4096, rope=0)

#: The split-invariance case: ONE score tile, where the row-tiled body must reproduce
#: `-040`'s body exactly. `topk == MS.MOVING_MAX` is the widest count `-040` serves.
SPLIT_CASE = dict(seq=2, heads=64, latent=512, topk=512, s_kv=1024, rope=0)

#: A cheap two-tile geometry, for the item that has to move the row-tiled counter off
#: zero after reading the zero.
MINIMAL_ROW_TILED_CASE = dict(seq=1, heads=8, latent=128, topk=1024, s_kv=2048, rope=0)

#: The combination D59-N6 keeps refused: a latent that needs `-041`'s tiling AND a row
#: count that needs this block's. 2,051 is `-041`'s own declared width.
COMBINATION_LATENT = 2051

#: Where the highest-scoring selected row is placed, and it is placed rather than left to
#: chance. See `test_rows_the_production_selected_row_count_matches_the_torch_oracle`.
#: 600 sits inside score tile 1 of 4 (tiles are 512 wide).
PEAK_POSITION = 600


def all_counters() -> tuple[int, int, int, int, int, int]:
    """`(seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb)` for the three seams.

    Read as one tuple because all three COMPOSE rather than partition: the seam counter
    counts every dispatch whichever body ran, and each tiling counter additionally counts
    its own. `-042` reads all three per decode step, so all three are stated together.
    """
    seam_nki, seam_fb = MS.mla_sparse_dispatch_counters()
    tiled_nki, tiled_fb = MS.mla_sparse_tiled_dispatch_counters()
    row_nki, row_fb = MS.mla_sparse_row_tiled_dispatch_counters()
    return seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb


def reset_all_counters() -> None:
    """Zero all three counter objects. Each increment owns its own reset."""
    MS.reset_mla_sparse_dispatch_counters()
    MS.reset_mla_sparse_tiled_dispatch_counters()
    MS.reset_mla_sparse_row_tiled_dispatch_counters()


def order_rows_so_the_peak_lands_in_tile_one(q_lift, c_kv, idx):
    """Reorder each query's selected rows so its highest-scoring row sits at
    :data:`PEAK_POSITION`, inside score tile 1 of 4.

    WHY THIS EXISTS, and it is a measured requirement rather than tidiness. The kernel
    exponentiates each score tile against that TILE's row max and then rescales the
    running denominator and accumulator when a later tile raises the max. Which of the
    two rescales does any work depends on WHERE the global max sits:

      * global max in the LAST tile  -> the tile rescale is exactly 1.0 every time, and a
        sign defect in it is INVISIBLE. Measured: on monotone data a flipped tile rescale
        moved the answer by 0.000e+00 (`probe-093-merge-algebra.out` section 5, and
        revision 1 of that probe failed its own control on exactly this).
      * global max in the FIRST tile -> the accumulator rescale is the blind one.

    So the peak is PLACED in a middle tile, which makes both rescales do work. Reordering
    the selection is legitimate and is the disclosure this docstring exists for: a
    softmax-weighted sum over a SET does not depend on the order the set is listed in --
    `-040` measured that when its first control could not fire (`investigation-040.md`,
    FOUND 1) -- so this changes WHICH arm of the kernel runs and not the expected value.
    """
    ordered = idx.clone()
    for s in range(int(idx.shape[0])):
        gathered = c_kv[idx[s].to(torch.int64)].to(torch.float64)      # [K, L]
        row_score = (q_lift[s].to(torch.float64) @ gathered.t()).max(dim=0).values
        ascending = torch.argsort(row_score)
        shift = int(ascending.numel()) - 1 - PEAK_POSITION
        rotated = torch.cat([ascending[shift:], ascending[:shift]])
        ordered[s] = idx[s][rotated]
    return ordered


def score_tile_diagnostics(q_lift, c_kv, idx, scale):
    """Per query: the scaled score max of each score tile, and WHICH tile holds each
    HEAD's max. NO KERNEL IS INVOLVED -- float64 torch over the inputs alone.

    THE PER-HEAD READING IS THE ONE THAT MATTERS, and getting that right took a second
    pass. The kernel's running max, and therefore both rescale factors, are PER HEAD ROW:
    head `h`'s accumulator is rescaled only when a later tile raises `h`'s own max. So a
    table of maxima taken over all heads says less than it looks like it does. What makes
    this item sensitive to a rescale defect is:

      * at least one head whose max is NOT in tile 0, so the ACCUMULATOR rescale is
        strictly below 1 for that head; and
      * at least one head whose max is NOT in the last tile, so the TILE rescale is
        strictly below 1 for that head.

    Both are asserted from this reading. The deliberate placement of the highest-scoring
    row in tile 1 guarantees at least the head that owns it satisfies both at once.
    """
    tiles = MS._score_tiles(int(idx.shape[1]))
    per_query_max = []
    per_query_argmax = []
    for s in range(int(idx.shape[0])):
        gathered = c_kv[idx[s].to(torch.int64)].to(torch.float64)
        scores = (q_lift[s].to(torch.float64) @ gathered.t()) * scale       # [H, K]
        # [H, T]: each head's max within each score tile.
        per_tile = torch.stack(
            [scores[:, lo:lo + extent].max(dim=1).values for lo, extent in tiles], dim=1
        )
        per_query_max.append([float(v) for v in per_tile.max(dim=0).values])
        per_query_argmax.append([int(t) for t in per_tile.argmax(dim=1)])
    return per_query_max, per_query_argmax


def test_rows_the_production_selected_row_count_matches_the_torch_oracle() -> None:
    """ROWS 1 of 4 -- the kernel serves 2,048 selected rows and matches float64 torch.

    CERTIFYING COMPONENT: `MS._attention_body_row_tiled`, reached through the seam's
    selected-row branch.

    THIS IS THE READING THE BLOCK EXISTS FOR. `-043` landed the selector at this
    checkpoint's `index_topk` of 2,048 and `-040`'s gate refused every count past 512, so
    until this item ran, no test had read the kernel at the width the decode path passes.

    THE ROUTE READINGS ARE PRINTED BEFORE THE NUMERIC COMPARE, `-040`'s and `-041`'s
    lesson: a failing assertion takes the dispatch lines with it, and then a mutation row
    cannot tell "the arithmetic broke" from "the kernel was never reached".

    THE COMPARISON'S CONTROL IS `-040`'S, the one already measured able to fire: one
    selected row is replaced by a cache row the selection does not hold, so the SET
    changes. A row PERMUTATION is NOT a control here and is not used as one -- the answer
    is order-invariant, which is the property this item's own row ordering relies on.
    """
    say("R1_CERTIFYING_COMPONENT=MS._attention_body_row_tiled via MS.mla_sparse_attention")
    case = dict(ROWS_CASE)
    scale = case_scale(case)
    say("R1_CASE=" + " ".join(f"{k}={v}" for k, v in case.items()) + f" scale={scale:.6e}")

    # THE TILING, read out of the module rather than written down here. The property is
    # stronger than the count: the tiles must PARTITION the selected-row axis, and every
    # extent must be a whole number of MM2 key chunks or the body would need a
    # partial-chunk case it does not have.
    tiles = MS._score_tiles(case["topk"])
    say(f"R1_SCORE_TILES={tiles}")
    say(f"R1_SCORE_TILE_COUNT={len(tiles)} EXPECTED_FROM_THE_IMAGE="
        f"{-(-case['topk'] // MS.MOVING_MAX)}")
    assert len(tiles) == -(-case["topk"] // MS.MOVING_MAX)
    covered = 0
    for lo, extent in tiles:
        assert lo == covered, f"the score tiles are not contiguous at offset {lo}"
        assert extent <= MS.MOVING_MAX, f"tile extent {extent} exceeds the moving axis"
        assert extent % MS.KEY_CHUNK == 0, (
            f"tile extent {extent} is not a whole number of {MS.KEY_CHUNK}-key chunks"
        )
        covered += extent
    say(f"R1_TILES_COVER_THE_AXIS_EXACTLY={int(covered == case['topk'])} "
        f"COVERED={covered} TOPK={case['topk']}")
    assert covered == case["topk"]

    q_lift, c_kv, idx, q_pe, k_pe = make_case(**case, seed=93)
    assert q_pe is None and k_pe is None, "the declared case carries no RoPE half"
    idx = order_rows_so_the_peak_lands_in_tile_one(q_lift, c_kv, idx)

    # THE NON-VACUITY GUARD, and it is internal to the item: unless the global row max
    # sits in a MIDDLE tile, one of the two rescale factors is exactly 1.0 and this
    # comparison cannot see a defect in it. Measured from the inputs, no kernel.
    maxima, argmax_tiles = score_tile_diagnostics(q_lift, c_kv, idx, scale)
    last = len(tiles) - 1
    for s, (row, heads_argmax) in enumerate(zip(maxima, argmax_tiles)):
        histogram = [heads_argmax.count(t) for t in range(len(tiles))]
        say(f"R1_PER_TILE_SCALED_SCORE_MAX_QUERY_{s}="
            + " ".join(f"{m:.6f}" for m in row) + f" ARGMAX_TILE={row.index(max(row))}")
        say(f"R1_HEADS_WHOSE_OWN_MAX_IS_IN_EACH_TILE_QUERY_{s}={histogram} "
            f"POPULATION_HEADS={len(heads_argmax)}")
        assert row.index(max(row)) == 1, (
            f"query {s}'s highest score is in tile {row.index(max(row))}, not tile 1; "
            f"the deliberate row ordering did not take effect"
        )
        # THE TWO CONDITIONS THAT MAKE THIS ITEM SENSITIVE, asserted per head rather than
        # over all heads at once. Each is a population count over the 64 head rows.
        raises = sum(histogram[1:])
        below = sum(histogram[:last])
        say(f"R1_HEADS_WHOSE_MAX_IS_NOT_IN_TILE_0={raises}  (the accumulator rescale is "
            f"below 1 for each of them)")
        say(f"R1_HEADS_WHOSE_MAX_IS_NOT_IN_THE_LAST_TILE={below}  (the tile rescale is "
            f"below 1 for each of them)")
        assert raises > 0, (
            f"query {s}: every head's max is in tile 0, so the accumulator rescale is "
            f"exactly 1.0 everywhere and this item is blind to a defect in it"
        )
        assert below > 0, (
            f"query {s}: every head's max is in the last tile, so the tile rescale is "
            f"exactly 1.0 everywhere and this item is blind to a defect in it -- the "
            f"case measured in probe-093-merge-algebra.out section 5"
        )
    say("R1_BOTH_RESCALE_ARMS_ARE_EXERCISED=1")

    can_run = MS.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    )
    say(f"R1_CAN_RUN_KERNEL={can_run}")
    assert can_run, "the production selected-row count must be admissible after this block"

    reset_all_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    counters = all_counters()
    say(f"R1_COUNTERS_SEAM_TILED_ROW={counters}  EXPECTED=(1, 0, 0, 0, 1, 0)")
    say(f"R1_OUTPUT_SHAPE={tuple(got.shape)}")
    assert counters == (1, 0, 0, 0, 1, 0)
    assert tuple(got.shape) == (case["seq"], case["heads"], case["latent"])

    want = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    err = float((got - want).abs().max())
    slack = float(((got - want).abs() / (ATOL + RTOL * want.abs())).max())
    say(f"R1_MAXABS_VS_INDEPENDENT_ORACLE={err:.3e}  RTOL={RTOL} ATOL={ATOL}")
    say(f"R1_SLACK_RATIO_AGAINST_THE_REGISTERED_BAND={slack:.3e}  (1.0 is the limit)")
    say(f"R1_POPULATION_ELEMENTS_COMPARED={got.numel()}")
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    say(f"R1_CASES_AGREED=1/1 over queries={case['seq']} rows={case['topk']}")

    substituted = idx.clone()
    for s in range(case["seq"]):
        selected = {int(v) for v in idx[s].tolist()}
        unused = [r for r in range(case["s_kv"]) if r not in selected]
        assert unused, "the selection covers the whole cache; no substitution exists"
        substituted[s, 0] = unused[0]
    wrong = sparse_mla_torch_reference(q_lift, c_kv, substituted, scale)
    control_err = float((got - wrong).abs().max())
    say(f"R1_CONTROL_MAXABS_AGAINST_A_DIFFERENT_SELECTION={control_err:.3e}")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, wrong, rtol=RTOL, atol=ATOL)
    say("R1_CONTROL_FIRES=1")
    assert control_err > ATOL


def test_rows_one_score_tile_is_bit_identical_to_the_untiled_body() -> None:
    """ROWS 2 of 4 -- at one score tile the row-tiled body IS `-040`'s body, exactly.

    CERTIFYING COMPONENT: `MS._attention_body_row_tiled`'s single-tile arm, against
    `MS._attention_body`.

    WHAT THE TWO SIDES ARE. Both entry points are called DIRECTLY rather than through the
    seam, because the seam routes `topk=512` to `-040`'s body by design -- so the
    row-tiled path has to be FORCED to be compared at a width `-040` also serves. That
    is the whole reason this comparison is possible at all: unlike `-041`, which had no
    width both bodies serve, the selected-row axis has an overlap at exactly
    `MS.MOVING_MAX`.

    WHY EXACT AND NOT `assert_close`. At one tile the merge emits NOTHING: the rescale
    block is behind a trace-time branch on the tile count, so the traced instruction
    sequence is `-040`'s, with the Q transpose hoisted above a single-iteration loop and
    one fp32 copy of the accumulator added. An fp32 copy does not change a value. The
    same claim was measured on the algebra before the kernel was written
    (`probe-093-merge-algebra.out` reading 1, bit-identical in Python floats).

    A NONZERO READING HERE IS `evidence_contradicts_design`. It would mean the single
    tile arm is not `-040`'s arithmetic, and the answer is to report that, not to loosen
    this line to a tolerance.
    """
    say("R2_CERTIFYING_COMPONENT=MS._attention_body_row_tiled single-tile arm vs "
        "MS._attention_body")
    case = dict(SPLIT_CASE)
    scale = case_scale(case)
    say("R2_CASE=" + " ".join(f"{k}={v}" for k, v in case.items()) + f" scale={scale:.6e}")
    tiles = MS._score_tiles(case["topk"])
    say(f"R2_SCORE_TILES={tiles} R2_TILE_COUNT={len(tiles)}  (one, which is the point)")
    assert len(tiles) == 1 and tiles[0] == (0, case["topk"])

    q_lift, c_kv, idx, _, _ = make_case(**case, seed=94)
    q_f32 = q_lift.contiguous().to(torch.float32)
    c_f32 = c_kv.contiguous().to(torch.float32)
    i_i32 = idx.contiguous().to(torch.int32)

    reset_all_counters()
    untiled = wrap_nki(MS.mla_sparse_attention_nope_kernel)(q_f32, c_f32, i_i32, scale)
    forced = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)(
        q_f32, c_f32, i_i32, scale
    )
    # A DIRECT CALL BYPASSES THE SEAM, so it counts nothing. Stated rather than assumed,
    # because it is also the reason the route predicate is item 4's reading and not this
    # item's: this item measures arithmetic, item 4 measures which body the seam picks.
    say(f"R2_A_DIRECT_ENTRY_POINT_CALL_COUNTS_NOTHING={all_counters()}")
    assert all_counters() == (0, 0, 0, 0, 0, 0)

    compared = untiled.numel()
    diff = float((untiled - forced).abs().max())
    say(f"R2_POPULATION_ELEMENTS_COMPARED={compared}")
    say(f"R2_MAXABS_UNTILED_VS_FORCED_ROW_TILED={diff:.3e}")
    say(f"R2_BIT_IDENTICAL={int(diff == 0.0)}")
    assert compared == case["seq"] * case["heads"] * case["latent"]
    assert diff == 0.0, (
        f"the forced row-tiled body disagrees with -040's body by {diff} at ONE score "
        f"tile, where it emits no merge at all. That is evidence_contradicts_design: "
        f"report it, do not widen this line to a tolerance"
    )

    # THE CONTROL FOR THE ZERO. Without it, a comparison of two calls that both returned
    # zeros -- or the same cached tensor -- would read 0.0 just as happily. One selected
    # row is changed on ONE side, and the same comparison must then see a difference.
    substituted = idx.clone()
    selected = {int(v) for v in idx[0].tolist()}
    unused = [r for r in range(case["s_kv"]) if r not in selected]
    assert unused, "the selection covers the whole cache; no substitution exists"
    substituted[0, 0] = unused[0]
    other = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)(
        q_f32, c_f32, substituted.contiguous().to(torch.int32), scale
    )
    control = float((untiled - other).abs().max())
    say(f"R2_CONTROL_MAXABS_WITH_ONE_SELECTED_ROW_CHANGED={control:.3e}")
    say(f"R2_CONTROL_FIRES={int(control > ATOL)}")
    assert control > ATOL, (
        f"changing a selected row moved the comparison by only {control}, so this item "
        f"cannot see a difference and its 0.0 above says nothing"
    )
    say(f"R2_AND_THE_OUTPUT_IS_NOT_ALL_ZERO={float(untiled.abs().max()):.3e}")
    assert float(untiled.abs().max()) > 0.0


def test_rows_the_gate_serves_the_production_count_and_still_refuses_by_name() -> None:
    """ROWS 3 of 4 -- the two `topk` clauses AFTER the relaxation, read at the candidate.

    CERTIFYING COMPONENT: the two relaxed `topk` clauses in `MS._require_admissible`.

    THIS IS THE `AFTER` HALF OF ONE TWO-SIDED READING. The `BEFORE` half cannot be a
    pytest item, because `-k rows` collects nothing at the parent commit -- so it is
    taken by a driver-level probe that imports the parent checkout's module and prints
    both refusal messages verbatim. That is disclosed as an instrument choice in the
    round's evidence record, and the two transcripts are diffed there. What this item
    owes is the other side of that diff, printed the same way.

    THE CONTROL: the declared admissible case passes the same check, so "2,048 is served"
    is not a check that stopped refusing anything. And `-041`'s own tiled width is still
    served at a narrow row count, so the combination refusal below did not take `-041`'s
    path with it.
    """
    say("R3_CERTIFYING_COMPONENT=MS._require_admissible, the two relaxed topk clauses")
    case = dict(ROWS_CASE)
    scale = case_scale(case)

    def admissibility(**overrides):
        args = dict(seq=case["seq"], heads=case["heads"], latent=case["latent"],
                    rope=case["rope"], topk=case["topk"], s_kv=case["s_kv"],
                    softmax_scale=scale)
        args.update(overrides)
        return args

    # SERVED: the production row count, which the parent refuses.
    MS._require_admissible(**admissibility())
    say(f"R3_SERVED topk={case['topk']} latent={case['latent']} REFUSAL=none")

    # STILL REFUSED, BY NAME: a count that is not a whole number of key chunks.
    with pytest.raises(MS.MlaSparseAttentionError) as caught:
        MS._require_admissible(**admissibility(topk=MS.KEY_CHUNK + 2))
    chunk_message = " ".join(str(caught.value).split())
    say(f"R3_TOPK_130_MESSAGE_VERBATIM={chunk_message}")
    say(f"R3_IT_STILL_NAMES_ITS_OWN_AXIS={int('multiple of it' in chunk_message)}")
    say(f"R3_IT_NO_LONGER_NAMES_041={int('inc-glm53f-041' not in chunk_message)}")
    assert "multiple of it" in chunk_message
    assert "inc-glm53f-041" not in chunk_message, (
        "the chunk refusal still promises inc-glm53f-041's padding increment, which this "
        "block was to remove"
    )

    # STILL REFUSED, BY NAME: both tilings in one call (D59-N6). The message names both
    # axes and promises NO increment -- a refusal that promised one would be the defect
    # this file already carried twice.
    with pytest.raises(MS.MlaSparseAttentionError) as caught:
        MS._require_admissible(**admissibility(latent=COMBINATION_LATENT))
    both_message = " ".join(str(caught.value).split())
    say(f"R3_COMBINATION_MESSAGE_VERBATIM={both_message}")
    for fragment in ("not served", "SAME call", f"topk={case['topk']}",
                     f"latent={COMBINATION_LATENT}"):
        assert fragment in both_message, (
            f"the combination refusal does not name {fragment!r}: {both_message!r}"
        )
    promises = [term for term in ("inc-glm53f-041", "inc-glm53f-093", "increment")
                if term in both_message]
    say(f"R3_THE_COMBINATION_REFUSAL_PROMISES_NOTHING={int(not promises)} "
        f"TERMS_FOUND={promises}")
    assert not promises, (
        f"the combination refusal names {promises}, so it promises an increment nobody "
        f"owns -- D59-N6 requires it be refused, not deferred"
    )

    # THE CONTROLS.
    MS._require_admissible(**admissibility(topk=MS.KEY_CHUNK))
    say(f"R3_CONTROL_THE_NARROW_COUNT_IS_STILL_ADMISSIBLE=1 topk={MS.KEY_CHUNK}")
    MS._require_admissible(**admissibility(latent=COMBINATION_LATENT,
                                           topk=MS.KEY_CHUNK))
    say(f"R3_CONTROL_041S_TILED_LATENT_IS_STILL_SERVED_AT_A_NARROW_COUNT=1 "
        f"latent={COMBINATION_LATENT} topk={MS.KEY_CHUNK}")
    say("R3_CONTROL_FIRES=2  (the check refuses two geometries and serves three)")


def test_rows_the_row_tiled_seam_counts_its_own_dispatch() -> None:
    """ROWS 4 of 4 -- the route predicate, D13 form R-1, on the row-tiled path.

    CERTIFYING COMPONENT: the selected-row branch in `MS.mla_sparse_attention` and
    `MS._MLA_SPARSE_ROW_TILED_COUNTERS`.

    THE THREE COUNTERS COMPOSE AND THIS ITEM MEASURES ALL THREE. `-040`'s seam counter
    counts every dispatch whichever body runs; `-041`'s counts the latent-tiled ones;
    this block's counts the row-tiled ones. At this checkpoint's geometry -- latent 512,
    an EXACT FIT, and 2,048 selected rows -- the reading is 1, 0, 1. That is the tuple
    `-042`'s decode predicate cites per step, and this block is its authority: it is
    stated here, per call, rather than inferred there.

    A pure-torch implementation reads 0 for the row-tiled counter and cannot pass.
    """
    say("R4_CERTIFYING_COMPONENT=MS.mla_sparse_attention selected-row branch + "
        "MS._MLA_SPARSE_ROW_TILED_COUNTERS")
    case = dict(ROWS_CASE)
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=93)
    idx = order_rows_so_the_peak_lands_in_tile_one(q_lift, c_kv, idx)

    reset_all_counters()
    say(f"R4_AFTER_RESET={all_counters()}")
    assert all_counters() == (0, 0, 0, 0, 0, 0)

    MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb = all_counters()
    say(f"R4_SEAM_NKI_DISPATCH={seam_nki}/1")
    say(f"R4_TILED_NKI_DISPATCH={tiled_nki}  (a counted zero: latent 512 is an exact fit)")
    say(f"R4_ROW_TILED_NKI_DISPATCH={row_nki}/1")
    say(f"R4_TORCH_FALLBACKS_SEAM_TILED_ROW={seam_fb} {tiled_fb} {row_fb}")
    distinct = len({id(MS._MLA_SPARSE_COUNTERS),
                    id(MS._MLA_SPARSE_TILED_COUNTERS),
                    id(MS._MLA_SPARSE_ROW_TILED_COUNTERS)})
    say(f"R4_DISTINCT_COUNTER_OBJECTS={distinct}/3")
    assert (seam_nki, tiled_nki, row_nki) == (1, 0, 1)
    assert (seam_fb, tiled_fb, row_fb) == (0, 0, 0)
    assert distinct == 3, "two increments are sharing one counter object"

    # THE COUNTED ZERO AND ITS CONTROL. An exact-fit, narrow call must leave the
    # row-tiled counter alone; a wider call in the SAME process must move it. Without
    # the second half the zero could be a counter that never moves at all.
    narrow = dict(SPLIT_CASE)
    nq, nc, nidx, _, _ = make_case(**narrow, seed=94)
    reset_all_counters()
    MS.mla_sparse_attention(nq, nc, nidx, case_scale(narrow))
    after_narrow = all_counters()
    say(f"R4_UNFORCED_AT_TOPK_{narrow['topk']}={after_narrow}  "
        f"(the row-tiled counter is the counted zero)")
    assert after_narrow == (1, 0, 0, 0, 0, 0)

    wide = dict(MINIMAL_ROW_TILED_CASE)
    wq, wc, widx, _, _ = make_case(**wide, seed=95)
    MS.mla_sparse_attention(wq, wc, widx, case_scale(wide))
    after_wide = all_counters()
    say(f"R4_CONTROL_A_WIDER_CALL_MOVES_IT topk={wide['topk']} -> {after_wide}")
    say(f"R4_CONTROL_FIRES={int(after_wide[4] == 1)}")
    assert after_wide == (2, 0, 0, 0, 1, 0)

    # Each increment's reset owns its own object, which is the other half of "none reads
    # another's": resetting this block's counter must leave the other two standing.
    MS.reset_mla_sparse_row_tiled_dispatch_counters()
    after_reset = all_counters()
    say(f"R4_THE_ROW_TILED_RESET_LEAVES_THE_OTHER_TWO_ALONE={after_reset}")
    assert after_reset == (2, 0, 0, 0, 0, 0)
