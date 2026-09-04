# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-040` -- the sparse MLA latent attention kernel.

SIX tests and NO `parametrize` decorator in this file. Four carry `geometry` in their
name and are the increment's DECLARED acceptance selection; two do not, and screen the
module rather than measure the kernel. Each test prints its counted value and names the
component whose behaviour it certifies.

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
topk width tiles correctly is `inc-glm53f-041`'s, also in this file but as its own
items -- this file's `width` selection does not exist yet and `-041` adds it.
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
        ("latent_not_a_multiple_of_the_partition_tile",
         dict(latent=MS.LATENT_TILE + 1), "multiple of it"),
        ("latent_past_the_moving_axis",
         dict(latent=MS.MOVING_MAX + MS.LATENT_TILE), "moving free axis"),
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

    CERTIFYING COMPONENT: `MS.mla_sparse_kernel_identity`, and through it the two
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
    assert len(identities) == 2, (
        f"this module authors two entry points, the NoPE one and the RoPE one; the "
        f"identity reading returned {len(identities)}"
    )
    naive = MS.mla_sparse_attention_nope_kernel.__module__
    say(f"K1_WITHOUT_THE_UNWRAP_IT_WOULD_READ={naive}")
    say(f"K1_THE_UNWRAP_CHANGES_THE_ANSWER={int(naive != MS.__name__)}")
