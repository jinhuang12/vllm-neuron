"""``inc-glm53f-051`` -- the DSA decoder layer, its indexer chain and its runner integration.

WHAT THIS BLOCK BUILDS. One decoder layer on the ``deepseek_sparse_attention`` half that turns
hidden states into attention output through the DSA indexer: score every candidate against the
indexer query, select the highest-scoring POOLS, expand those pool ids into token indices, gather
the selected rows, and hand the result to ``inc-glm53f-042``'s ``attend`` as its ``topk_indices``.
Nine landed NKI seams do the work; this layer is the orchestration around them.

THE ACCEPTANCE ITEMS, cited BY ANCHOR and never by line number -- the plan block
``#### `inc-glm53f-051``` moved twice while this file was being written, and a folded line number is
stale at the next lap (D-18, review item B72-N6).

  (1)   layer numerics: a 3-layer DSA stack matches a torch reference at
        ``assert_close(rtol=5e-3, atol=1e-5)``, 1/1 tiny case.
  (2)   tiling transparency below the SAI ceiling: tiled against untiled at S = 1,024 and
        S = 1,536, max abs diff <= 1e-5, 4/4 arms.
  (3)   tile-configuration invariance above it: tile 512 against tile 256 at S = 2,048, 2/2 arms.
  (P1)  the route predicate, part 1: 7/7 distinct seam counter families nonzero after the 3-layer
        forward, each at the reading DECLARED in the predictions file before the run; entry-point
        attribution by the test-side call spy below, 8/9 called with per-layer counts printed; the
        1-layer stack reads exactly one third of every family. The predicate reads EIGHT of nine,
        not nine, because ``dsa_ragged_unpack`` has no admissible payload on either case -- ruled
        at design entry ``design-20260905-aj`` (plan revision 218, DECISIONS §52) on this seat's
        finding F11. Its declared zero is covered by ``inc-glm53f-045``'s LANDED bit-identical
        round-trip item, so the direction is tested; it is tested there and not here.
  (P2)  parts 2 and 3 of the predicate: the tiling seam's dispatch count equals each arm's reported
        tile count.
  (P3)  the PROJECTION substrate reading, taken from attempt 13 and absent before it: the EIGHTH
        counter family, ``mla_projection``, is reset and read around EVERY indexer call in this file,
        and each case's total is the closed form printed WITH ITS TERMS. A completing indexer call
        reads ``nki_dispatch == 4`` and ``torch_fallback == 0``, one dispatch per projection site;
        the eight refusal calls read ``(0, 0)``, because the refusal fires before ``project_stage``.
        Review item B86.
  (S)   standing on every run: every family's torch-fallback counter reads 0. The PROJECTION
        family's zero is weaker than the seven and says so where it is asserted -- no code path can
        increment it today, so it is a reading that would only acquire teeth if a torch projection
        route were ever added. Its ``nki_dispatch`` figure is the one with teeth.

WHY THE PER-FAMILY READINGS ARE NOT ALL THE SAME NUMBER. The nine entry points sit behind SEVEN
counter families, because ``kpool_hadamard`` counts its fused and stage-alone entries on one counter
and ``ragged_pack`` counts pack and unpack on one counter, both by their own docstrings. And two
entry points are phase-exclusive: ``decode_tail_update``'s own docstring says prefill is served by
``kpool_hadamard`` while decode sees one token at a time. So a one-phase case would read ZERO on one
family, which is a contradiction rather than a pass, and the case below runs a prefill forward AND a
decode step. The declared table is in ``DECLARED_PER_LAYER`` and its 3-layer totals in
``DECLARED_FAMILY_TOTALS``; the derivation and its controls are in
``artifacts/campaigns/glm-5.3-flash-port/increments/probe-051-prep-derivation.out``.

AND PART 1 IS TWO CASES, BECAUSE ONE FAMILY IS UNREACHABLE AT BATCH ONE. Upstream reaches its pack
pair only under ``if decode_metadata.requires_padding:``, and a batch of one is uniform by
construction, so the layer case's ``ragged_pack`` reading is a DECLARED ZERO. The seventh family is
reached by a second case instead -- a non-uniform decode batch called on ``Glm5NextDSAIndexer``
directly -- and ``7/7`` is read over the UNION of the two. That arm's table is
``DECLARED_PER_LAYER_RAGGED_ARM``, its totals ``DECLARED_FAMILY_TOTALS_RAGGED_ARM``, and each case
carries its own one-third moving control. Ruled at design entry ``design-20260905-aa``.

THE ARM PACKS ONCE AND UNPACKS NEVER, and the reason is a dtype gate rather than a preference.
``dsa_ragged_pack`` admits bf16 alone (``ragged_pack.py:136``, gating both directions at ``ragged_pack.py:609``
and ``ragged_pack.py:624``), and the only tensors the arm could pack are the indexer query and the gate weights.
The weights are fp32 because ``dsa_score_gemm`` admits fp32 weights alone (``score_gemm.py:158``,
whose own comment says a bf16 weight would quietly lose the fold's precision), so packing them is
not available at any price a design may pay. And nothing on the arm needs the padded form back:
``mla_sparse_attention`` requires DENSE 2-D ``[seq, topk]`` indices and raises by name on any other
rank (``mla_sparse.py:1345-1347``), so a padded rank-3 tensor would be REFUSED by the consumer. So
one pack, one counter moved for the family, no unpack. This seat raised the collision as F11 and
chose no route; the lead ruled route (b) and refused both a bf16 round trip taken to move a counter
and a declared torch fallback. The unpack direction's own coverage is ``inc-glm53f-045``'s landed
bit-identical round-trip item, cited above.

WHY A CALL SPY AS WELL AS THE COUNTERS. A shared counter reading 9 cannot say which of its two entry
points contributed what. The counters and the spy count the same events by different means, so the
per-family agreement check below is a real cross-check: a call that bypassed the spy moves a counter
without moving a spy count, and a double-counting wrapper moves a spy count without moving a counter.

WHY THE PROJECTION SEAM NEEDED AN EIGHTH PAIR, and this is a defect of this file rather than a
refinement of it. ``reset_all_counters`` and ``read_all_counters`` walk ``FAMILIES``, ``FAMILIES`` is
derived from ``ENTRY_POINTS``, and every entry point there lives under ``vllm_neuron.functional.dsa``.
The projection seam lives under ``vllm_neuron.functional.attention``. So the walk could not reach it
at any reading, and this file took none for twelve attempts -- an instrument whose POPULATION excluded
the thing its own sentence was about, which is the defect class this increment has produced over and
over. The seam's reader states the purpose of the pair: "The counter is kept so a test can STATE that
reading rather than assume it, which is what makes the zero a measurement"
(``mla_projections.py:148-156``). The eighth family keeps its own pair and its own declared closed
form, and the seven-family tables above are untouched: the projection seam is not a dsa entry point,
so folding it into ``FAMILIES`` would move ``DECLARED_FAMILY_TOTALS``, the one-third control and the
spy-versus-counter agreement, none of which is about projections.

WHERE THE SPY PATCHES, AND WHY IT IS NOT A PREFERENCE. ``model_fp8.py`` imports every functional seam
INSIDE the method body and never at module level, so each name is looked up fresh out of its seam
module at every call. The spy therefore patches the SEAM MODULE attribute. Patching the model's
namespace instead would read 0/9 while the counters moved -- an instrument that fails silently. No
``functional/`` file is edited: the patch is a ``monkeypatch`` attribute set that is reverted at
teardown, so this increment's file surface is unchanged. Measured in
``probe-051-spy-feasibility.out``.

THE PRECONDITION EVERY COUNT RESTS ON. ``can_run_kernel()`` returns ``False`` in CPU mode unless
``NKI_SIMULATOR`` is exactly ``"1"``. Without it every gate refuses, all seven families read 0 AND
the standing torch-fallback-reads-0 check fails -- a false negative on both legs at once. So the gate
is asserted LIVE before any count is read.

WHAT THESE TESTS DO NOT ESTABLISH. Exit status is rung 1. Whether the coverage is ADEQUATE is rung 2
and belongs to review, which is why every counted zero here owns a control that fires. The tiling
residue -- an error invariant to tile size and absent at every S <= 1,536 -- is not carried here; the
plan assigns it to the M4/M7 gate's own HF-reference comparator.
"""

from __future__ import annotations

import functools
import importlib
from collections import Counter

import pytest
import torch

# --------------------------------------------------------------------------- #
# THE DECLARED GEOMETRY. Every value is the checkpoint's or is derived from a gate this file
# names, and the derivation is stated beside it, because a reader needs to know which numbers
# may not be shrunk for speed.

#: ``index_kpool`` -- how many tokens one indexer pool covers. ``fixtures/hf-config.json``.
#: A power of two, which ``can_run_dsa_index_expand`` requires and whose reason its own
#: docstring gives: the kernel derives ``tail_start`` with a mask that is exact only then.
POOL_SIZE = 4

#: ``index_head_dim``. ``can_run_dsa_hadamard128`` refuses any other width -- the transform is
#: a 128-point one -- so this may not be shrunk.
INDEX_HEAD_DIM = 128

#: ``index_n_heads`` and ``index_topk`` at production scale, recorded for the reader. The tiny
#: case runs its own pool-granular ``TOPK_POOLS`` instead; 2,048 tokens is 512 pools here.
INDEX_N_HEADS = 32
INDEX_TOPK = 2048

#: The tiny case: eight complete pools plus a tail of ``POOL_SIZE - 1``, so the FIRST decode step
#: completes a pool. ``35 = 8 * 4 + 3``.
#:
#: WHY 35 AND NOT 19, which is what this constant read until the geometry sweep. Both satisfy the
#: two properties the case needs -- ``35 % 4 == 3`` keeps every tail column a REAL token index
#: (``index_expand.py:9-10``), and ``completes_pool(35, 4)`` is True (``decode_tail_update.py:455``)
#: so the one decode step still completes a pool and moves the tail counter. 35 was chosen over 19
#: because it makes the candidate width the top-k seam sees WIDER, and the narrow end of that width
#: is the one undetermined risk in this fixture:
#:
#:   ``dsa_topk_select``'s gate admits any ``0 < k < width`` (``topk_select.py:294``) and its dry run
#:   cannot see a body-level failure (``topk_select.py:299`` catches only ``AssertionError`` from the config
#:   factories). At ``k = 2`` the body takes the ``k % 8 != 0`` branch, which calls
#:   ``nisa.max8(dst=val_buf[8 wide], src=data[:, :width])``
#:   (``rotational_topk_utils.py:1028-1039``). Whether that instruction tolerates a source free axis
#:   NARROWER than its 8 fixed lanes is not readable from this repository -- nki is not installed on
#:   the authoring host -- and the narrowest width the fork has ever measured for this seam is 4096
#:   (``test/vllm_neuron/functional/dsa/test_topk_select.py:47``). Two indirect readings point at an
#:   8-element granularity without settling a minimum: ``functional/moe/router.py:939-949`` says
#:   max8 "emits exactly 8 values per partition ... fixed by the instructions, not by a choice
#:   here", and ``functional/argsort_unstable.py:185-199`` pads its axis to a multiple of 8 with
#:   sentinels first and asserts ``"SBUF input requires N to be a multiple of 8"``.
#:
#: 19 tokens gave candidate widths 4 (prefill) and 5 (decode); 35 gives 8 and 9. Neither pair can be
#: made two multiples of 8 at once, and that is arithmetic rather than a shortfall of searching: a
#: full tail forces ``T = 4q + 3``, so the prefill width is ``q`` and the decode width is ``q + 1``,
#: two consecutive integers. So 35 is the better of two, not a fix: it puts BOTH widths at or above
#: the 8 lanes and lands the prefill leg exactly on the granularity, at the cost of nothing but
#: activation size. THE RISK IS NOT ASSUMED AWAY. If the seam declines at the decode leg's width 9
#: it declines VISIBLY -- a gate decline routes to torch and the standing ``torch_fallback == 0``
#: item fails; a body refusal raises. Either way this file reports it, and either way it is a SEAM
#: finding for the lead and not a fixture number to re-tune until the suite goes quiet.
PREFILL_TOKENS = 35
COMPLETE_POOLS = PREFILL_TOKENS // POOL_SIZE
TAIL_TOKENS = PREFILL_TOKENS % POOL_SIZE
DECODE_STEPS = 1

#: Pools selected per query. ``can_run_dsa_topk_select`` refuses ``k == width`` -- its condition is
#: ``0 < k < width`` and it is load-bearing, because the kernel's own assert is not reachable by the
#: factory dry run. So a 2-pool prefix would force the degenerate ``k == 1``; four pools do not.
TOPK_POOLS = 2

#: Paged storage for the gather. The page is aligned to the pool so a pool never straddles a page.
PAGE_SIZE = 4
PAGES = 8

#: The stack sizes: the declared case and the moving control.
LAYERS = 3
CONTROL_LAYERS = 1

#: ``attend`` refuses any ``batch_size != 1`` by name -- an equality, not a threshold.
BATCH = 1

# --------------------------------------------------------------------------- #
# THE TINY MODEL GEOMETRY, and which seam forces each entry to be what it is.
#
# WHY A TINY GEOMETRY AT ALL. The checkpoint's own widths make a 3-layer stack roughly 1.4 GB of
# float32 weights, which is not a unit test. So the widths below are the SMALL ones -- and every
# entry is annotated with the gate that admits it or the gate that pins it, because a reader has to
# know which of these may be shrunk further and which may not. Each cite was taken by reading the
# seam's own validator, never from a memory of it.
#
# THE THREE THAT MAY NOT MOVE:
#   * ``index_head_dim = 128`` is PINNED, not chosen. Three seams refuse any other width by name --
#     ``kpool_hadamard.py:485`` and ``kpool_hadamard.py:564`` ("the Hadamard path is a 128-point transform"),
#     ``score_gemm.py:362``, ``decode_tail_update.py:475`` -- and the transform is 128-specific in
#     its own body: ``HADAMARD_STAGES`` (``kpool_hadamard.py:102-115``) is a fixed 7-entry table
#     whose stated invariant is ``groups * 2 * stride == 128``, and ``HADAMARD_SCALE``
#     (``kpool_hadamard.py:134``) is that width's reciprocal square root baked in as a literal.
#   * ``index_kpool = 4`` stays a power of two, which ``can_run_dsa_index_expand`` requires
#     (``index_expand.py:442``).
#   * ``qk_rope_head_dim = 0`` is the checkpoint's value and the whole point of the increment:
#     ``mla_sparse.py:1287`` admits it explicitly ("Zero IS admissible"), which is what this
#     increment's acceptance is meant to exercise.
#
# THE REST ARE FREE, and the sweep says so from the validators rather than from silence:
#   * ``num_attention_heads``: ``mla_sparse.py:1273`` is ``heads < 1 or heads > 128`` -- a floor of
#     one and a ceiling, no multiple-of rule; heads ride the matmul stationary free axis as a full
#     width, so there is no head-tiling granularity to satisfy. ``mla_absorb.py:260`` likewise.
#   * ``kv_lora_rank``: ``mla_sparse.py:1278`` is ``latent < 1``, and its message says why the old
#     bounds are gone ("inc-glm53f-041 TILES both axes this used to be bounded on"). 128 is chosen
#     over a ragged value ON PURPOSE: ``latent % 128 == 0`` selects the UNTILED body
#     (``mla_sparse.py:1420``), the same body production's 512 takes, so the tiled counters
#     (``mla_sparse.py:1421-1422``) stay at the zero this file's table declares. A ragged latent would move a
#     counter the declared table does not predict.
#   * ``q_lora_rank``, ``hidden_size``, ``qk_nope_head_dim``, ``v_head_dim``: ``mla_projection``
#     checks POSITIVITY only, and says so ("ONLY POSITIVITY IS CHECKED, and the absence of an upper
#     bound is the whole point of this module", ``mla_projections.py:219-223``).
#   * ``index_n_heads``: ``score_gemm.py:358`` is ``heads <= 0``; the module's ``INDEX_N_HEADS = 32``
#     (``score_gemm.py:126-132``) is annotated in its own source as "recorded for the reader; NOT a limit".
#
# ONE COUPLING TO RESPECT, so the cache shape is derived and not typed:
# ``head_size = kv_lora_rank + qk_rope_head_dim`` (``model_fp8.py:407-416``), so ``latent_cache``
# is ``[slots, 1, head_size]``; and ``attend`` refuses an absorb-in width that is not the latent
# (``model_fp8.py:4561-4565``), which holds because ``W_UK`` is
# ``[heads, qk_nope_head_dim, kv_lora_rank]``.
TINY_GEOMETRY: dict[str, int] = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "q_lora_rank": 128,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 0,
    "v_head_dim": 64,
    "index_n_heads": 4,
    "index_head_dim": INDEX_HEAD_DIM,
    "index_kpool": POOL_SIZE,
}

#: ``head_size``, DERIVED from the two config fields the way the implementation derives it
#: (``model_fp8.py:407-416``) rather than typed, so a config change reaches this file.
TINY_HEAD_SIZE = TINY_GEOMETRY["kv_lora_rank"] + TINY_GEOMETRY["qk_rope_head_dim"]

#: Registered tolerance for item (1), the plan block's own.
RTOL = 5e-3
ATOL = 1e-5

SENT = "DSALAYER"


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


# --------------------------------------------------------------------------- #
# THE NINE ENTRY POINTS AND THEIR SEVEN COUNTER FAMILIES.
#
# The family of an entry point is the module that holds it, because a counter is per module. Two
# modules hold two entry points each and say so in their own docstrings; that is the whole reason
# the readings below are not one number repeated.

ENTRY_POINTS: dict[str, str] = {
    "dsa_decode_tail_update": "decode_tail_update",
    "dsa_kpool_hadamard": "kpool_hadamard",
    "dsa_hadamard128": "kpool_hadamard",
    "dsa_paged_gather": "paged_gather",
    "dsa_ragged_pack": "ragged_pack",
    "dsa_ragged_unpack": "ragged_pack",
    "dsa_score_gemm": "score_gemm",
    "dsa_topk_select": "topk_select",
    "dsa_index_expand": "index_expand",
}

FAMILIES: tuple[str, ...] = tuple(sorted(set(ENTRY_POINTS.values())))

# PART 1 IS TWO CASES, AND THE SEVENTH FAMILY IS WHY.
#
# The layer case below never packs, and that is upstream's design rather than a gap here.
# Upstream guards its whole pack/unpack region with ``if decode_metadata.requires_padding:``
# (``sparse_attn_indexer_kpool.py:744`` at pin ``878631b6``, the unpack at ``sparse_attn_indexer_kpool.py:889``), and a
# batch of one is uniform by construction -- ``attend()`` refuses any ``batch_size != 1``
# (``model_fp8.py:3372``). So the layer case's ``ragged_pack`` reading is a DECLARED ZERO,
# not a miss, and inserting a pack there to move a counter would be the instrument driving
# the design. The seventh family is reached instead by a second case: a NON-UNIFORM decode
# batch called on ``Glm5NextDSAIndexer`` directly, which is where padding is required.
# ``7/7`` is read over the UNION of the two cases. Ruled by the lead at design entry
# ``design-20260905-aa`` (plan revision 207, DECISIONS §35), on this seat's evidence.

#: Calls per LAYER, per phase, per entry point in the LAYER CASE: ``(prefill, decode)``.
#: Each figure's reason is in ``probe-051-tiny-case-design.md``; the three zeros are the two
#: phase-exclusive entries plus the pack pair upstream does not reach at batch one.
DECLARED_PER_LAYER: dict[str, tuple[int, int]] = {
    "dsa_decode_tail_update": (0, 1),   # decode-only: the tail ring advances one token per step
    "dsa_kpool_hadamard": (1, 0),       # prefill-only: prefill sees whole pools
    "dsa_hadamard128": (1, 1),          # the indexer query is rotated every step; never pooled
    "dsa_paged_gather": (1, 1),         # the selected rows are read out of paged storage
    "dsa_ragged_pack": (0, 0),          # DECLARED ZERO: batch one is uniform, so no padding
    "dsa_ragged_unpack": (0, 0),        # and nothing to unpack
    "dsa_score_gemm": (1, 1),           # the indexer scores candidates
    "dsa_topk_select": (1, 1),          # and selects
    "dsa_index_expand": (1, 1),         # pool ids become token indices
}

#: Calls per LAYER in the RAGGED DECODE ARM, one non-uniform decode step, ``(prefill, decode)``.
#: These are ``Glm5NextDSAIndexer.forward_ragged``'s OWN declared readings, which its docstring
#: states entry point by entry point and ``probe-051-arm-authored.out`` compares against the
#: arm's resolved call graph, 31/31, exit 0.
#:
#: SUPERSEDED, and recorded rather than quietly replaced. An earlier draft of this table read
#: ``dsa_ragged_pack`` (0, 2) and ``dsa_ragged_unpack`` (0, 1), derived from upstream's branch:
#: upstream packs the quantised query (``sparse_attn_indexer_kpool.py:748`` or ``sparse_attn_indexer_kpool.py:756``), packs the query scale only when one
#: exists (``sparse_attn_indexer_kpool.py:751``), packs the weights (``sparse_attn_indexer_kpool.py:759``), and unpacks once (``sparse_attn_indexer_kpool.py:889``). Two of those three
#: figures do not survive the FORK's own gates. The query scale never existed here, which was
#: already known. The WEIGHTS cannot pack, because they are fp32 by ``dsa_score_gemm``'s gate and
#: ``dsa_ragged_pack`` admits bf16 alone -- so upstream's pair is a single pack here. And the
#: unpack has no consumer, because ``mla_sparse_attention`` refuses a rank-3 index tensor by name.
#: That collision is finding F11, ruled route (b).
DECLARED_PER_LAYER_RAGGED_ARM: dict[str, tuple[int, int]] = {
    "dsa_decode_tail_update": (0, 0),   # DECLARED ZERO: the arm selects; the ring is the
                                        # uniform case's business and advances there
    "dsa_kpool_hadamard": (0, 0),       # no prefill in the arm, so no whole pools to pool
    "dsa_hadamard128": (0, 1),          # the query is still rotated
    "dsa_paged_gather": (0, 1),
    "dsa_ragged_pack": (0, 1),          # the query ALONE: the fp32 weights cannot pack
    "dsa_ragged_unpack": (0, 0),        # DECLARED ZERO per F11 route (b); covered by -045
    "dsa_score_gemm": (0, 1),
    "dsa_topk_select": (0, 1),
    "dsa_index_expand": (0, 1),
}


def declared_family_totals(layers: int, table: dict[str, tuple[int, int]] | None = None) -> dict[str, int]:
    """The counter reading each family must show for a ``layers``-layer stack of one case."""
    out = {family: 0 for family in FAMILIES}
    for entry, (prefill, decode) in (table or DECLARED_PER_LAYER).items():
        out[ENTRY_POINTS[entry]] += (prefill + decode) * layers
    return out


#: The LAYER CASE's declared readings, spelled out so a reader sees them without running the
#: derivation. NOTE, disclosed rather than absorbed: plan L797's parenthetical illustration
#: reads "3 on the five single-entry families and 6 on the two shared ones", which is the
#: reading of a ONE-PHASE forward. This case runs both phases because two entry points are
#: phase-exclusive, so six families differ from that illustration. The two-phase table is the
#: declared expectation the lead ruled binding (DECISIONS §31); the illustration is not
#: re-derived here and no criterion moves.
DECLARED_FAMILY_TOTALS: dict[str, int] = {
    "decode_tail_update": 3,
    "index_expand": 6,
    "kpool_hadamard": 9,
    "paged_gather": 6,
    "ragged_pack": 0,
    "score_gemm": 6,
    "topk_select": 6,
}

#: The RAGGED ARM's declared readings at three layers. Its one-third control is the 1-layer arm.
#: ``kpool_hadamard`` reads 3 on the ROTATION half alone, which is why the entry-point table above
#: and not this family table is the place the arm's substrate is read.
DECLARED_FAMILY_TOTALS_RAGGED_ARM: dict[str, int] = {
    "decode_tail_update": 0,
    "index_expand": 3,
    "kpool_hadamard": 3,
    "paged_gather": 3,
    "ragged_pack": 3,
    "score_gemm": 3,
    "topk_select": 3,
}


def _seam_module(family: str):
    return importlib.import_module(f"vllm_neuron.functional.dsa.{family}")


def _discover_counter_api(module):
    """``(reset, read)`` on one module, DISCOVERED rather than spelled out.

    Spelling the names out would be a fifth place they are written, and one of them does not follow
    the module name: ``decode_tail_update.py``'s pair is ``reset_decode_tail_dispatch_counters`` and
    ``decode_tail_dispatch_counters``. Discovery also fails loudly if a module ever grows a second
    pair, which a hardcoded name would silently ignore.

    THE RULE LIVES HERE ONCE so the eighth family below is discovered by the SAME rule as the seven
    and not by a second convention written beside it. It takes a module rather than a family name
    because the eighth family is not under ``functional.dsa`` at all -- and that package prefix,
    baked into the family-name form this helper replaced, is the whole reason its reading was
    missing for twelve attempts.
    """
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    reset = [n for n in names if n.startswith("reset_")]
    read = [n for n in names if not n.startswith("reset_")]
    assert len(reset) == 1 and len(read) == 1, (module.__name__, reset, read)
    return getattr(module, reset[0]), getattr(module, read[0])


def _counter_api(family: str):
    """``(reset, read)`` for one of the SEVEN dsa families."""
    return _discover_counter_api(_seam_module(family))


def reset_all_counters() -> None:
    for family in FAMILIES:
        _counter_api(family)[0]()


def read_all_counters() -> dict[str, tuple[int, int]]:
    """``{family: (nki_dispatch, torch_fallback)}`` since the last reset."""
    return {family: tuple(int(v) for v in _counter_api(family)[1]()) for family in FAMILIES}


# --------------------------------------------------------------------------- #
# THE EIGHTH COUNTER FAMILY -- THE PROJECTION SEAM, AND WHY IT NEEDS ITS OWN PAIR.
#
# WHAT WAS MISSING, stated plainly because it is this increment's own defect and not a refinement.
# Review item B86 found ZERO occurrences of a reset/read pair for ``mla_projection`` in this file.
# The reason is structural rather than an oversight of attention: ``reset_all_counters`` and
# ``read_all_counters`` walk ``FAMILIES``, ``FAMILIES`` is derived from ``ENTRY_POINTS``, and every
# entry there lives under ``vllm_neuron.functional.dsa``. The projection seam lives under
# ``vllm_neuron.functional.attention``. So the walk could not reach it at any reading -- an
# instrument whose population EXCLUDED the thing the sentence was about, which is this increment's
# recurring defect class in its purest form. The seam's own reader says what it is for:
# "The counter is kept so a test can STATE that reading rather than assume it, which is what makes
# the zero a measurement" (``mla_projections.py:148-156``).
#
# WHY IT IS NOT FOLDED INTO ``FAMILIES``. ``FAMILIES`` is the census the CALL SPY attributes against,
# and the projection seam is not a dsa entry point: folding it in would move
# ``DECLARED_FAMILY_TOTALS``, the one-third control and the spy-versus-counter agreement check, none
# of which is about projections. The eighth family gets its own pair, its own declared closed form
# and its own readings, and the seven-family tables are untouched.

#: The module that holds the projection seam. NOT under ``functional.dsa``; see above.
PROJECTION_MODULE = "vllm_neuron.functional.attention.mla_projections"


def _projection_counter_api():
    """``(reset, read)`` for the projection seam, by the same discovery rule as the seven."""
    return _discover_counter_api(importlib.import_module(PROJECTION_MODULE))


def reset_projection_counter() -> None:
    _projection_counter_api()[0]()


def read_projection_counter() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the projection seam since the last reset."""
    return tuple(int(v) for v in _projection_counter_api()[1]())


#: The two phases the layer case runs, named once. The closed form below multiplies by
#: ``len(PHASES)`` and run 1 iterates this same tuple, so the count and the loop cannot drift.
PHASES: tuple[str, ...] = ("prefill", "decode")

#: THE CLOSED FORM, TERM BY TERM: projection dispatches ONE LAYER owes for ONE phase of the LAYER
#: CASE, each term with the caller that makes it and the seam call sites it reaches. A reader checks
#: any row against the source; the total is the sum and is never typed on its own.
#:
#: THE SAME NINE ARRIVE FROM THE REFERENCE SIDE, which is why this closed form is a derivation and
#: not a guess dressed as one. The torch reference in this file reaches ``_ref_projection`` nine times
#: per layer per phase, by a call graph written independently of the production one: ``_ref_layer``
#: projects once itself (the indexer's latent), calls ``_ref_indexer`` -> ``_ref_project_stage``
#: which projects four times, and calls ``_ref_attend`` which projects four times (q_a_proj, q_b_proj,
#: kv_a_proj_with_mqa, o_proj). Production splits that last four as 3 + 1 across
#: ``project_query_and_latent`` and ``project_output``; the sites are the same sites. Two structures
#: built by different passes agreeing on nine is worth more than either one restated twice.
#:
#: The third term is THREE and not two, and the reason is composition rather than arithmetic:
#: ``project_query_and_latent`` calls ``project_query_latent`` as its own first statement
#: (``model_fp8.py:4416``, whose docstring says so at ``model_fp8.py:4397``), so the nested dispatch
#: belongs to it. A count of the DIRECT ``mla_projection(`` lines in that method reads two and is the
#: wrong reading -- exactly the defect class this file's own history is made of.
#: Each row is ``(key, what calls it, dispatches, the source that makes the figure)``. The key is
#: stable and is what the arithmetic below selects on -- selecting on the prose would make a
#: reworded label change a number.
DECLARED_PROJECTION_TERMS: tuple[tuple[str, str, int, str], ...] = (
    (
        "layer_query_latent",
        "layer.forward -> attention.project_query_latent",
        1,
        "model_fp8.py:4693 calls :4332, whose one dispatch is :4375 (q_a_proj)",
    ),
    (
        "indexer",
        "indexer.forward -> project_stage",
        4,
        "model_fp8.py:3662 calls :3029, whose four dispatches are :3093 wq_b, :3106 wk, "
        ":3113 weights_proj, :3120 index_kpool_compress_gate",
    ),
    (
        "attend_query_and_latent",
        "attention.attend -> project_query_and_latent",
        3,
        "model_fp8.py:4543 calls :4378 = the NESTED project_query_latent at :4416 (-> :4375) "
        "plus :4417 q_b_proj plus :4420 kv_a_proj_with_mqa",
    ),
    (
        "attend_output",
        "attention.attend -> project_output",
        1,
        "model_fp8.py:4576 calls :4426, whose one dispatch is :4447 (o_proj)",
    ),
)

#: The arm's closed form. It calls the indexer DIRECTLY -- "no layer and no ``attend()`` is
#: involved" (``model_fp8.py:3710``) -- so the indexer term is the whole of it, and it projects the
#: padded grid ONCE rather than once per request, which is the commutation claim the arm exists to
#: test (``model_fp8.py:3824-3827``).
#:
#: FOUR, WHERE THE PREDICTIONS FILE SAYS TWELVE -- disclosed rather than settled by editing either
#: number. Prediction 6 reads "12 for the arm (indexer only, three layers' worth of four) and 4 for
#: its control", which is the THREE-LAYER idiom this file's own ``DECLARED_FAMILY_TOTALS_RAGGED_ARM``
#: uses ("at three layers"). The arm as written builds ONE layer and calls ``forward_ragged`` ONCE,
#: so the figure it owes is four -- which is prediction 6's own control figure. Nothing regressed,
#: no criterion moves, and the two documents describe the same closed form at two layer counts.
DECLARED_PROJECTION_TERMS_RAGGED_ARM: tuple[tuple[str, str, int, str], ...] = (
    (
        "indexer",
        "indexer.forward_ragged -> project_stage",
        4,
        "model_fp8.py:3829 calls :3029 ONCE on the flattened padded grid, four dispatches as above",
    ),
)


def projection_terms(table: tuple[tuple[str, str, int, str], ...]) -> dict[str, int]:
    """``{key: dispatches}``, and a duplicate key is a failure rather than a silent overwrite."""
    out: dict[str, int] = {}
    for key, _label, count, _cite in table:
        assert key not in out, f"duplicate projection term key {key!r}"
        out[key] = count
    return out


#: Dispatches ONE COMPLETING indexer call owes. Read OUT of the terms table rather than typed a
#: second time, so the per-call reading and the per-case total rest on one declaration.
PROJECTION_PER_INDEXER_CALL = projection_terms(DECLARED_PROJECTION_TERMS)["indexer"]

PROJECTION_PER_LAYER_PER_PHASE = sum(projection_terms(DECLARED_PROJECTION_TERMS).values())
PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE = (
    PROJECTION_PER_LAYER_PER_PHASE - PROJECTION_PER_INDEXER_CALL
)
PROJECTION_PER_ARM_CASE = sum(projection_terms(DECLARED_PROJECTION_TERMS_RAGGED_ARM).values())

# WHAT THE CLOSED FORM DELIBERATELY EXCLUDES, named so the total is not silently wrong later.
# ``Glm5NextMLAAttention.project_qkv`` (``model_fp8.py:4261``) holds FOUR more dispatch sites
# (:4287, :4289, :4292, :4294) and has NO caller anywhere in the tree -- it is dead on every path
# this file exercises, so the layer's reading is 9 and not 13. ``Glm5NextMLAAttention.forward``
# is still a stub (``model_fp8.py:4578``) and is likewise never on the path; that stub is one of
# the arms ``test_kv_spec.py`` keeps. Neither exclusion is asserted by a source scan here: the
# measured per-case total is the guard, because wiring either one in would move it.
#
# AND WHAT THE FALLBACK ZERO IS WORTH, disclosed rather than presented as a strong reading. This
# family's ``torch_fallback`` CANNOT be incremented by any code path: the module has no torch
# projection route and an inadmissible geometry raises instead (``mla_projections.py:151-155``).
# So the zero is a statement that stays true by construction, and it is asserted only because a
# torch route added later would make it a real reading. The reading with teeth is
# ``nki_dispatch``: it falls short if a dispatch is missed, rises if one is added, and falls short
# if a call is served by ``mla_projection_torch_oracle`` (``mla_projections.py:278``), which moves
# no counter at all.


# --------------------------------------------------------------------------- #
# THE TEST-SIDE CALL SPY.


class SeamSpy:
    """Counts every seam call and attributes it to the layer whose forward is running.

    It calls through and returns the real value unchanged: it alters no argument, caches nothing,
    swallows nothing, and moves no counter itself. The counters are moved by the real seams, which
    is what makes the agreement check a cross-check and not a tautology.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self._open: list[int | None] = []
        # The dtype of each call's FIRST tensor argument, logged for rider B71-N3. Kept in its own
        # list so no existing reading changes shape: `calls` still holds `(entry, layer)` and every
        # count above is computed from it exactly as before.
        self.first_arg_dtypes: list[tuple[str, torch.dtype]] = []

    # -- what the spy is asked --------------------------------------------------------------
    @property
    def current_layer(self) -> int | None:
        return self._open[-1] if self._open else None

    def per_entry(self) -> dict[str, int]:
        return dict(Counter(entry for entry, _ in self.calls))

    def per_family(self) -> dict[str, int]:
        out = {family: 0 for family in FAMILIES}
        for entry, _ in self.calls:
            out[ENTRY_POINTS[entry]] += 1
        return out

    def per_layer(self) -> dict[int | None, dict[str, int]]:
        out: dict[int | None, dict[str, int]] = {}
        for entry, layer in self.calls:
            out.setdefault(layer, {})
            out[layer][entry] = out[layer].get(entry, 0) + 1
        return out

    def uncalled(self) -> list[str]:
        called = self.per_entry()
        return sorted(e for e in ENTRY_POINTS if called.get(e, 0) == 0)

    def dtypes_for(self, entry: str) -> set[torch.dtype]:
        """Every dtype this entry's first tensor argument arrived as, on the PRODUCTION path.

        Rider B71-N3 needs this rather than a reconstruction: what matters is the dtype the indexer
        actually hands the seam when the layer runs, not the dtype a test can build by hand.
        """
        return {dt for name, dt in self.first_arg_dtypes if name == entry}

    # -- installation ----------------------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for entry, family in ENTRY_POINTS.items():
            module = _seam_module(family)
            monkeypatch.setattr(module, entry, self._wrap(entry, getattr(module, entry)))
        from vllm_neuron.model.glm5_next import model_fp8

        layer_cls = model_fp8.Glm5NextDSALayer
        monkeypatch.setattr(layer_cls, "forward", self._bracket(layer_cls.forward))

    def _wrap(self, entry: str, real):
        @functools.wraps(real)
        def wrapper(*args, **kwargs):
            self.calls.append((entry, self.current_layer))
            for value in (*args, *kwargs.values()):
                if isinstance(value, torch.Tensor):
                    self.first_arg_dtypes.append((entry, value.dtype))
                    break
            return real(*args, **kwargs)

        return wrapper

    def _bracket(self, real_forward):
        """Wrap the layer's forward so every seam call lands inside exactly one layer bracket.

        The bracket is what separates a seam called once per LAYER from one hoisted out of the layer
        loop and called once per MODEL: the second reads the same at one layer and at three, which is
        the defect the moving control exists to catch.
        """

        @functools.wraps(real_forward)
        def wrapper(layer_self, *args, **kwargs):
            self._open.append(int(getattr(layer_self, "layer_idx", -1)))
            try:
                return real_forward(layer_self, *args, **kwargs)
            finally:
                self._open.pop()

        return wrapper

    # -- what the spy reports --------------------------------------------------------------
    def report(self, label: str) -> None:
        per_entry = self.per_entry()
        say(label, "ENTRY_POINTS_CALLED", f"{len(per_entry)}/{len(ENTRY_POINTS)}")
        for entry in sorted(ENTRY_POINTS):
            say(label, "entry", entry, ENTRY_POINTS[entry], per_entry.get(entry, 0))
        for layer in sorted(self.per_layer(), key=lambda v: (v is None, v)):
            counts = self.per_layer()[layer]
            say(label, "layer", layer, "|".join(f"{k}={v}" for k, v in sorted(counts.items())))


def agreement(spy: SeamSpy, counters: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """``{family: (spy_sum, counter_delta)}`` -- two instruments counting the same events."""
    per_family = spy.per_family()
    return {family: (per_family[family], counters[family][0]) for family in FAMILIES}


# --------------------------------------------------------------------------- #
# THE PER-INDEXER-CALL PROJECTION READING.


class IndexerProjectionProbe:
    """Reads the projection counter's DELTA across every indexer call that completes.

    WHY A DELTA AND NOT A RESET INSIDE EACH CALL. A reset per call would destroy the per-case
    total, and the total is the SECOND instrument: the per-call readings and the case total are
    taken by different means over the same events, so the two can cross-check each other the way
    the spy and the seven family counters already do. A before-and-after read is the same
    reset/read pair with the reset replaced by a reading -- strictly more measurement, because it
    also leaves the running total intact to be checked against the closed form.

    It calls through and returns the real value unchanged, and it moves no counter itself: the
    counter is moved by the real seam inside the real call. The reading is recorded in a ``finally``
    so a call that RAISES still records what it dispatched before raising, rather than vanishing.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cls = _impl().Glm5NextDSAIndexer
        for method in ENTRY_METHODS:
            monkeypatch.setattr(cls, method, self._wrap(method, getattr(cls, method)))

    def _wrap(self, method: str, real):
        @functools.wraps(real)
        def wrapper(*args, **kwargs):
            before = read_projection_counter()
            try:
                return real(*args, **kwargs)
            finally:
                after = read_projection_counter()
                self.calls.append((method, after[0] - before[0], after[1] - before[1]))

        return wrapper

    @property
    def dispatched(self) -> int:
        """Every completing indexer call's dispatches, summed -- the indexer's share of the case."""
        return sum(nki for _method, nki, _fallback in self.calls)

    def check(self, label: str, *, want_calls: int) -> None:
        """Every recorded call read exactly ``(4, 0)``, and the call COUNT is itself a reading."""
        say(label, "indexer_calls", len(self.calls), "want", want_calls)
        # NON-EMPTINESS FIRST, as its own claim (DECISIONS §79.1). A per-call assertion over an
        # empty list passes while measuring nothing, which is how this file's rider B71-N3 passed
        # round 1 reading nothing at all.
        assert self.calls, (
            f"{label}: the probe recorded NO indexer call, so every per-call assertion below would "
            f"be a statement about an empty list and would pass having measured nothing. Either the "
            f"probe was not installed inside the measured window or the case never reached the "
            f"indexer"
        )
        assert len(self.calls) == want_calls, (
            f"{label}: the probe recorded {len(self.calls)} indexer calls where the case owes "
            f"{want_calls}. The count is a reading in its own right: a seam hoisted out of the "
            f"layer loop reads the same per call and a different number of times"
        )
        for idx, (method, nki, fallback) in enumerate(self.calls):
            say(
                label, "indexer_call", idx, method,
                "nki_dispatch", nki, "want", PROJECTION_PER_INDEXER_CALL,
                "torch_fallback", fallback, "want", 0,
            )
            assert nki == PROJECTION_PER_INDEXER_CALL, (
                f"{label}: indexer call {idx} ({method}) dispatched {nki} projections where "
                f"project_stage owes exactly {PROJECTION_PER_INDEXER_CALL} -- one per site at "
                f"model_fp8.py:3093, :3106, :3113, :3120. A short count means a site was served by "
                f"mla_projection_torch_oracle (which moves no counter) or was not reached at all; "
                f"a long one means a site projects more than once"
            )
            assert fallback == 0, (
                f"{label}: indexer call {idx} ({method}) recorded {fallback} torch fallbacks. This "
                f"counter has no code path that increments it today (mla_projections.py:151-155), "
                f"so a non-zero reading here means a torch projection route was added and the "
                f"substrate declaration for this increment must be re-derived, not this number"
            )


def projection_closed_form(
    label: str, table: tuple[tuple[str, str, int, str], ...], *, layers: int, phases: int
) -> int:
    """Print the closed form TERM BY TERM with each term's source, and return the total it declares.

    Printed rather than only asserted because a bare total tells a reader nothing about which term
    moved when it changes. Every row carries the caller and the seam lines that make its figure.
    """
    per_layer_per_phase = sum(count for _key, _what, count, _cite in table)
    total = 0
    for key, what, count, cite in table:
        owed = count * layers * phases
        total += owed
        say(
            label, "projection_term", key, what,
            f"{count} x {layers} layers x {phases} phases = {owed}", cite,
        )
    say(
        label, "projection_closed_form",
        " + ".join(str(count) for _key, _what, count, _cite in table),
        f"= {per_layer_per_phase} per layer per phase",
        f"x {layers} layers x {phases} phases = {total}",
    )
    return total


# --------------------------------------------------------------------------- #
# THE GATE PRECONDITION.


def gate_live() -> bool:
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return bool(can_run_kernel())


def test_the_nki_gate_is_live_before_any_count_is_read() -> None:
    """Without this, every family reads 0 and the fallback check fails -- both false negatives.

    Asserted rather than assumed, and its inputs are printed, so a run in the wrong mode says which
    condition refused rather than reporting seven silent zeros.
    """
    import os

    from vllm_neuron import envs

    say("gate", "VLLM_NEURON_CPU_MODE", bool(envs.VLLM_NEURON_CPU_MODE))
    say("gate", "NKI_SIMULATOR", os.environ.get("NKI_SIMULATOR"))
    say("gate", "VLLM_NEURON_DISABLE_NKI_KERNELS", bool(envs.VLLM_NEURON_DISABLE_NKI_KERNELS))
    say("gate", "can_run_kernel", gate_live())
    assert gate_live(), (
        "the NKI route is not available, so every seam would take its torch oracle and all seven "
        "counter families would read 0. Run as VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 with "
        "VLLM_NEURON_DISABLE_NKI_KERNELS unset"
    )


def test_the_declared_table_is_internally_consistent() -> None:
    """The declared readings must cover the census, reach every family, and keep the one-third form.

    This is a control on the DECLARATION, not on the layer: it fails if the table and the module
    census ever drift apart, which is the way a passing count could come to mean nothing.
    """
    assert sorted(DECLARED_PER_LAYER) == sorted(ENTRY_POINTS)
    assert sorted(DECLARED_PER_LAYER_RAGGED_ARM) == sorted(ENTRY_POINTS)
    assert len(ENTRY_POINTS) == 9
    assert len(FAMILIES) == 7
    shared = sorted(f for f in FAMILIES if sum(1 for m in ENTRY_POINTS.values() if m == f) > 1)
    assert shared == ["kpool_hadamard", "ragged_pack"]

    totals = declared_family_totals(LAYERS)
    assert totals == DECLARED_FAMILY_TOTALS, (totals, DECLARED_FAMILY_TOTALS)
    arm = declared_family_totals(LAYERS, DECLARED_PER_LAYER_RAGGED_ARM)
    assert arm == DECLARED_FAMILY_TOTALS_RAGGED_ARM, (arm, DECLARED_FAMILY_TOTALS_RAGGED_ARM)

    # 7/7 IS READ OVER THE UNION, which is what the two-case shape of part 1 is for.
    union_zero = sorted(f for f in FAMILIES if totals[f] == 0 and arm[f] == 0)
    assert union_zero == [], f"families no case reaches: {union_zero}"
    say("union", "families_nonzero", sum(1 for f in FAMILIES if totals[f] or arm[f]), "of", len(FAMILIES))

    # The layer case's one zero is DECLARED, and naming it here is what stops a silent miss
    # from passing as a declaration later.
    assert sorted(f for f, v in totals.items() if v == 0) == ["ragged_pack"]
    assert arm["ragged_pack"] > 0, "the arm exists to reach the family the layer case declares zero"

    # The one-third moving control holds SEPARATELY on each case, over the families that case
    # reaches. A family the case declares zero cannot exercise the control, so it is excluded by
    # its own declared value rather than by an exception typed here.
    for name, table, spelled in (
        ("layer_case", DECLARED_PER_LAYER, totals),
        ("ragged_arm", DECLARED_PER_LAYER_RAGGED_ARM, arm),
    ):
        control = declared_family_totals(CONTROL_LAYERS, table)
        for family in FAMILIES:
            if spelled[family]:
                assert spelled[family] == LAYERS * control[family], (name, family, spelled, control)
            say(name, family, spelled[family], "one_third", control[family])

    # EIGHT OF NINE ENTRY POINTS ARE REACHED, and the ninth is named here with the evidence that
    # covers it. This check keeps its teeth: exactly ONE name is admitted, sourced to the lead's
    # ruling, so any OTHER unreached entry point still fails. An unreached entry point with no
    # ruling behind it is the contradiction the lead declared and must STOP rather than be
    # reworded -- which is what this seat did when it found F11 instead of inventing a payload.
    RULED_DECLARED_ZERO = {
        "dsa_ragged_unpack": (
            "design-20260905-aj / plan revision 218 / DECISIONS §52, route (b): no admissible "
            "payload -- the fp32 gate weights cannot enter a bf16-only pack, and "
            "mla_sparse_attention refuses a rank-3 index tensor by name. The direction's own "
            "coverage is inc-glm53f-045's LANDED bit-identical round-trip item"
        )
    }
    unreached = sorted(
        e
        for e in ENTRY_POINTS
        if sum(DECLARED_PER_LAYER[e]) + sum(DECLARED_PER_LAYER_RAGGED_ARM[e]) < 1
    )
    for entry in unreached:
        say("declared_zero", entry, RULED_DECLARED_ZERO.get(entry, "NO RULING"))
    assert unreached == sorted(RULED_DECLARED_ZERO), (
        f"entry points no case calls: {unreached}; ruled declared zeros: "
        f"{sorted(RULED_DECLARED_ZERO)}"
    )
    reached = len(ENTRY_POINTS) - len(unreached)
    say("union", "entry_points_reached", f"{reached}/{len(ENTRY_POINTS)}")
    assert reached == 8, reached

    # The arm's own declared zero is a FAMILY, and the union is what covers it: the arm never
    # advances the ring, and the layer case's decode step does it three times.
    assert DECLARED_FAMILY_TOTALS_RAGGED_ARM["decode_tail_update"] == 0
    assert DECLARED_FAMILY_TOTALS["decode_tail_update"] > 0

    # The phase-exclusive pair is declared as such, so a one-phase case cannot pass unnoticed.
    assert DECLARED_PER_LAYER["dsa_decode_tail_update"] == (0, 1)
    assert DECLARED_PER_LAYER["dsa_kpool_hadamard"] == (1, 0)

    # The arm is decode-only, so its prefill column is entirely zero. A stray prefill figure here
    # would mean the arm had quietly become a second layer case.
    assert all(p == 0 for p, _ in DECLARED_PER_LAYER_RAGGED_ARM.values())

    # THE PROJECTION CLOSED FORM, checked here as a DECLARATION and measured in the two counted runs.
    # These figures are the predictions file's prediction 6 -- "9 per layer per phase, four in the
    # indexer's project_stage, one in the layer's own project_query_latent, three inside
    # project_query_and_latent, one in project_output" -- so the file and the predictions agree by an
    # assertion rather than by two readers hoping they match. Editing a term without re-deriving the
    # closed form fails HERE, before any run reads a counter.
    terms = projection_terms(DECLARED_PROJECTION_TERMS)
    arm_terms = projection_terms(DECLARED_PROJECTION_TERMS_RAGGED_ARM)
    say("projection", "terms", "|".join(f"{k}={v}" for k, v in sorted(terms.items())))
    say("projection", "per_layer_per_phase", PROJECTION_PER_LAYER_PER_PHASE,
        "indexer", PROJECTION_PER_INDEXER_CALL,
        "non_indexer", PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE,
        "arm_case", PROJECTION_PER_ARM_CASE, "phases", len(PHASES))
    assert sorted(terms) == [
        "attend_output", "attend_query_and_latent", "indexer", "layer_query_latent"
    ], sorted(terms)
    assert PROJECTION_PER_LAYER_PER_PHASE == 9, PROJECTION_PER_LAYER_PER_PHASE
    assert PROJECTION_PER_INDEXER_CALL == 4, PROJECTION_PER_INDEXER_CALL
    assert PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE == 5
    assert len(PHASES) == 2, PHASES

    # THE ARM'S ONLY TERM IS THE INDEXER'S, at the same figure, because the arm calls the indexer
    # directly with no layer and no attend around it. So the arm reads FOUR for its one call.
    assert sorted(arm_terms) == ["indexer"], sorted(arm_terms)
    assert arm_terms["indexer"] == PROJECTION_PER_INDEXER_CALL
    assert PROJECTION_PER_ARM_CASE == PROJECTION_PER_INDEXER_CALL


def test_the_counter_api_is_discovered_and_not_spelled_out() -> None:
    """Each family exposes exactly one reset and one reader, and a fresh reset reads (0, 0).

    The discovery is what keeps the accessor names out of this file. ``decode_tail_update.py``'s pair
    does not follow its module name, so a spelled-out name is a defect waiting for a rename.
    """
    reset_all_counters()
    readings = read_all_counters()
    for family in FAMILIES:
        say("counters", family, readings[family])
    assert set(readings) == set(FAMILIES)
    assert all(v == (0, 0) for v in readings.values()), readings

    # THE EIGHTH FAMILY, discovered by the SAME rule out of a DIFFERENT package -- and its pair does
    # not follow its module name either: ``mla_projections.py`` exposes
    # ``reset_mla_projection_dispatch_counters``, singular where the module is plural. That is a
    # second reason the rule is discovery and not spelling.
    reset_fn, read_fn = _projection_counter_api()
    say("counters", "projection_api", reset_fn.__name__, read_fn.__name__)
    reset_projection_counter()
    projection = read_projection_counter()
    say("counters", PROJECTION_MODULE, projection)
    assert projection == (0, 0), projection

    # THE COUNTER IS ITS OWN, and this is the reading that matters rather than a restatement of the
    # import path. If the projection seam shared a counter object with any dsa family, the per-call
    # readings below would be that family's dispatches as well and every figure would be double.
    # Distinct reset functions is what says they are distinct counters.
    for family in FAMILIES:
        assert _counter_api(family)[0] is not reset_fn, family
        assert _counter_api(family)[1] is not read_fn, family
    # And the seven-family walk cannot reach it, which is WHY its reading was missing until now.
    assert not PROJECTION_MODULE.startswith("vllm_neuron.functional.dsa."), PROJECTION_MODULE


# --------------------------------------------------------------------------- #
# THE REFUSALS, AND WHY THEY ARE TESTS RATHER THAN DEFENSIVE HABITS.
#
# Each refusal below is a design decision the lead RULED, so each one is worth a test that
# fails if the refusal is ever softened into a branch. Two of them share one property that
# the tests assert directly: they fire BEFORE the indexer touches a weight. That is what
# lets an unmaterialised indexer exercise them, and it is also what stops a load error from
# masking a config error.
#
# These items are NOT among the counted runs. Each one resets the counters, expects a raise,
# and then asserts every counter still reads zero -- a refusal that dispatched first would be
# a refusal that already did the wrong thing.

ENTRY_METHODS: tuple[str, ...] = ("forward", "forward_ragged")


def _impl():
    """Import the implementation module INSIDE a test body, never at import.

    The sibling KDA layer test's idiom (``test_kda_layer.py:164``), for its reason: a
    module-level import would make a collection error out of what should be a test failure.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _tiny_text_config(**overrides):
    """The checkpoint's own config, narrowed to :data:`TINY_GEOMETRY` and the tiny ``select_k``.

    ``dataclasses.replace`` rather than a fresh construction, so every field this file does NOT
    name keeps the checkpoint's value and a config drift reaches this file instead of being
    overwritten by it.

    ``select_k()`` is ``index_topk // index_kpool``, so ``TOPK_POOLS`` pools per query needs
    ``index_topk = TOPK_POOLS * POOL_SIZE``: the dial this file may move is the one upstream itself
    expresses in TOKENS, and the pool granularity stays derived rather than typed.

    THE NARROWING IS THE POINT, not a convenience. At the checkpoint's own widths a 3-layer stack
    is roughly 1.4 GB of float32 weights, which no unit test can carry. :data:`TINY_GEOMETRY`
    records, entry by entry, which seam admits each width and which three may not be shrunk.
    """
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    fields = dict(TINY_GEOMETRY)
    fields["index_topk"] = TOPK_POOLS * POOL_SIZE
    fields.update(overrides)
    return replace(Glm5NextTextConfig(), **fields)


def _bare_indexer(**overrides):
    """An indexer with NO weight materialised, which is enough to reach both preconditions."""
    return _impl().Glm5NextDSAIndexer(_tiny_text_config(**overrides))


def _reach(indexer, method: str, *, max_seq_len: int, pool_rows: int | None = None):
    """Call one entry point with operands shaped to reach its preconditions and no further.

    Both entry points run ``require_dials()`` first and ``_require_serviceable()`` second, so
    the operands here need only survive being passed. Shaping them well enough to reach a
    LATER stage would be a different test.
    """
    head_dim = int(indexer.index_head_dim)
    rows = PAGES * PAGE_SIZE if pool_rows is None else int(pool_rows)
    pool_cache = torch.zeros(rows, head_dim, dtype=torch.bfloat16)
    hidden = torch.zeros(1, int(indexer.hidden_size), dtype=torch.float32)
    latent = torch.zeros(1, int(indexer.q_lora_rank), dtype=torch.float32)
    seq_lens = torch.full((1,), int(max_seq_len), dtype=torch.int32)
    if method == "forward":
        return indexer.forward(
            hidden,
            latent,
            pool_cache,
            seq_lens,
            max_seq_len=int(max_seq_len),
            page_size=PAGE_SIZE,
            slot_mapping=torch.zeros(1, dtype=torch.int32),
        )
    return indexer.forward_ragged(
        hidden.reshape(1, 1, -1),
        latent.reshape(1, 1, -1),
        pool_cache,
        seq_lens,
        [1],
        max_seq_len=int(max_seq_len),
        page_size=PAGE_SIZE,
    )


def _refuses(method: str, indexer, *, max_seq_len: int, pool_rows: int | None = None) -> str:
    """Run one entry point, require a named refusal, and require that NOTHING dispatched.

    THE PROJECTION SEAM IS READ HERE TOO, and this is the one class of indexer call whose declared
    reading is ZERO rather than four. A refusal fires inside ``require_dials()`` and
    ``_require_serviceable()``, both of which run before ``project_stage`` is reached, so a
    refusing call that had already projected would have spent four kernel dispatches on operands it
    then declared unserviceable -- the wrong answer, already computed. Every OTHER indexer call in
    this file owes exactly four; these eight owe none, and both figures are the same claim about
    where the refusal sits.
    """
    reset_all_counters()
    reset_projection_counter()
    with pytest.raises(_impl().Glm5NextDSAIndexerError) as caught:
        _reach(indexer, method, max_seq_len=max_seq_len, pool_rows=pool_rows)
    readings = read_all_counters()
    projection = read_projection_counter()
    for family in FAMILIES:
        say("refusal", method, family, readings[family])
    say("refusal", method, "mla_projection", projection, "want", (0, 0))
    assert all(v == (0, 0) for v in readings.values()), (
        f"a precondition refused AFTER dispatching: {readings}. The refusal exists to keep the "
        f"wrong answer unreachable, so anything it lets run first is already the wrong answer"
    )
    assert projection == (0, 0), (
        f"a precondition refused AFTER projecting: the projection seam read {projection}. "
        f"require_dials() and _require_serviceable() both run before project_stage, so a non-zero "
        f"reading here means the refusal moved and now fires downstream of four kernel dispatches"
    )
    return str(caught.value)


DIALS: tuple[str, ...] = ("index_kpool_compress", "index_kpool_always_select_tail")


@pytest.mark.parametrize("method", ENTRY_METHODS)
@pytest.mark.parametrize("dial", DIALS)
def test_a_false_compress_dial_is_refused_by_name_before_anything_dispatches(
    method: str, dial: str
) -> None:
    """F9's precondition: BOTH dials are read, and neither is a switch.

    Upstream refuses the same two values with ``NotImplementedError``
    (``transformers_utils/configs/glm5_next.py:118-126``); this indexer refuses them because
    the landed ``dsa_index_expand`` seam expands POOL ids and appends the tail itself, so
    ``False`` on either dial would need an expansion no landed seam provides. Ruled at
    design entry ``design-20260905-af`` as a PRECONDITION plus a READING, which is why this
    is two items per entry point and not a branch.

    ON "THE TAIL", stated precisely because this docstring said "the tail block" and that was
    loose twice over: the appended columns carry RAW TOKEN indices ``tail_start + t``, not a
    pool id, and there are ``pool_size - 1`` of them whether or not any is populated -- the
    whole append masks to ``-1`` when ``seq_len % pool_size == 0`` (``index_expand.py:9``,
    ``index_expand.py:533-544``). This case's ``PREFILL_TOKENS = 19`` gives ``19 % 4 = 3``, so three tail
    columns are populated and the reading is non-degenerate; the assertion below does not
    depend on that, but a reader comparing the two numbers should not have to derive it.

    Parametrised over BOTH entry points on purpose: the precondition was extracted into one
    method so the two paths cannot drift apart, and a shared refusal that only guards one
    caller is worse than no extraction.
    """
    indexer = _bare_indexer(**{dial: False})
    assert getattr(indexer, dial) is False
    message = _refuses(method, indexer, max_seq_len=PREFILL_TOKENS)
    say("refusal", method, dial, message.split(";")[0])
    assert dial in message, f"the refusal must NAME the dial it refuses; got: {message}"
    assert "glm5_next.py:118-126" in message, "and cite the upstream refusal it mirrors"


@pytest.mark.parametrize("method", ENTRY_METHODS)
def test_a_sequence_too_short_to_select_is_refused_by_name_and_routed_to_099(
    method: str,
) -> None:
    """F10's refusal: below the selection bound the indexer refuses instead of clamping ``k``.

    The bound is ``candidates > select_k()``, and it is strict for a reason worth the test:
    ``dsa_topk_select``'s gate is ``0 < k < width`` and it RETURNS FALSE rather than raising
    (``topk_select.py:294``), so serving ``k == width`` would silently take the torch route
    and break the standing torch-fallback-reads-zero check on a case that looked like a pass.
    Upstream bypasses selection entirely in this regime
    (``sparse_attn_indexer_kpool.py:203-217``); that route is ``inc-glm53f-099``'s, so this
    refusal NAMES it rather than half-implementing it.

    ``max_seq_len`` here is ``REFUSED_SEQ_LEN``, which yields exactly ``select_k()``
    candidates -- one short of the strict bound. That is the interesting value: a clamp would
    pass here, and a non-strict bound would too.
    """
    indexer = _bare_indexer()
    select_k = int(indexer.select_k())
    assert select_k == TOPK_POOLS, (select_k, TOPK_POOLS)
    # The refused length is DERIVED from the bound rather than typed, so a config change moves
    # the case instead of silently making it pass.
    refused_seq_len = select_k * POOL_SIZE
    assert refused_seq_len // POOL_SIZE == select_k
    assert PREFILL_TOKENS // POOL_SIZE > select_k, (
        "the counted case must sit ABOVE the bound, or every counted run would refuse"
    )
    message = _refuses(method, indexer, max_seq_len=refused_seq_len)
    say("refusal", method, "F10", refused_seq_len, message.split(".")[0])
    assert "inc-glm53f-099" in message, "the refusal must name the increment that owns the route"
    assert "topk_select.py:294" in message, "and cite the gate that returns False rather than raising"
    assert str(select_k) in message and str(refused_seq_len) in message


def test_the_two_entry_points_refuse_in_the_SAME_order() -> None:
    """A config that breaks BOTH preconditions must report the DIAL, on both entry points.

    This is the extraction's real contract. ``require_dials()`` runs before
    ``_require_serviceable()``, so a config that is both undialled and too short must name the
    dial; if one entry point reported the length instead, the two paths would have drifted and
    a reader could not predict which refusal a bad config produces.
    """
    messages = {}
    for method in ENTRY_METHODS:
        indexer = _bare_indexer(index_kpool_compress=False)
        messages[method] = _refuses(method, indexer, max_seq_len=TOPK_POOLS * POOL_SIZE)
    for method, message in messages.items():
        say("order", method, message.split(";")[0])
        assert "index_kpool_compress" in message
        assert "inc-glm53f-099" not in message, (
            "the dial must be reported first: it is the cheaper, earlier and more likely "
            "operator error, and reporting the length would send a reader to the wrong route"
        )


# =========================================================================== #
# THE THREE SELECTION-FOLD RIDERS, folded here at the lead's word (DECISIONS §63, revision 225).
#
# These are review findings bound to this increment at revisions 201 and 202. None of them changes a
# declared acceptance number; each one adds a reading the acceptance did not previously take. The
# third rider is comment-only and rides in `ragged_pack.py` itself, not here.
#
# WHY THE FIRST TWO ARE WORTH A TEST AT ALL, and it is the same reason in both cases: a dtype that
# drifts to float32 breaks NOTHING VISIBLY. Every shape check still passes, the seam's gate quietly
# returns False, the torch oracle serves the call, and the only symptom is a counter reading that a
# reader has to notice. So the readings below are about the ROUTE, not about numerics.
# =========================================================================== #


def test_rider_B71_N3_the_indexer_hands_the_pooling_seam_bf16_keys() -> None:
    """B71-N3: the indexer's key and gate reach ``dsa_kpool_hadamard`` as bf16, or production runs torch.

    THE FINDING, in the reviewer's terms (``impl-batch-B71.md:26-28``): a non-bf16 ``slot_k`` fails
    ``can_run_dsa_kpool_hadamard``'s dtype clause and silently takes the torch oracle, so a passing
    acceptance could sit on top of a production path that never reaches the kernel.

    WHAT THIS TEST MEASURES, and it needs no simulator. ``project_stage`` is pure torch, so the dtypes
    it returns are readable on any host. It casts the key to bf16 BEFORE the norm and the norm casts
    back to its input's dtype (``model_fp8.py:3002``, ``return normed.to(x.dtype)``), so the bf16
    survives the norm -- which is the link the finding turns on and the one a refactor would break.

    WHAT IT DOES NOT MEASURE, said plainly rather than implied. It does not read the gate: every
    ``can_run_dsa_*`` predicate begins with ``can_run_kernel()``, which is False in CPU mode without
    ``NKI_SIMULATOR=1``, so a gate call here would return False for the environment rather than for the
    dtype and would settle nothing. The gate's own verdict is read on the production path in run 1,
    where the simulator is live.

    THE SECOND HALF OF THE FINDING IS NOT MINE TO FIX, and this records it rather than papering over
    it: ``can_run_dsa_hadamard128`` (``kpool_hadamard.py:505-509``) tests rank and width and has NO
    dtype clause at all, while its sibling ``can_run_dsa_kpool_hadamard`` (``kpool_hadamard.py:496``) does. So the two
    entry points of one counter family admit different dtypes. That asymmetry belongs to the seam's
    owner; this increment reports it and does not edit another block's gate.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import _SUPPORTED_DTYPES

    assert _SUPPORTED_DTYPES == (torch.bfloat16,), (
        f"the seam's admitted dtypes are {_SUPPORTED_DTYPES}, not bf16 alone, so this rider's premise "
        f"has changed and the finding needs re-reading rather than this assertion relaxing"
    )

    gen = torch.Generator().manual_seed(9_051_071)
    indexer = _bare_indexer()
    _materialise_indexer(indexer, gen)
    tokens = PREFILL_TOKENS
    hidden = torch.randn(tokens, int(indexer.hidden_size), generator=gen, dtype=torch.float32)
    q_latent = torch.randn(tokens, int(indexer.q_lora_rank), generator=gen, dtype=torch.float32)

    # THE RESET/READ PAIR AROUND THIS CALL. It is the third class of indexer call in this file and
    # the only one that reaches ``project_stage`` directly, so the pair sits here literally rather
    # than through the probe the two counted runs install on ``forward``/``forward_ragged``. Four
    # dispatches for four sites, and the rider's own claim depends on it: the dtypes below are what
    # the KERNELS returned only if the kernels ran, and a torch oracle serving all four would return
    # the same shapes with no other symptom.
    reset_projection_counter()
    query, key, weights, gate_score = indexer.project_stage(hidden, q_latent)
    projection = read_projection_counter()
    say("B71-N3", "project_stage", "mla_projection", projection,
        "want", (PROJECTION_PER_INDEXER_CALL, 0))
    assert projection == (PROJECTION_PER_INDEXER_CALL, 0), (
        f"project_stage read {projection} where it owes "
        f"({PROJECTION_PER_INDEXER_CALL}, 0) -- one dispatch per site at model_fp8.py:3093, :3106, "
        f":3113, :3120. Without this the dtypes asserted below could be a torch oracle's, and this "
        f"rider is about the ROUTE the seam takes"
    )
    for name, tensor, want in (
        ("query", query, torch.bfloat16),
        ("key", key, torch.bfloat16),
        ("gate_score", gate_score, torch.bfloat16),
        ("weights", weights, torch.float32),
    ):
        say("B71-N3", name, str(tensor.dtype))
        assert tensor.dtype == want, (
            f"project_stage returned {name} as {tensor.dtype}, not {want}. A {name} of the wrong "
            f"dtype passes every shape check and sends the pooling or scoring seam to its torch "
            f"oracle, which shows up only as a counter reading"
        )


def test_rider_B67_N4_the_ragged_pack_admits_bf16_only_and_preserves_it() -> None:
    """B67-N4: the pack's output dtype is its input's, and bf16 is the only dtype it admits.

    THE FINDING: ``-045``'s own tests never read the packed output's dtype, so this is a genuine
    addition rather than a duplicate. The module declares ``_SUPPORTED_DTYPES = (torch.bfloat16,)``
    (``ragged_pack.py:136``) and both kernels allocate their output in the INPUT's dtype
    (``ragged_pack.py:334``, ``ragged_pack.py:450``), so a dtype change would be visible here and nowhere else in the suite.

    Read without a simulator on purpose: the declared admission and the torch path's dtype
    preservation are both readable on any host, and the ARM in run 2 reads the same property on the
    NKI path where the gate is live. Two readings of one property by different routes.
    """
    from vllm_neuron.functional.dsa import ragged_pack as rp

    assert rp._SUPPORTED_DTYPES == (torch.bfloat16,), (
        f"the pack admits {rp._SUPPORTED_DTYPES}; the rider is written against bf16 alone"
    )
    lengths = [PREFILL_TOKENS // 2, PREFILL_TOKENS]
    max_len, width = max(lengths), INDEX_HEAD_DIM
    gen = torch.Generator().manual_seed(9_051_067)
    padded = torch.randn(
        len(lengths), max_len, width, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)

    packed = rp.dsa_ragged_pack(padded, lengths)
    say("B67-N4", "packed", tuple(packed.shape), str(packed.dtype))
    assert packed.dtype == padded.dtype, (
        f"the pack returned {packed.dtype} from a {padded.dtype} input; the seam contracts to return "
        f"the input's dtype and a silent widening would change what the indexer scores"
    )
    assert tuple(packed.shape) == (sum(lengths), width), (
        f"the packed shape {tuple(packed.shape)} is not the closed form {(sum(lengths), width)} -- "
        f"the caller never states the packed length, so this is a measurement and not an echo"
    )


# =========================================================================== #
# THE TORCH REFERENCE FOR ITEM (1).
#
# WHAT IS INDEPENDENT HERE AND WHAT IS NOT, stated first, because item (1) is only worth running if
# the answer is not smuggled in from the thing under test.
#
#   * THE ORCHESTRATION IS RE-WRITTEN. The chain -- four projections, the pooling window, the tail
#     ring, the cache writes, the gather, the score, the selection, the expansion, and the attention
#     -- is written again below from the implementation's declared contracts. This is the part item
#     (1) actually tests: a reference that shared the orchestration would cancel a composition error
#     out and pass vacuously.
#   * THE SEAM MATHEMATICS IS ALSO RE-WRITTEN, and for a reason that is not purity. Each DSA seam
#     ships its own torch reference and its own docstring calls it "the reference the NKI route is
#     measured against". Calling those would have been the obvious move. It is WRONG here, because
#     the ``torch_fallback`` counter bump sits INSIDE the private reference for three of the five
#     seams and in the PUBLIC WRAPPER for the other two:
#         inside the reference  -- kpool_hadamard.py:607, :616; topk_select.py:362;
#                                  decode_tail_update.py:584
#         in the wrapper        -- score_gemm.py:410; index_expand.py:387
#     So calling three of those five would make ``torch_fallback`` nonzero for reasons that have
#     nothing to do with the production path, and the route predicate's standing ``torch_fallback ==
#     0`` item -- the instrument that proves the kernels ran at all -- would read a number this test
#     itself put there. That inconsistency is reported to the lead as a finding about landed code; it
#     is not worked around by resetting counters at a lucky moment.
#   * TWO THINGS ARE IMPORTED, and both are DATA rather than algorithm: ``hadamard_matrix`` and
#     ``HADAMARD_SCALE``. The matrix fixes the transform's BASIS and sign convention -- hand-building
#     a second Sylvester matrix would risk a convention mismatch that reads as a numeric failure of
#     a correct kernel. The scale is a literal whose docstring records that the obvious spelling
#     ``1.0 / math.sqrt(128)`` is one ULP LOW (``kpool_hadamard.py:134``), so re-deriving it is a
#     known way to fail a correct kernel. Importing both and multiplying by hand still compares the
#     kernel's in-place butterfly ``_fwht128_inplace`` against a plain matrix multiply, which is the
#     comparison that has content.
#
# EVERY FUNCTION BELOW CITES THE CONTRACT IT MIRRORS. The cites were read from source at this base,
# never recalled.


def _ref_scale(indexer) -> float:
    """``projection_scale()``: EXACTLY two folded factors (``model_fp8.py:3027``).

    Computed, never typed. The product is not the tidy reciprocal a reader expects -- at
    ``index_head_dim = 128`` and ``index_n_heads = 4`` it is not exactly ``1/16`` in binary -- so a
    typed literal would be a different number from the one the implementation applies.
    """
    return float(int(indexer.index_head_dim) ** -0.5) * float(int(indexer.index_n_heads) ** -0.5)


def _ref_projection(x: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    """``mla_projection``: a plain contraction, float32 in and out.

    The seam's own oracle is ``x.to(float32) @ weight.to(float32)``
    (``mla_projections.py:285``) and its validator fixes the orientation at
    ``weight.shape[0] == in_features`` (``mla_projections.py:261``). This helper takes the CHECKPOINT-shaped
    ``[out, in]`` leaf and transposes it here, which is the same thing
    ``prepare_projection_weights`` does once at load time (``model_fp8.py:2916``) -- done
    independently so the reference does not read the implementation's cache.
    """
    return x.to(torch.float32) @ weight_out_in.to(torch.float32).t()


def _ref_absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``mla_absorb``: ``[S,H,K] x [H,K,N] -> [S,H,N]`` (``mla_absorb.py:338``)."""
    return torch.einsum("shk,hkn->shn", x.to(torch.float32), w.to(torch.float32))


def _ref_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """``_key_norm``: normalise in float32, cast back to the INPUT's dtype (``model_fp8.py:2995``).

    The cast back matters and is not cosmetic: ``project_stage`` casts the key to bf16 BEFORE the
    norm so the norm's own cast-back lands on bf16 (``model_fp8.py:3100-3109``), which is upstream's order.
    """
    width = int(x.shape[1])
    normed = torch.nn.functional.layer_norm(
        x.float(), (width,), weight.float(), bias.float(), float(eps)
    )
    return normed.to(x.dtype)


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Glm5NextDSALayer._input_norm`` (``model_fp8.py:4641-4645``)."""
    f = x.to(torch.float32)
    normed = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    return (normed * weight.to(torch.float32)).to(x.dtype)


def _ref_hadamard_matrix(width: int) -> torch.Tensor:
    """The transform's BASIS, taken from the seam so the sign convention cannot drift.

    ``hadamard_matrix`` (``kpool_hadamard.py:584``) builds the UNNORMALISED Sylvester matrix by
    doubling; the transposition and the scale are applied by the caller in both the seam's reference
    and here. This function exists so the import site is named once and the reason travels with it.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import hadamard_matrix

    return hadamard_matrix(int(width), dtype=torch.float32)


def _ref_hadamard_scale() -> float:
    """``HADAMARD_SCALE`` (``kpool_hadamard.py:134``), imported and never re-derived.

    Its own docstring records that ``1.0 / math.sqrt(128)`` is one ULP LOW, so a re-derivation is a
    documented way to fail a correct kernel.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import HADAMARD_SCALE

    return float(HADAMARD_SCALE)


def _ref_rotate(x: torch.Tensor) -> torch.Tensor:
    """``dsa_hadamard128``: rotate rows in float32, scale, cast back (``kpool_hadamard.py:617``)."""
    rotated = x.float() @ _ref_hadamard_matrix(int(x.shape[1])).t()
    return (rotated * _ref_hadamard_scale()).to(x.dtype)


def _ref_compress_prefill(
    slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """``dsa_kpool_hadamard``: pool then rotate, ONE cast at the end.

    Mirrors ``kpool_hadamard.py:608-611``. The softmax runs over the POOL axis and the score is
    PER FEATURE, so the weights are ``[n_pools, pool, head_dim]`` -- a per-feature weighting, not one
    scalar per pool member. Getting that axis wrong is the likeliest silent error in this reference,
    which is why the shape is asserted rather than trusted.
    """
    assert slot_k.ndim == 3 and slot_score.shape == slot_k.shape, (
        f"slot_k and slot_score must both be [n_pools, pool, head_dim]; got "
        f"{tuple(slot_k.shape)} and {tuple(slot_score.shape)}"
    )
    weights = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=1)
    pooled = (weights * slot_k.float()).sum(dim=1)
    rotated = pooled @ _ref_hadamard_matrix(int(slot_k.shape[2])).t()
    return (rotated * _ref_hadamard_scale()).to(slot_k.dtype)


def _ref_compress_decode(
    pool_key: torch.Tensor, pool_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """``_compress_pool_torch``: the SAME pooling, plus ONE EXTRA intermediate cast.

    A SEPARATE function from :func:`_ref_compress_prefill` on purpose, and the difference is one
    line. Decode rounds the pooled row through the tail's dtype BEFORE rotating
    (``decode_tail_update.py:575``, ``pooled = pooled.to(dtype).float()``); prefill rotates the
    float32 sum directly and casts once at the end (``kpool_hadamard.py:610-611``). At bf16 that is
    a real numeric difference, so folding the two into one function with a flag would make the
    reference agree with at most one of the two legs.

    The pool axis is ``dim=0`` here because the decode operand is a single ``[pool, head_dim]``
    window, against prefill's batched ``dim=1``.
    """
    dtype = pool_key.dtype
    weights = torch.softmax(pool_score.float() + ape.float(), dim=0)
    pooled = (weights * pool_key.float()).sum(dim=0, keepdim=True)
    pooled = pooled.to(dtype).float()  # the extra round trip: decode only
    rotated = pooled @ _ref_hadamard_matrix(int(pool_key.shape[1])).t()
    return (rotated * _ref_hadamard_scale()).to(dtype)


def _ref_score(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``dsa_score_gemm``: per-head scores, RECTIFIED, then head-weighted and summed.

    Mirrors ``score_gemm.py:447-448``. The ``clamp(min=0)`` is load-bearing -- it is a ReLU on each
    head's score before the head weight is applied, so dropping it would change which pools win, not
    merely the score values.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)


def _ref_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``dsa_topk_select`` then ``select_pools``' cast (``topk_select.py:369``, ``model_fp8.py:3439-3440``)."""
    return torch.topk(scores, int(k), dim=-1).indices.to(torch.int32)


def _ref_expand(pool_ids: torch.Tensor, seq_lens: torch.Tensor, pool_size: int) -> torch.Tensor:
    """``dsa_index_expand``: pool ids become token indices, with a raw-token tail appended.

    Written as an explicit LOOP where the seam's own reference is vectorised
    (``index_expand.py:525-546``). That is deliberate: a loop and a gather-and-mask are different
    enough that a transcription slip in either shows up as a mismatch rather than as a shared bug.

    THIS REFERENCE RETURNS THE RAW WIDTH, AND THAT IS THE FIXED CHOICE. ``-102`` landed two widths.
    ``index_expand_raw_width`` is ``n_groups * pool_size + pool_size - 1``
    (``index_expand.py:242-252``) and covers the columns that CARRY MEANING; ``index_expand_width``
    rounds that up to a whole number of ``KEY_CHUNK`` (``index_expand.py:255-267``) and is what the
    seam allocates and emits (``index_expand.py:527``). This function is a content reference, so it
    returns the meaningful columns and nothing else -- which is the consumer
    ``index_expand_raw_width``'s own docstring describes at ``index_expand.py:249-250``.

    The emitted width and the padding are therefore NOT this function's claims. The ragged arm makes
    them, as two claims of its own beside the content one, because they fail for different reasons: a
    wrong emitted width means the seam allocated the wrong shape, and a padding column that is not
    ``-1`` means the seam wrote meaning where it promised none. Rolling all three into one shape
    comparison is what an earlier round of this increment did, and it read a hand-typed width as the
    emitted width for a whole counted run.
    """
    rows, n_groups = (int(d) for d in pool_ids.shape)
    pool = int(pool_size)
    history_cols = n_groups * pool
    out_cols = history_cols + pool - 1
    out = torch.full((rows, out_cols), -1, dtype=torch.int32)
    for r in range(rows):
        seq = int(seq_lens[r])
        tail_start = (seq // pool) * pool
        tail_count = seq - tail_start
        for c in range(history_cols):
            pid = int(pool_ids[r, min(c // pool, n_groups - 1)])
            out[r, c] = pid * pool + (c % pool) if pid >= 0 else -1
        for t in range(pool - 1):
            if t < tail_count:
                out[r, history_cols + t] = tail_start + t
    return out


def _ref_sparse_attention(
    q_lift: torch.Tensor, c_kv: torch.Tensor, topk_indices: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Sparse attention over each row's NON-SENTINEL columns only.

    HAND-WRITTEN RATHER THAN THE SEAM'S OWN ORACLE, AND THE REASON CHANGED WHEN ``-098`` LANDED.
    It used to be that the seam's oracle was simply wrong here: it gathered with ``cache[idx[s]]``,
    and a ``-1`` does not skip a column in torch -- it WRAPS onto the last cache row and silently
    attends a real key. ``-098`` fixed that inside the oracle itself. It now builds
    ``keep = idx >= 0`` (``mla_sparse.py:1483``), clamps ``-1`` to row 0 (``mla_sparse.py:1484``), gathers
    on the clamped rows (``mla_sparse.py:1488``), masks the sentinel columns to ``-inf``
    (``mla_sparse.py:1493``), and replaces the NaN a wholly-sentinel row's softmax would otherwise
    produce (``mla_sparse.py:1496``). So the oracle and the
    kernel now AGREE about ``-1``, and the original justification for this function is GONE.

    IT IS KEPT ANYWAY, on a weaker reason stated rather than implied: a loop is a genuinely
    independent transcription of a gather-and-mask, so a slip in either shows up as a mismatch
    instead of as a shared bug. The design ruled this reference's semantics at design entry
    ``design-20260905-af`` (plan revision 215), and the lead settled the oracle question separately
    -- the module oracle masks, and the only gap was coverage, which ``-098`` carries.

    DUPLICATES ARE KEPT. A selected pool can cover the tail region, so an expanded row can name the
    same token twice, and the kernel does not de-duplicate -- its softmax normalises over the columns
    it was handed. De-duplicating here would make the reference disagree with a correct kernel.
    """
    q = q_lift.to(torch.float32)
    cache = c_kv.to(torch.float32)
    idx = topk_indices.to(torch.int64)
    seq, heads, latent = (int(d) for d in q.shape)
    out = torch.zeros(seq, heads, latent, dtype=torch.float32)
    for s in range(seq):
        live = idx[s][idx[s] >= 0]
        assert int(live.numel()) > 0, (
            f"row {s} of the expansion is all sentinel, so there is nothing to attend and the "
            f"softmax below would be undefined; the fixture must give every row a live column"
        )
        assert int(live.max()) < int(cache.shape[0]), (
            f"row {s} names cache row {int(live.max())} but the cache has "
            f"{int(cache.shape[0])}; the seam refuses this at mla_sparse.py:1263 and the "
            f"reference must not paper over it"
        )
        gathered = cache[live]
        weights = torch.softmax((q[s] @ gathered.t()) * float(softmax_scale), dim=-1)
        out[s] = weights @ gathered
    return out


# --------------------------------------------------------------------------- #
# THE ORCHESTRATION, re-written. This is the part item (1) tests.


def _raw(module, name: str) -> torch.Tensor:
    """The RAW ``[out, in]`` checkpoint leaf, read from the module by its declared attribute name.

    NOT ``_prepared_weight``: the reference transposes for itself so it does not read the
    implementation's own prepared cache. The attribute name comes from the module's own
    ``PROJECTION_PARAMETERS`` map where it has one, because that map is not uniform -- three
    indexer sites carry a ``_weight`` suffix and ``index_kpool_compress_gate`` does not
    (``model_fp8.py:2800-2805``), so guessing the suffix would work for three of four.
    """
    mapping = getattr(module, "PROJECTION_PARAMETERS", None)
    attribute = mapping[name] if mapping and name in mapping else f"{name}_weight"
    weight = getattr(module, attribute, None)
    assert weight is not None, (
        f"{attribute} is not materialised on {type(module).__name__}; the fixture must set every "
        f"site the module's own projection_widths() declares"
    )
    return weight.detach()


def _ref_project_stage(indexer, hidden: torch.Tensor, q_latent: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """``Glm5NextDSAIndexer.project_stage`` (``model_fp8.py:3088-3124``), step for step.

    Four projections and their dtypes, in the implementation's own order. The ORDER of the casts is
    the part worth mirroring exactly: the key is cast to bf16 BEFORE its LayerNorm so the norm's
    cast-back lands on bf16, and the weights stay float32 the whole way because ``dsa_score_gemm``
    admits float32 weights alone (``score_gemm.py:158``).
    """
    tokens = int(hidden.shape[0])
    heads = int(indexer.index_n_heads)
    head_dim = int(indexer.index_head_dim)
    hidden_f32 = hidden.to(torch.float32)

    query = _ref_projection(q_latent.to(torch.float32), _raw(indexer, "wq_b"))
    query = _ref_rotate(
        query.reshape(tokens * heads, head_dim).to(torch.bfloat16)
    ).reshape(tokens, heads, head_dim)

    key = _ref_layer_norm(
        _ref_projection(hidden_f32, _raw(indexer, "wk")).to(torch.bfloat16),
        indexer.k_norm_weight,
        indexer.k_norm_bias,
        indexer.KEY_NORM_EPS,
    )
    weights = _ref_projection(hidden_f32, _raw(indexer, "weights_proj")) * _ref_scale(indexer)
    gate_score = _ref_projection(
        hidden_f32, _raw(indexer, "index_kpool_compress_gate")
    ).to(torch.bfloat16)
    return query, key, weights, gate_score


def _ref_candidate_keys(pool_cache: torch.Tensor, candidates: int) -> torch.Tensor:
    """``_gather_candidates`` collapses to a PREFIX SLICE, and the collapse is proven not assumed.

    The implementation sends ``page = j // page_size`` and ``slot = j % page_size`` for
    ``j in arange(candidates)`` (``model_fp8.py:3566-3572``) and the kernel recomputes
    ``page * page_size + slot``, which is the identity on ``j``. So the gather returns
    ``pool_cache[:candidates]``. The reference states the identity as an assertion on the arithmetic
    rather than quietly slicing, because if the page arithmetic ever stops being the identity this
    line is where a reader needs to be told.
    """
    probe = torch.arange(int(candidates), dtype=torch.int64)
    page_size = PAGE_SIZE
    assert torch.equal(
        torch.div(probe, page_size, rounding_mode="floor") * page_size
        + torch.remainder(probe, page_size),
        probe,
    ), "the page arithmetic is no longer the identity map, so the prefix slice below is wrong"
    return pool_cache[: int(candidates)]


def _ref_indexer(
    indexer,
    hidden: torch.Tensor,
    q_latent: torch.Tensor,
    pool_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    candidates: int,
    trash: int,
    slot_mapping: torch.Tensor | None = None,
    tail: torch.Tensor | None = None,
    position: int | None = None,
    probe: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """``Glm5NextDSAIndexer.forward``, both legs (``model_fp8.py:3645-3689``).

    ``probe``, when given, collects the pool SCORES this call selected from. The caller uses them for
    the tie control: item (1) compares the layer's final output, so a selection decided by a tie
    would make the whole chain diverge for a reason that is not a defect.

    MUTATES ``pool_cache`` and ``tail`` exactly as the implementation does, so the caller must hand
    this function its OWN clones. A shared cache would let the two runs contaminate each other and
    the comparison would be of one run against itself.
    """
    pool = int(indexer.index_kpool)
    is_decode = tail is not None
    query, key, weights, gate_score = _ref_project_stage(indexer, hidden, q_latent)
    ape = indexer.index_kpool_compress_ape.to(torch.float32)

    if is_decode:
        # ``tail_step`` -> ``dsa_decode_tail_update`` (``model_fp8.py:3285``,
        # ``decode_tail_update.py:585-599``). ``slot_of`` is ``position % pool_size``.
        slot = int(position) % pool
        pool_key = tail[0].clone()
        pool_score = tail[1].clone()
        pool_key[slot] = key[0].to(pool_key.dtype)
        pool_score[slot] = gate_score[0].to(pool_score.dtype)
        pooled = _ref_compress_decode(pool_key, pool_score, ape) if slot == pool - 1 else None
        tail[0, slot] = key[0].to(tail.dtype)
        tail[1, slot] = gate_score[0].to(tail.dtype)
        if pooled is not None:
            pool_cache[int(position) // pool] = pooled.to(pool_cache.dtype)[0]
    else:
        # ``pool_window`` (``model_fp8.py:3206-3216``): a sliding window of ``pool`` positions
        # ending at each token, clamped at the start; a row is WRITTEN only where the slot is real
        # and the window is full. Everything else is steered to the trash row.
        tokens = int(key.shape[0])
        pos = torch.arange(tokens)
        offsets = torch.arange(pool)
        window = (pos - (pool - 1)).clamp_min(0)[:, None] + offsets[None, :]
        write_mask = (slot_mapping >= 0) & (pos >= pool - 1)
        pooled = _ref_compress_prefill(key[window], gate_score[window], ape)
        destination = torch.where(
            write_mask, slot_mapping.to(torch.int64), torch.tensor(int(trash), dtype=torch.int64)
        )
        pool_cache.index_copy_(0, destination, pooled.to(pool_cache.dtype))

    candidate_keys = _ref_candidate_keys(pool_cache, candidates)
    scores = _ref_score(query, candidate_keys, weights)
    if probe is not None:
        probe.append(scores.detach())
    pool_ids = _ref_topk(scores, int(indexer.select_k()))
    return _ref_expand(pool_ids, seq_lens, pool)


def _ref_attend(
    attention,
    hidden: torch.Tensor,
    latent_cache: torch.Tensor,
    start_position: int,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """``Glm5NextMLAAttention.attend`` (``model_fp8.py:4543-4576``).

    MUTATES ``latent_cache``, as the implementation does at ``model_fp8.py:4549``. The write happens BEFORE the
    read on purpose -- so a decode step attends to its own token -- and mirroring that order is the
    whole reason this is not written as a pure function.
    """
    tokens = int(hidden.shape[0])
    heads = int(attention.num_attention_heads)
    eps = float(attention.rms_norm_eps)
    x = hidden.to(torch.float32)

    # project_query_and_latent (:4260-4268), which routes through project_query_latent (:4219-4220).
    q_latent = _ref_projection(x, _raw(attention, "q_a_proj"))
    q_latent = _ref_latent_norm(q_latent, attention.q_a_layernorm_weight, eps)
    query = _ref_projection(q_latent, _raw(attention, "q_b_proj"))
    query = query.reshape(tokens, heads, int(query.shape[1]) // heads).to(hidden.dtype)
    kv_latent = _ref_projection(x, _raw(attention, "kv_a_proj_with_mqa"))
    kv_latent = _ref_latent_norm(kv_latent, attention.kv_a_layernorm_weight, eps).to(hidden.dtype)

    start = int(start_position)
    latent_cache[start : start + tokens, 0, :] = kv_latent.to(latent_cache.dtype)
    c_kv = latent_cache[: start + tokens, 0, :]

    w_uk, w_uv = _ref_absorb_operands(attention)
    q_lift = _ref_absorb(query, w_uk)
    attended = _ref_sparse_attention(q_lift, c_kv, topk_indices, softmax_scale)
    reduced = _ref_absorb(attended.to(hidden.dtype), w_uv)
    projected = _ref_projection(
        reduced.reshape(int(reduced.shape[0]), -1).contiguous(), _raw(attention, "o_proj")
    )
    return projected.to(reduced.dtype)


def _ref_latent_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``_latent_norm`` (``model_fp8.py:4257-4259``): an RMS norm with a gain and NO cast back."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + float(eps)) * gain.to(torch.float32)


def _ref_absorb_operands(attention) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the raw ``kv_b_proj`` into ``W_UK`` and ``W_UV`` (``model_fp8.py:4208-4215``).

    Re-derived here rather than read from ``_absorb_weight``, because the SPLIT is part of the
    chain under test: a permutation error there would move every head's output and a reference that
    reused the implementation's split could not see it.
    """
    heads = int(attention.num_attention_heads)
    nope = int(attention.qk_nope_head_dim)
    vdim = int(attention.v_head_dim)
    latent = int(attention.kv_lora_rank)
    prepared = _raw(attention, "kv_b_proj").to(torch.float32).t().contiguous()
    per_head = prepared.reshape(latent, heads, nope + vdim)
    return (
        per_head[:, :, :nope].permute(1, 2, 0).contiguous(),
        per_head[:, :, nope:].permute(1, 0, 2).contiguous(),
    )


def _ref_layer(
    layer,
    hidden: torch.Tensor,
    *,
    latent_cache: torch.Tensor,
    pool_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    start_position: int,
    softmax_scale: float,
    candidates: int,
    trash: int,
    slot_mapping: torch.Tensor | None = None,
    tail: torch.Tensor | None = None,
    position: int | None = None,
    probe: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """``Glm5NextDSALayer.forward`` (``model_fp8.py:4687-4712``).

    Note what the landed layer does NOT do, mirrored here rather than corrected: no MLP and no
    post-attention norm. Both are built and declared but unthreaded, and BOTH layer families say so
    in their own docstrings (``model_fp8.py:4673`` for this one, ``model_fp8.py:2690`` for the KDA sibling), so the omission
    is a staged design and not a gap this reference should quietly fill in.
    """
    attention = layer.attention
    residual = hidden
    normed = _ref_rms_norm(hidden, layer.input_layernorm_weight, layer.rms_norm_eps)
    q_latent = _ref_latent_norm(
        _ref_projection(normed.to(torch.float32), _raw(attention, "q_a_proj")),
        attention.q_a_layernorm_weight,
        float(attention.rms_norm_eps),
    )
    topk_indices = _ref_indexer(
        attention.indexer,
        normed,
        q_latent,
        pool_cache,
        seq_lens,
        candidates=candidates,
        trash=trash,
        slot_mapping=slot_mapping,
        tail=tail,
        position=position,
        probe=probe,
    )
    attn_out = _ref_attend(
        attention, normed, latent_cache, int(start_position), topk_indices, float(softmax_scale)
    )
    return residual + attn_out


# =========================================================================== #
# THE FIXTURE. A materialised stack at :data:`TINY_GEOMETRY`.


#: The softmax scale, DERIVED. ``q_lift`` and the gathered rows both carry the latent width, so the
#: contraction width is ``kv_lora_rank`` (``qk_rope_head_dim`` is 0 on this checkpoint). ``attend``
#: takes the scale as a caller's argument on purpose and its own comment says why -- "no block
#: registers a value for it ... deriving one here would mint a registered value this increment has no
#: authority to mint" (``model_fp8.py:4480-4482``). So this file derives one for its own run and
#: registers nothing.
SOFTMAX_SCALE = float(TINY_GEOMETRY["kv_lora_rank"] ** -0.5)


def _materialise_indexer(indexer, gen: torch.Generator) -> None:
    """Fill the indexer's seven declared leaves and prepare its four projections."""
    for name, in_features, out_features in indexer.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features**-0.5)
        setattr(
            indexer, indexer.PROJECTION_PARAMETERS[name], torch.nn.Parameter(weight)
        )
    head_dim = int(indexer.index_head_dim)
    pool = int(indexer.index_kpool)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.05
    )
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.02
    )
    # The ape is a bf16 checkpoint leaf; the caller casts it to float32 per call
    # (``model_fp8.py:3215``), so the leaf is stored in the checkpoint's dtype here rather than
    # pre-cast -- otherwise the reference would agree with a cast the implementation still has to do.
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (torch.randn(pool, head_dim, generator=gen, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    assert indexer.prepare_projection_weights() == 4, "four indexer projections must prepare"


def build_layer_stack(*, layers: int = LAYERS, seed: int = 51_051_051):
    """A stack of :class:`Glm5NextDSALayer` at the tiny geometry, every leaf materialised.

    The weight scaling mirrors the ``-042`` sibling's fixture (``test_mla_decode.py:139-143``):
    ``randn * in_features ** -0.5``, so activations stay order one through a 3-layer chain instead of
    growing and turning a tolerance comparison into a test of overflow.

    EVERY LAYER GETS ITS OWN SEED STREAM from one generator, so the three layers are genuinely
    different maps. A stack of three identical layers would let a per-layer indexing error pass.
    """
    model_fp8 = _impl()
    cfg = _tiny_text_config()
    gen = torch.Generator().manual_seed(int(seed))
    stack = []
    for layer_idx in range(int(layers)):
        layer = model_fp8.Glm5NextDSALayer(cfg, layer_idx, 1)
        layer.input_layernorm_weight = torch.nn.Parameter(
            1.0 + torch.randn(int(cfg.hidden_size), generator=gen, dtype=torch.float32) * 0.05
        )
        attention = layer.attention
        for name, in_features, out_features in attention.projection_widths():
            weight = torch.randn(
                out_features, in_features, generator=gen, dtype=torch.float32
            ) * (in_features**-0.5)
            setattr(attention, f"{name}_weight", torch.nn.Parameter(weight))
        for gain_name, width in (
            ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
            ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
        ):
            setattr(
                attention,
                gain_name,
                torch.nn.Parameter(
                    1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
                ),
            )
        assert attention.prepare_projection_weights() == 5, "five MLA projections must prepare"
        assert attention.prepare_absorb_weights() == 2, "and both absorb operands must split"
        _materialise_indexer(attention.indexer, gen)
        stack.append(layer)
    say("fixture", "layers", len(stack), "hidden", int(cfg.hidden_size), "latent",
        int(cfg.kv_lora_rank), "index_heads", int(cfg.index_n_heads))
    return stack, cfg, gen


def prefill_slot_mapping(tokens: int, pool: int) -> torch.Tensor:
    """The pool-granular slot per position: the pool's own id where a pool COMPLETES, else ``-1``.

    ``pool_window``'s write mask is ``(slot_mapping >= 0) & (pos >= pool - 1)``
    (``model_fp8.py:3209``), so a ``-1`` here is how a position says "my window is not a whole pool";
    those rows are steered to the trash row rather than dropped (``model_fp8.py:3679-3684``).
    """
    slots = torch.full((int(tokens),), -1, dtype=torch.int32)
    for p in range(int(tokens)):
        if (p + 1) % int(pool) == 0:
            slots[p] = p // int(pool)
    return slots


def case_operands(cfg, *, tokens: int = PREFILL_TOKENS):
    """Every operand both runs need, plus the two derived counts the indexer will re-derive.

    ``candidates = max_seq_len // pool`` and ``trash = pool_cache.shape[0] - 1``
    (``model_fp8.py:3523``, ``model_fp8.py:3538-3544``). Both are recomputed here from the same closed forms so
    the reference does not have to call ``_require_serviceable``.
    """
    pool = int(cfg.index_kpool)
    head_dim = int(cfg.index_head_dim)
    rows = PAGES * PAGE_SIZE
    candidates = int(tokens) // pool
    trash = rows - 1
    assert candidates > TOPK_POOLS, (
        f"{candidates} candidate pool(s) at {tokens} token(s) is not more than the "
        f"{TOPK_POOLS} selected, and the indexer refuses that (model_fp8.py:3379)"
    )
    assert candidates <= trash, (
        f"{candidates} candidates leaves no trash row above them in {rows} pool_cache row(s)"
    )
    return {
        "slot_mapping": prefill_slot_mapping(int(tokens), pool),
        # One length per ROW of the scores, and the rows are tokens -- so this is each token's own
        # context length. That makes the TAIL each token's own incomplete pool.
        "seq_lens": torch.arange(1, int(tokens) + 1, dtype=torch.int32),
        "candidates": candidates,
        "trash": trash,
        "rows": rows,
    }


def per_layer_caches(cfg, layers: int, *, tokens: int = PREFILL_TOKENS) -> list[dict]:
    """ONE set of caches PER LAYER, which is not a detail.

    Every layer of a real stack owns its own KV slice and its own pooled-key cache. A single shared
    set would let layer 1 overwrite layer 0's latents, and the whole 3-layer chain would then be
    measuring the last layer twice. That defect passes a 1-layer arm silently, which is exactly why
    it is worth stating here rather than trusting the loop below to be read carefully.
    """
    rows = PAGES * PAGE_SIZE
    head_dim = int(cfg.index_head_dim)
    pool = int(cfg.index_kpool)
    return [
        {
            "pool_cache": torch.zeros(rows, head_dim, dtype=torch.bfloat16),
            "latent_cache": torch.zeros(
                int(tokens) + DECODE_STEPS, 1, TINY_HEAD_SIZE, dtype=torch.float32
            ),
            "tail": torch.zeros(2, pool, head_dim, dtype=torch.bfloat16),
        }
        for _ in range(int(layers))
    ]


def report_close(label: str, got: torch.Tensor, reference: torch.Tensor) -> None:
    """Print the achieved error BEFORE asserting, so a failure arrives with its own measurement.

    The margin is printed too. A pass that sits a hair under the tolerance and a pass with three
    orders of magnitude of headroom are different facts, and only one of them survives a weight
    reseed -- so the transcript records which one this run got.
    """
    assert got.shape == reference.shape, f"{label}: {tuple(got.shape)} vs {tuple(reference.shape)}"
    diff = (got.float() - reference.float()).abs()
    scale = reference.float().abs()
    max_abs = float(diff.max())
    allowed = float((ATOL + RTOL * scale).max())
    say(label, "max_abs_diff", f"{max_abs:.3e}", "allowed_at_worst", f"{allowed:.3e}",
        "ref_absmax", f"{float(scale.max()):.3e}")
    torch.testing.assert_close(got.float(), reference.float(), rtol=RTOL, atol=ATOL)


#: The floor a selection gap must clear. UNCHANGED in value at the round-2 repair; it is named here
#: only so the two call paths below cannot drift to two different numbers. Two definitions of one
#: control's arithmetic is the defect class this file keeps finding elsewhere.
TIE_FLOOR = 1e-4


def kth_place_gap(scores: torch.Tensor, k: int) -> float:
    """The smallest gap at the k-th place, MEASURED AND RETURNED rather than asserted on.

    Split out at the round-2 repair so a caller with several labels can read every one of them
    before any of them is allowed to abort. Round 1 asserted inside the loop, stopped at the ragged
    arm's request 0, and left request 1 unmeasured -- so the fixture's admissibility was only half
    known and the transcript could not say whether the near-tie was one row's bad luck or the whole
    draw's. The arithmetic is unchanged.
    """
    top = torch.topk(scores.float(), int(k) + 1, dim=-1).values
    return float((top[:, int(k) - 1] - top[:, int(k)]).min())


def say_and_check_gaps(gaps: list[tuple[str, float]]) -> None:
    """Print EVERY label's gap, then assert over all of them at once.

    The printing comes first on purpose: a reading that only reaches the transcript when it passes
    is not a reading. With this order a failing run still shows every label's number, so a reader
    can see at a glance whether one row was unlucky or the seed is bad everywhere.
    """
    for label, gap in gaps:
        say(label, "kth_place_gap_min", f"{gap:.3e}")
    bad = [(label, gap) for label, gap in gaps if not gap > TIE_FLOOR]
    assert not bad, (
        f"{', '.join(label for label, _ in bad)}: the smallest gap at the k-th place is "
        f"{', '.join(f'{gap:.3e}' for _, gap in bad)}, so at least one row's selection is "
        f"effectively a tie and this case cannot tell a tie-break difference from a defect. "
        f"Reseed the fixture rather than loosening the comparison"
    )


def assert_selection_is_not_a_tie(scores: torch.Tensor, k: int, label: str) -> None:
    """No row may be decided by a tie at the k-th place.

    WHY THIS CONTROL EXISTS. Item (1) compares the layer's FINAL output, and the selection sits in
    the middle of that chain. If the kernel and the reference broke a tie differently they would
    attend different cache rows and the numeric comparison would fail for a reason that is not a
    defect -- or, worse, a real defect could be excused as a tie. ``torch.topk``'s tie order is not
    contracted to match the kernel's, so the fixture must make ties impossible rather than hope.

    The single-label form, kept because run 1 reads one label per layer per phase and each of those
    is its own reading. It now routes through the shared reader above so there is ONE definition of
    the gap arithmetic and ONE floor.
    """
    say_and_check_gaps([(label, kth_place_gap(scores, int(k)))])


# =========================================================================== #
# RUN 1 OF 2 -- THE LAYER CASE. Acceptance item (1) and route-predicate part 1.


@pytest.mark.parametrize("layers", [LAYERS, CONTROL_LAYERS])
def test_run_1_a_dsa_stack_matches_the_torch_reference_and_moves_every_seam(
    monkeypatch: pytest.MonkeyPatch, layers: int
) -> None:
    """Item (1): a DSA stack matches the reference at ``rtol=5e-3``/``atol=1e-5``.

    THE 1-LAYER ARM IS A CONTROL, not a second declared case. Item (1)'s declared case is the
    3-layer one; the 1-layer arm exists so that a per-layer indexing error -- reading layer 0's
    weights on every layer, say -- cannot pass both arms, because the two arms' declared counter
    totals differ by a factor of three.

    THE ORDER OF OPERATIONS IS LOAD-BEARING AND IS NOT AN ACCIDENT OF WRITING:

      1. build the operands once,
      2. CLONE every mutable cache and run the REFERENCE on the clones,
      3. reset the counters, and assert the reset landed,
      4. run the implementation on the originals,
      5. read the counters, then compare the numbers.

    Steps 2 and 3 are in that order because the caches are mutated in place by BOTH sides -- the
    pool cache, the latent cache and the tail ring all are -- so sharing them would compare a run
    against itself. And the reset sits AFTER the reference because a reference is allowed to move a
    counter; what the route predicate measures is the IMPLEMENTATION's dispatches alone.

    THE PROJECTION COUNTER IS THE ONE EXCEPTION TO STEP 3, and it is deliberate rather than an
    inconsistency. It is reset ONCE before the phase loop instead of per phase, because its per-case
    total is the second instrument that the per-call probe is checked against, and a reset inside the
    loop would leave nothing to check. That is only sound if the reference dispatches nothing, so
    each leg READS the counter after its own reference and asserts it did not move -- which turns the
    thing step 3 ASSUMES for the seven families into a measurement for the eighth.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=int(layers))
    ops = case_operands(cfg)
    pool = int(cfg.index_kpool)
    hidden = torch.randn(
        PREFILL_TOKENS, int(cfg.hidden_size),
        generator=torch.Generator().manual_seed(9_051_001), dtype=torch.float32,
    )
    # TWO INDEPENDENT SETS, one per side, each with one cache set PER LAYER. The reference carries
    # its OWN state from prefill into decode rather than cloning the implementation's -- otherwise
    # the decode comparison would run both sides against a cache only one of them computed, and a
    # wrong prefill cache write would make the decode leg agree with itself.
    impl_caches = per_layer_caches(cfg, int(layers))
    ref_caches = per_layer_caches(cfg, int(layers))

    spy = SeamSpy()
    # THE PROJECTION COUNTER IS RESET ONCE FOR THE WHOLE CASE, not once per phase, and that is the
    # difference between one reading and two instruments. The probe below reads each indexer call's
    # DELTA; this running total reads the case; after the loop the two are checked against each
    # other and against the closed form. A reset inside the loop would leave only per-phase numbers
    # and nothing to cross-check them with. The seven family counters keep their per-phase reset --
    # their declared tables are per phase.
    projection_probe = IndexerProjectionProbe()
    reset_projection_counter()
    assert read_projection_counter() == (0, 0), (
        "the projection counter did not reset to zero, so every reading below would be partly some "
        "earlier test's dispatches"
    )
    for phase in PHASES:
        if phase == "prefill":
            step_hidden = hidden
            start = 0
            kwargs = {"slot_mapping": ops["slot_mapping"], "seq_lens": ops["seq_lens"]}
        else:
            step_hidden = torch.randn(
                1, int(cfg.hidden_size),
                generator=torch.Generator().manual_seed(9_051_002), dtype=torch.float32,
            )
            start = PREFILL_TOKENS
            kwargs = {
                "position": PREFILL_TOKENS,
                "seq_lens": torch.tensor([PREFILL_TOKENS + 1], dtype=torch.int32),
            }

        # The running projection total at the START of this phase, so both the reference check and
        # the phase delta below are DIFFERENCES rather than absolute numbers that would each have to
        # know what the previous phase left behind.
        projection_mark = read_projection_counter()

        # (2) THE REFERENCE, on its own per-layer caches.
        candidates = (PREFILL_TOKENS if phase == "prefill" else PREFILL_TOKENS + 1) // pool
        ref_hidden = step_hidden
        probe: list[torch.Tensor] = []
        for layer, caches in zip(stack, ref_caches):
            ref_hidden = _ref_layer(
                layer,
                ref_hidden,
                latent_cache=caches["latent_cache"],
                pool_cache=caches["pool_cache"],
                seq_lens=kwargs["seq_lens"],
                start_position=start,
                softmax_scale=SOFTMAX_SCALE,
                candidates=candidates,
                trash=ops["trash"],
                slot_mapping=kwargs.get("slot_mapping"),
                tail=None if phase == "prefill" else caches["tail"],
                position=kwargs.get("position"),
                probe=probe,
            )
        reference = ref_hidden

        # THE TIE CONTROL, on every layer's own selection. Run BEFORE the comparison so a tie is
        # reported as a fixture problem rather than surfacing as a numeric failure downstream.
        assert len(probe) == int(layers), (
            f"the probe collected {len(probe)} score tensors for {layers} layer(s); every layer "
            f"selects once per phase, so a short count means a layer was skipped"
        )
        for idx, scores in enumerate(probe):
            assert_selection_is_not_a_tie(
                scores, int(stack[idx].attention.indexer.select_k()), f"{phase}-layer{idx}"
            )

        # THE REFERENCE MOVED NO PROJECTION COUNTER, read rather than assumed. The seven family
        # counters are reset AFTER the reference precisely because a reference is ALLOWED to move
        # one; the projection counter is deliberately NOT reset here, because the case total has to
        # survive both phases. That is only sound if the reference is torch-only, so it is measured
        # rather than trusted: the reference projects through `_ref_projection`, a plain matmul in
        # this file, and reaches no seam. If it ever did, the case total would be part reference and
        # the closed form below would read high for a reason no assertion could name.
        after_reference = read_projection_counter()
        say(phase, "projection_after_reference", after_reference, "at_phase_start", projection_mark)
        assert after_reference == projection_mark, (
            f"the torch reference moved the projection counter from {projection_mark} to "
            f"{after_reference} on the {phase} leg. The reference and the implementation have to be "
            f"computed by different means, and a reference that dispatches the seam under test is "
            f"comparing the seam against itself"
        )

        # (3) RESET, and prove the reset landed before anything is measured against it.
        reset_all_counters()
        after_reset = read_all_counters()
        assert all(v == (0, 0) for v in after_reset.values()), (
            f"the counters did not reset to zero: {after_reset}. Every reading below would then be "
            f"partly the reference's dispatches, which is exactly the confusion this order avoids"
        )

        # (4) THE IMPLEMENTATION, on the originals, with the spy installed.
        spy_here = SeamSpy()
        spy_here.install(monkeypatch)
        # The projection probe is installed in the SAME window as the spy and is undone by the same
        # `monkeypatch.undo()`, so it can only ever see the implementation's calls. The probe OBJECT
        # outlives the loop, so its recorded calls accumulate over both phases.
        projection_probe.install(monkeypatch)
        got = step_hidden
        for layer, caches in zip(stack, impl_caches):
            got = layer.forward(
                got,
                latent_cache=caches["latent_cache"],
                pool_cache=caches["pool_cache"],
                seq_lens=kwargs["seq_lens"],
                start_position=start,
                softmax_scale=SOFTMAX_SCALE,
                max_seq_len=PREFILL_TOKENS if phase == "prefill" else PREFILL_TOKENS + 1,
                page_size=PAGE_SIZE,
                slot_mapping=kwargs.get("slot_mapping"),
                tail=None if phase == "prefill" else caches["tail"],
                position=kwargs.get("position"),
            )
        monkeypatch.undo()
        # BOTH of the spy's lists are carried across, and this line is the repair. Round 1 merged
        # ``calls`` alone, so ``first_arg_dtypes`` never left the per-phase spy: the rider check below
        # then read an empty set off the OUTER spy, asserted a subset of nothing, printed "not called"
        # and passed -- while the per-entry counts in the same transcript showed those seams called
        # three times each. Two lists filled by one wrapper have to be merged by one step.
        spy.calls.extend(spy_here.calls)
        spy.first_arg_dtypes.extend(spy_here.first_arg_dtypes)

        # (5) THE READINGS, then the comparison.
        readings = read_all_counters()
        for family in FAMILIES:
            say(phase, "counter", family, readings[family])
        assert all(v[1] == 0 for v in readings.values()), (
            f"a torch fallback ran on the {phase} leg: {readings}. Every DSA seam here is "
            f"kernel-class (P13), so a fallback is a route failure and not a slow path"
        )
        # THE PHASE'S PROJECTION DELTA, against the closed form for this many layers and one phase.
        phase_reading = read_projection_counter()
        phase_nki = phase_reading[0] - projection_mark[0]
        phase_fallback = phase_reading[1] - projection_mark[1]
        want_phase = PROJECTION_PER_LAYER_PER_PHASE * int(layers)
        say(phase, "projection_phase_delta", phase_nki, "want", want_phase,
            "torch_fallback", phase_fallback, "running", phase_reading)
        assert phase_nki == want_phase, (
            f"the {phase} leg dispatched {phase_nki} projections where {layers} layer(s) owe "
            f"{want_phase}, at {PROJECTION_PER_LAYER_PER_PHASE} per layer per phase. The terms and "
            f"their source lines are printed after the loop; a miss is a finding about the closed "
            f"form or about the layer, and not a number to edit"
        )
        assert phase_fallback == 0, (
            f"the {phase} leg recorded {phase_fallback} projection torch fallbacks, which no code "
            f"path can produce today (mla_projections.py:151-155), so the substrate declaration for "
            f"this increment must be re-derived rather than this number relaxed"
        )
        spy_here.report(f"{phase}-L{layers}")
        report_close(f"item-1-{phase}-L{layers}", got, reference)

    # PART 1 OF THE ROUTE PREDICATE, over this case: six of the seven families move here and the
    # pack pair is the DECLARED ZERO entry `aa` ruled. The seventh is run 2's.
    per_family = spy.per_family()
    say("part1", "families_moved", sum(1 for v in per_family.values() if v), "of", len(FAMILIES))
    for family in FAMILIES:
        say("part1", "family", family, per_family[family])
    expected = declared_family_totals(int(layers))
    assert per_family == expected, (
        f"the spy's per-family totals {per_family} do not match the declared table {expected}. "
        f"The table is the prediction and the spy is the measurement, so a mismatch is a finding "
        f"about one of them and not a number to edit"
    )

    # THE PROJECTION SUBSTRATE READING FOR THIS CASE, WITH ITS TERMS -- the reading this file did not
    # take for twelve attempts. `mla_projection` is the EIGHTH counter family and the seven-family
    # walk above structurally cannot reach it, because it lives under `functional.attention` while
    # `FAMILIES` is built from entry points under `functional.dsa`. Review item B86.
    label = f"item-1-L{layers}"
    want_total = projection_closed_form(
        label, DECLARED_PROJECTION_TERMS, layers=int(layers), phases=len(PHASES)
    )
    total_nki, total_fallback = read_projection_counter()
    say(label, "projection_total", total_nki, "want", want_total, "torch_fallback", total_fallback)
    assert total_nki == want_total, (
        f"the case dispatched {total_nki} projections where the closed form owes {want_total}. Every "
        f"term and its source lines are printed above, so a mismatch names itself: a short count is "
        f"a call served by mla_projection_torch_oracle or not made at all, and a long one is a term "
        f"the closed form does not know about -- Glm5NextMLAAttention.project_qkv holds four more "
        f"dispatch sites and is dead on this path, so wiring it in would read here first"
    )
    assert total_fallback == 0, f"the case recorded {total_fallback} projection torch fallbacks"

    # TWO INSTRUMENTS OVER THE SAME EVENTS, AND THE RESIDUAL IS THE CHECK. The probe read each
    # indexer call's delta; the counter read the whole case. So the case total MINUS the indexer's
    # share must be exactly the non-indexer terms. A probe that double-counted, or an indexer that
    # projected three times while something else projected five, agrees with neither.
    projection_probe.check(label, want_calls=int(layers) * len(PHASES))
    indexer_share = projection_probe.dispatched
    want_indexer = PROJECTION_PER_INDEXER_CALL * int(layers) * len(PHASES)
    want_rest = PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE * int(layers) * len(PHASES)
    say(label, "projection_indexer_share", indexer_share, "want", want_indexer,
        "residual", total_nki - indexer_share, "want", want_rest)
    assert indexer_share == want_indexer, (
        f"the probe read {indexer_share} projections across the indexer calls where "
        f"{int(layers) * len(PHASES)} calls at {PROJECTION_PER_INDEXER_CALL} each owe {want_indexer}"
    )
    assert total_nki - indexer_share == want_rest, (
        f"the case total {total_nki} less the indexer's {indexer_share} leaves "
        f"{total_nki - indexer_share} for the layer's own projections, where the non-indexer terms "
        f"owe {want_rest}. The two instruments read the same events by different means, so this is "
        f"where one of them being wrong shows up"
    )

    # RIDER B71-N3, read on the PRODUCTION PATH rather than reconstructed. The standalone rider test
    # reads what `project_stage` returns; this reads what the seams were actually handed while the
    # layer ran, which is the thing the finding is about. `torch_fallback == 0` above already proves
    # the kernels served these calls, so the two readings together say the bf16 reached the gate AND
    # the gate admitted it.
    for entry in ("dsa_kpool_hadamard", "dsa_hadamard128", "dsa_decode_tail_update"):
        seen = spy.dtypes_for(entry)
        say("B71-N3", "production dtype", entry, sorted(str(d) for d in seen) or "not called")
        # NON-EMPTINESS FIRST, as its own claim with its own message. A subset assertion over a MEASURED
        # set also asserts the set is non-empty, or it is not a measurement: `set() <= {bfloat16}` is
        # True, so round 1 passed this rider while reading nothing at all. The two claims are separate
        # because they fail for different reasons -- an empty set means the INSTRUMENT did not record,
        # and a wrong dtype means the SEAM was handed the wrong thing.
        assert seen, (
            f"{entry} has no recorded production dtype, so the dtype check below would assert a subset "
            f"of nothing and pass while measuring nothing. The per-entry counts printed above say "
            f"whether the seam was called; if it was, the spy's dtype log did not reach this spy"
        )
        assert seen == {torch.bfloat16}, (
            f"{entry} was handed {sorted(str(d) for d in seen)} on the production path; anything but "
            f"bf16 fails the seam's dtype gate and takes the torch oracle while every shape still "
            f"checks out (kpool_hadamard.py:147, :496)"
        )


# =========================================================================== #
# RUN 2 OF 2 -- THE RAGGED DECODE ARM. The seventh counter family.


def test_run_2_the_ragged_arm_packs_and_each_request_matches_itself_run_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seventh family, and a correctness reading that is EXACT rather than a tolerance.

    WHY THIS ARM EXISTS. ``ragged_pack`` cannot move on the layer case: a batch of one is uniform,
    ``attend`` refuses any other batch size, and upstream guards its whole pack region with
    ``requires_padding`` which this fork has no producer for. So the arm reaches the seam the way
    entry ``design-20260905-aa`` ruled -- by calling ``Glm5NextDSAIndexer.forward_ragged`` directly
    on a test-constructed non-uniform batch. The arm is honest about being coverage of a landed seam
    along a path production does not currently take; that disclosure is in
    ``watch-item-051-trace-bound.md``.

    THE CORRECTNESS READING IS BIT-EXACT, and the implementation's own docstring is why. It claims
    the projections are row-wise and therefore "projecting the padded grid and then packing gives
    bit-for-bit what packing and then projecting would" (``model_fp8.py:3824-3827``). This test
    takes that claim at its word and compares INDICES for equality -- no tolerance -- because the
    output is an int32 index tensor and a tolerance on an index would hide an off-by-one.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    hidden_size = int(cfg.hidden_size)
    q_lora = int(cfg.q_lora_rank)

    # A NON-UNIFORM batch: the arm refuses a uniform one by name (``model_fp8.py:3809-3816``).
    lengths = [PREFILL_TOKENS // 2, PREFILL_TOKENS]
    max_len = max(lengths)
    tokens = sum(lengths)
    assert len(set(lengths)) > 1, "a uniform batch is refused by the arm, and rightly"

    # THE SEED, RESEEDED AT ROUND 2 UNDER THE LEAD'S RULING (DECISIONS §78), and the rule is stated
    # rather than the number chosen to make the suite quiet. This file allocates seeds as
    # ``9_051_NNN``: run 1 takes 001 and 002 for its two tensors, rider B67 takes 067, rider B71
    # takes 071. The arm was written with 9_051_002, which is a COLLISION -- it duplicated run 1's
    # ``q_latent`` seed, so five seed sites held only four distinct values and the arm was never an
    # independent draw at all. The next value in the file's own sequence is 003, which both follows
    # the rule and removes the duplication. Round 1's 002 draw put request 0's 2nd and 3rd scores
    # 3.296e-05 apart, under the TIE_FLOOR, so the fixture could not tell a tie-break difference
    # from a defect and refused to measure -- exactly as designed. ONE try: if 003 also near-ties,
    # that is a structural finding for the lead and not a third seed.
    gen = torch.Generator().manual_seed(9_051_003)
    hidden = torch.randn(len(lengths), max_len, hidden_size, generator=gen, dtype=torch.float32)
    q_latent = torch.randn(len(lengths), max_len, q_lora, generator=gen, dtype=torch.float32)

    # The arm only READS the pool cache -- it writes no pool -- so the cache is seeded directly
    # rather than by running a prefill first. That keeps this run one case and not two.
    rows = PAGES * PAGE_SIZE
    pool_cache = (
        torch.randn(rows, int(cfg.index_head_dim), generator=gen, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    max_seq_len = PREFILL_TOKENS
    candidates = max_seq_len // pool
    seq_lens = torch.cat(
        [torch.arange(1, n + 1, dtype=torch.int32) for n in lengths]
    )
    assert int(seq_lens.shape[0]) == tokens, "one seq_len per PACKED row (model_fp8.py:3673)"

    # THE REFERENCE: each request's own valid rows, projected and selected with NO pack at all,
    # concatenated in the pack's own order. This is "the same request run alone" as the design
    # names it, and it is what makes the commutation claim falsifiable.
    # Scored first for EVERY request, then read, then expanded. Round 1 asserted inside one loop and
    # aborted at request 0, so request 1's gap never reached the transcript and nobody could tell
    # whether one row was unlucky or the whole draw was bad. The three passes cost one extra list.
    # Reset the projection counter BEFORE the reference, so the next reading says whether the
    # reference dispatched the seam under test. The arm's reference calls `_ref_project_stage` once
    # per request, which is this file's own torch matmul and should reach no seam at all.
    reset_projection_counter()
    projection_probe = IndexerProjectionProbe()

    select_k = int(indexer.select_k())
    scored: list[tuple[int, torch.Tensor]] = []
    for b, n in enumerate(lengths):
        q_own, _k, w_own, _g = _ref_project_stage(
            indexer, hidden[b, :n], q_latent[b, :n]
        )
        scored.append((n, _ref_score(q_own, _ref_candidate_keys(pool_cache, candidates), w_own)))

    say_and_check_gaps(
        [
            (f"arm-request-{b}", kth_place_gap(scores, select_k))
            for b, (_n, scores) in enumerate(scored)
        ]
    )

    ref_rows = []
    offset = 0
    for n, scores in scored:
        pool_ids = _ref_topk(scores, select_k)
        ref_rows.append(_ref_expand(pool_ids, seq_lens[offset : offset + n], pool))
        offset += n
    reference = torch.cat(ref_rows, dim=0)

    # THE REFERENCE DISPATCHED NOTHING, read rather than assumed -- the same claim run 1 makes on
    # each of its legs, for the same reason: the two sides have to be computed by different means.
    after_reference = read_projection_counter()
    say("arm", "projection_after_reference", after_reference, "want", (0, 0))
    assert after_reference == (0, 0), (
        f"the arm's torch reference moved the projection counter to {after_reference}; the reference "
        f"projects through `_ref_projection` in this file and must reach no seam"
    )

    reset_all_counters()
    reset_projection_counter()
    after_reset = read_all_counters()
    assert all(v == (0, 0) for v in after_reset.values()), f"reset did not land: {after_reset}"
    assert read_projection_counter() == (0, 0), "the projection reset did not land"

    spy = SeamSpy()
    spy.install(monkeypatch)
    projection_probe.install(monkeypatch)
    got = indexer.forward_ragged(
        hidden, q_latent, pool_cache, seq_lens, lengths,
        max_seq_len=max_seq_len, page_size=PAGE_SIZE,
    )
    monkeypatch.undo()

    readings = read_all_counters()
    for family in FAMILIES:
        say("arm", "counter", family, readings[family])
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the ragged arm: {readings}"
    )
    assert readings["ragged_pack"][0] > 0, (
        "the ragged arm did not move the pack family, which is the ONLY reason this second run "
        "exists -- part 1 reads 7/7 over the union of the two runs and this is the seventh"
    )

    # THE ARM'S PROJECTION READING, WITH ITS TERMS. One term, because the arm calls the indexer
    # directly, and the term is the indexer's four. The probe's per-call reading and this total are
    # the SAME number here, and that is not a duplication: they arrive by different means, so a
    # probe that missed the call reads zero calls while the total still reads four.
    want_arm = projection_closed_form(
        "arm", DECLARED_PROJECTION_TERMS_RAGGED_ARM, layers=1, phases=1
    )
    arm_nki, arm_fallback = read_projection_counter()
    say("arm", "projection_total", arm_nki, "want", want_arm, "torch_fallback", arm_fallback)
    assert arm_nki == want_arm, (
        f"the arm dispatched {arm_nki} projections where it owes {want_arm}. The arm projects the "
        f"PADDED GRID ONCE rather than once per request -- that commutation is the claim this run "
        f"exists to test (model_fp8.py:3824-3827) -- so a reading of "
        f"{PROJECTION_PER_INDEXER_CALL * len(lengths)} would mean it projected per request and the "
        f"bit-exact comparison below is passing for the wrong reason"
    )
    assert arm_fallback == 0, f"the arm recorded {arm_fallback} projection torch fallbacks"
    projection_probe.check("arm", want_calls=1)
    assert projection_probe.dispatched == arm_nki, (
        f"the probe read {projection_probe.dispatched} projections across the arm's indexer calls "
        f"where the counter read {arm_nki} for the case; the arm makes ONE indexer call and nothing "
        f"else projects, so the two instruments must agree exactly"
    )
    spy.report("arm")

    # THE EXPANSION IS THREE CLAIMS, NOT ONE. `-051` round 2 asserted a single shape equality against a
    # reference that had the raw width TYPED into it, so a claim about the EMITTED width was settled by a
    # number nobody measured. The three below fail for three different reasons and say so separately:
    # the emitted width is the seam's own allocation rule, the meaningful columns are the content, and
    # the padding is a promise that those columns carry nothing.
    expand_mod = _seam_module("index_expand")
    raw_want = int(expand_mod.index_expand_raw_width(select_k, pool))
    emitted_want = int(expand_mod.index_expand_width(select_k, pool))
    # KEY_CHUNK read from the module that DEFINES it (mla_sparse.py:111), which is where index_expand
    # imports it from too (index_expand.py:152). Reading it here is what keeps claim 1 from resting on
    # index_expand_width alone: if that helper were wrong, the multiple-of-KEY_CHUNK arm and the
    # not-below-raw arm would still catch a width that cannot be what the sparse kernel admits.
    key_chunk = int(importlib.import_module("vllm_neuron.functional.attention.mla_sparse").KEY_CHUNK)
    raw_got = int(reference.shape[1])
    say("arm", "raw_width", raw_got, "from_helper", raw_want, "key_chunk", key_chunk)
    assert raw_got == raw_want, (
        f"the reference emitted {raw_got} columns where index_expand_raw_width({select_k}, {pool}) "
        f"says {raw_want}; the reference's own width is a reading before it is a yardstick"
    )

    # CLAIM 1, THE EMITTED WIDTH.
    say("arm", "emitted_width", int(got.shape[1]), "want", emitted_want, "rows", int(got.shape[0]))
    assert int(got.shape[0]) == int(reference.shape[0]), (
        f"the arm returned {int(got.shape[0])} rows against the reference's "
        f"{int(reference.shape[0])}; the row count is the packed token count and must agree exactly"
    )
    assert int(got.shape[1]) == emitted_want, (
        f"the arm emitted {int(got.shape[1])} columns where index_expand_width({select_k}, {pool}) "
        f"says {emitted_want} (index_expand.py:255-267, allocated at :527)"
    )
    assert int(got.shape[1]) % key_chunk == 0, (
        f"the emitted width {int(got.shape[1])} is not a whole multiple of KEY_CHUNK={key_chunk}, so "
        f"mla_sparse_attention would refuse it (mla_sparse.py:1162-1168) whatever index_expand_width "
        f"returns -- this arm holds even if that helper is wrong"
    )
    assert int(got.shape[1]) >= raw_want, (
        f"the emitted width {int(got.shape[1])} is below the raw width {raw_want}, so meaningful "
        f"columns were dropped by the allocation itself"
    )

    # CLAIM 2, THE MEANINGFUL COLUMNS.
    head = got[:, :raw_got]
    mismatches = int((head.to(torch.int64) != reference.to(torch.int64)).sum())
    say("arm", "content_mismatches", mismatches, "of", int(reference.numel()), "cols", raw_got)
    assert mismatches == 0, (
        f"{mismatches} of {int(reference.numel())} meaningful expanded indices differ over the first "
        f"{raw_got} columns. The implementation claims the pack COMMUTES with the row-wise "
        f"projections bit-for-bit (model_fp8.py:3679-3682); this is that claim failing, not a "
        f"tolerance to widen"
    )

    # CLAIM 3, THE PADDING. Its own claim because a padding column carrying a real token index is a
    # DIFFERENT fault from a wrong content column: mla_sparse drops `-1` and attends anything else
    # (mla_sparse.py:1483-1496), so meaning written here would be silently attended.
    tail = got[:, raw_got:]
    pad_cols = int(tail.shape[1])
    bad_pad = int((tail.to(torch.int64) != -1).sum())
    say("arm", "padding_columns", pad_cols, "non_sentinel", bad_pad)
    assert pad_cols == emitted_want - raw_want, (
        f"the padding region measured {pad_cols} columns where the two widths say "
        f"{emitted_want - raw_want}; the region this claim is about has to be the region it measures"
    )
    assert pad_cols > 0, (
        f"there is no padding region at this geometry ({emitted_want} emitted, {raw_want} raw), so "
        f"the sentinel claim below would be a statement about an empty set. A subset or all-equal "
        f"assertion over a measured set also asserts the set is non-empty, or it is not a measurement"
    )
    assert bad_pad == 0, (
        f"{bad_pad} of {pad_cols * int(got.shape[0])} padding entries are not -1. mla_sparse keeps "
        f"every column that is not -1 (mla_sparse.py:1483-1484) and attends it, so a real token "
        f"index written past the raw width is extra attention with no other symptom"
    )

    per_family = spy.per_family()
    expected = declared_family_totals(1, DECLARED_PER_LAYER_RAGGED_ARM)
    assert per_family == expected, (
        f"the arm's per-family totals {per_family} do not match its declared table {expected}"
    )

    # RIDER B67-N4 on the NKI path. The standalone rider test reads the pack's dtype behaviour through
    # torch; this reads it here, where `ragged_pack` moved a counter and `torch_fallback` is 0 above --
    # so the dtype the seam was handed is the dtype the KERNEL admitted, which torch alone cannot say.
    seen = spy.dtypes_for("dsa_ragged_pack")
    say("B67-N4", "production dtype", sorted(str(d) for d in seen) or "not called")
    assert seen == {torch.bfloat16}, (
        f"the pack was handed {sorted(str(d) for d in seen)}; the module admits bf16 alone "
        f"(ragged_pack.py:121) and any other dtype routes to the torch path with no other symptom"
    )
