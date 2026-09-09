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


# THE F10 LENGTH REFUSAL RETIRED HERE AT ``inc-glm53f-099``, and its two items retired with it.
#
# What stood here was ``test_a_sequence_too_short_to_select_is_refused_by_name_and_routed_to_099``,
# parametrised over both entry points, asserting that below the strict selection bound the indexer
# REFUSES by name rather than clamping ``k``. The refusal named ``inc-glm53f-099`` as the route it
# was standing in for. That route is now built, so there is nothing left to refuse and no test can
# assert a refusal that no longer exists. The two items are replaced by the three SERVED-regime
# items in the bypass section at the end of this file, and the guarantee they protected --
# ``dsa_topk_select`` never sees ``k == width`` -- is protected there instead, by reading
# ``topk_select``'s counter as a zero on the bypass case.
#
# ``_refuses`` STAYS. It has a third caller, the dial item above, which is four items of
# ``inc-glm53f-051``'s; retiring the helper would break them. Ruled at DECISIONS §97 (i) on this
# seat's finding F1.


def test_the_two_entry_points_refuse_in_the_SAME_order() -> None:
    """A config that breaks BOTH preconditions must report the DIAL, on both entry points.

    This is the extraction's real contract. ``require_dials()`` runs before
    ``_require_serviceable()``, so a config that breaks both must name the dial; if one entry
    point reported the other fault instead, the two paths would have drifted and a reader
    could not predict which refusal a bad config produces.

    RE-POINTED AT THE TRASH-ROW REFUSAL AT ``inc-glm53f-099``, ruled at DECISIONS §97 (i) on
    this seat's finding F2. The second fault used to be the F10 length, and that refusal is
    gone -- so ``"inc-glm53f-099" not in message`` had become TRIVIALLY true and the item
    would have gone on looking like an ordering test while asserting nothing. The trash-row
    refusal (``model_fp8.py:3547-3552``) is the live replacement: it also fires inside
    ``_require_serviceable``, it also fires before any dispatch, and its message cannot be
    confused with the dial's.

    CASE B IS CASE A's FIRING CONTROL, which is why one item carries two cases. Case A's
    claim is an ABSENCE -- the trash-row wording must not appear -- and an absence is only a
    reading if the same operands can produce it. Case B keeps the same undersized
    ``pool_cache`` and repairs only the dial, and the wording appears. Without case B this
    item would pass on an indexer that had no trash-row refusal at all.
    """
    # ABOVE the bypass bound, so the regime still selects and the trash row is still addressed:
    # this item is about refusal ORDER, not about the short regime.
    long_enough = PREFILL_TOKENS
    assert long_enough // POOL_SIZE > TOPK_POOLS, (
        "the ordering case must sit above the strict selection bound, or the indexer would take "
        "inc-glm53f-099's bypass and never reach the trash-row check at all"
    )
    # Fewer pool_cache rows than there are addressable candidate pools, so `candidates > trash`.
    starved_rows = 2
    assert long_enough // POOL_SIZE > starved_rows - 1, (
        f"{starved_rows} pool_cache row(s) must leave no trash row above the "
        f"{long_enough // POOL_SIZE} candidate pool(s), or case B refuses nothing"
    )
    TRASH_WORDING = "leaves no trash row above"

    for method in ENTRY_METHODS:
        # CASE A: both faults present. The DIAL must be the one reported.
        both_broken = _bare_indexer(index_kpool_compress=False)
        message = _refuses(
            method, both_broken, max_seq_len=long_enough, pool_rows=starved_rows
        )
        say("order", method, "both_broken", message.split(";")[0])
        assert "index_kpool_compress" in message
        assert TRASH_WORDING not in message, (
            "the dial must be reported first: it is the cheaper, earlier and more likely "
            "operator error, and reporting the pool_cache geometry would send a reader to "
            "resize a cache that is not the problem"
        )

        # CASE B, THE CONTROL: the dial repaired, the same starved cache. The other refusal
        # must now appear, or case A's absence measured nothing.
        dial_ok = _bare_indexer()
        control = _refuses(
            method, dial_ok, max_seq_len=long_enough, pool_rows=starved_rows
        )
        say("order", method, "control_dial_ok", control.split(";")[0])
        assert TRASH_WORDING in control, (
            f"the trash-row refusal did not fire on {starved_rows} row(s) with the dial "
            f"repaired, so case A's absence of that wording is not a reading. Got: {control}"
        )
        assert "index_kpool_compress" not in control


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
    return (
        _ref_score_per_head(q, k).clamp(min=0.0) * weights.float().unsqueeze(-1)
    ).sum(dim=1)


def _ref_score_per_head(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The per-head scores BEFORE the ReLU and before the head weights.

    Split out at `-103`'s r7 read so the tie diagnostic can print the four per-head numbers behind a
    tied row without a SECOND spelling of the einsum. The arithmetic above is unchanged: it now calls
    this instead of inlining the same call.
    """
    return torch.einsum("mhd,nd->mhn", q.float(), k.float())


def _ref_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``dsa_topk_select`` then ``select_pools``' cast (``topk_select.py:369``, ``model_fp8.py:3439-3440``)."""
    return torch.topk(scores, int(k), dim=-1).indices.to(torch.int32)


# --------------------------------------------------------------------------- #
# THE CAUSAL BOUND, IN THE REFERENCE. `inc-glm53f-103`, declared as an amendment to a LANDED test
# in the block's Surface bullet at plan revision 260 -- not discovered at review.
#
# WHY THE REFERENCE HAD TO CHANGE AT ALL. `_ref_indexer` below selects pools from unbounded scores,
# so before this increment it agreed with the implementation and after it, it cannot: the bound
# changes WHICH pools a row selects, which is the whole point of the increment. Measured, not
# argued: the first counted run read `Mismatched elements: 5601 / 8960` and `8910 / 8960` on the two
# `test_run_1` arms. A reference that does not know about the bound disagrees BY CONSTRUCTION, and a
# stale reference is a wrong expectation rather than a finding.
#
# THESE THREE FUNCTIONS ARE TORCH AND STAY TORCH, AND THAT IS LOAD-BEARING TWICE OVER. The reference
# must reach NO seam -- `:2182-2196` MEASURES that it moves no projection counter and says the
# arrangement "is only sound if the reference is torch-only" -- so calling `dsa_causal_bound` here
# would make the reference compare the seam against itself and would fire that reading. And P13 is
# not in play: this is the test's own oracle, never a production path.
#
# TWO NUMBERS ARE READ OFF THE PRODUCT MODULE AND NEITHER IS TYPED HERE (repair `103r5`). The fill is
# no longer `-inf`: `dsa_causal_bound` writes the finite `BOUND_FILL` and the marker fires at or below
# `BOUND_FILL_MARK`, because the vendored selector pads its own input with a FINITE value and moves
# selected values through a 0/1 permutation matmul where `0 * -inf` is NaN -- read at the bytes in
# `increments/contradiction-103-selector-pad-6874a0f5.md`. :func:`_fill_constants` asks the module for
# both, so a change to either reaches this file as a mismatch instead of being overwritten by a
# hand-typed copy. Reading a module CONSTANT is not reaching a seam: it dispatches nothing, and the
# counters are reset AFTER the reference runs anyway (`:2192-2198` step 3).
#
# WHAT THEY DO NOT BUY. Because the reference now transcribes the same declared semantics as the
# implementation, `test_run_1` can no longer catch a defect in the ORDER of the sentinel columns --
# both sides pin it the same way. That defect class is caught by `test_run_2`, which compares a row
# against ITSELF run alone and needs no reference at all. The separation is deliberate: a reference
# checks the values, and the pack-commutation item checks that a row's answer does not depend on its
# neighbours.
# --------------------------------------------------------------------------- #


def _fill_constants() -> tuple[float, float]:
    """``(BOUND_FILL, BOUND_FILL_MARK)``, ASKED OF THE PRODUCT MODULE rather than typed here.

    Two numbers, one source. If the module ever moves the fill or the mark, this file's reference
    moves with it and `test_run_1` keeps measuring the implementation instead of measuring a stale
    copy of one of its constants. The pair is also why the reference needs no ``-inf`` anywhere: the
    fill is finite on purpose, and the reason is `103r5`'s contradiction record.
    """
    module = _seam_module("causal_bound")
    return float(module.BOUND_FILL), float(module.BOUND_FILL_MARK)


def _ref_causal_bound(
    scores: torch.Tensor, seq_lens: torch.Tensor, pool_size: int, width: int
) -> torch.Tensor:
    """``dsa_causal_bound``: ``BOUND_FILL`` at every pool column the row does not complete.

    Pool ``p`` completes at token ``(p + 1) * pool_size - 1`` and is attendable exactly when that
    token is ``< causal_len``, which is the same statement as ``p < causal_len // pool_size`` --
    the block's Surface bullet's predicate, written here as the inequality on the token so a reader
    can check it against upstream's ``[cu_seqlen_ks, cu_seqlen_ke)`` without re-deriving the floor
    division.

    THE FILL IS FINITE AND THAT IS THE `103r5` REPAIR, not a rounding of ``-inf``. A bounded column
    is moved across SBUF partitions by a 0/1 permutation matmul inside the vendored selector, and
    ``0 * -inf`` is NaN, so an ``-inf`` written here need not come back as one. The number itself is
    read from the module by :func:`_fill_constants`.

    ``causal_len`` IS ``seq_lens``, the same column the expansion receives -- one producer, two
    consumers, exactly as ``Glm5NextDSAIndexer.select_bounded_pools`` says of itself, named by the
    method because this increment has already pushed those line numbers twice.
    """
    fill, _mark = _fill_constants()
    causal = seq_lens.to(torch.int64).reshape(-1, 1)
    columns = torch.arange(int(width), dtype=torch.int64).reshape(1, -1)
    completes = (columns + 1) * int(pool_size) - 1 < causal
    return scores.masked_fill(~completes, fill)


def _ref_causal_sentinel(
    bounded: torch.Tensor, pool_ids: torch.Tensor, width: int
) -> torch.Tensor:
    """``dsa_causal_sentinel``: ``-1`` at every selection the row may not see.

    TWO ARMS, BECAUSE THE IMPLEMENTATION HAS TWO (repair ``103r5``). A selection is replaced when its
    VALUE is at or below ``BOUND_FILL_MARK`` -- or is NaN -- or when its INDEX is at or past
    ``width``, the count of real pool columns the selector was handed.

    THE INDEX ARM IS UNREACHABLE IN THIS REFERENCE, AND SAYING SO IS THE HONEST PART. It exists in
    the implementation because the vendored selector PADS its own input with a finite ``-9948.0`` at
    column positions that keep counting past the real width, so a pad can win a slot and come back as
    an out-of-range pool id. This reference's selector is ``_ref_topk``, a plain ``torch.topk`` over
    exactly ``width`` columns, which cannot return an index at or past ``width``. So this function
    transcribes the arm and never exercises it: the pad arm is READ by
    ``test/vllm_neuron/functional/dsa/test_causal_bound.py``'s pad case against the kernel's own
    config, and at the dispatch site by
    :func:`test_forward_at_a_striking_select_k_sentinelises_every_pad_the_selector_returns` below.

    THE REFERENCE GATHERS AND THE IMPLEMENTATION NO LONGER DOES, on purpose (repair ``103r4``,
    review finding F1). ``select_bounded_pools`` now hands the sentinel ``dsa_topk_select``'s own
    returned ``values``, because the vendored selector strikes its input buffer on the multi-fold
    branch and an index returned beside a struck value can then point at an originally finite
    column. THIS reference is safe to gather because ``torch.topk`` never modifies its input, so the
    gathered score IS the selected value here by construction -- and the two sides therefore reach the
    same answer by two different spellings, which is what a reference is for. ``isnan`` is carried for
    the same reason the module's oracle carries it: nothing here produces a NaN, and if one ever
    arrives the reference agrees with the implementation about it rather than silently disagreeing.
    """
    _fill, mark = _fill_constants()
    selected = bounded.gather(1, pool_ids.to(torch.int64))
    struck = torch.le(selected, mark) | torch.isnan(selected)
    padded = torch.ge(pool_ids.to(torch.int64), int(width))
    return torch.where(struck | padded, torch.full_like(pool_ids, -1), pool_ids)


def _ref_canonical_sentinel_order(pool_ids: torch.Tensor) -> torch.Tensor:
    """Sentinels to the trailing columns; real ids keep their relative order.

    Mirrors ``Glm5NextDSAIndexer._canonical_sentinel_order``. It exists in the reference for the
    same reason it exists in the implementation: the selector promises "highest first" and promises
    NOTHING about the order among EQUAL values (``topk_select.py:312``), and the bound manufactures
    equal values in bulk at ``BOUND_FILL`` -- a FINITE ``-1e30`` since ``-103``, not the ``-inf``
    this line said until rev 268. Without this the two sides would differ on the PLACES of
    identical contents, which is not a numeric disagreement and would read like one.

    THE FILL IS NOT THE ONLY SOURCE OF EQUAL VALUES, which is why this ordering is necessary but not
    sufficient. Two REAL candidates can also hold the same score -- the product's own ReLU floors
    every all-non-positive candidate to an identical ``0.0`` -- and their relative order is pinned by
    nothing here. That case is handled by the tie-equivalence comparison rather than by this sort.
    """
    k = int(pool_ids.shape[1])
    position = torch.arange(k, dtype=torch.int64)
    key = (pool_ids < 0).to(torch.int64) * k + position
    return pool_ids.gather(1, key.argsort(dim=1))


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
    probe_per_head: list[torch.Tensor] | None = None,
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
    # THE PROBE KEEPS THE UNBOUNDED SCORES, ON PURPOSE. Its one consumer is the tie control, which
    # asserts the selection was not decided by a tie. The bound below writes `-inf` into every column
    # a row does not complete, so handing it BOUNDED scores would make the tie control fire on every
    # row that completes fewer than `select_k` pools -- reporting the behaviour under test as a
    # fixture problem. The control still does its job on these scores and the reasoning is short: for
    # a row completing at least `select_k` pools the selection is made among finite columns, and "no
    # two of all columns tie" implies "no two of those tie"; for a row completing fewer, every
    # remaining pick is a `-inf` sentinel whose PLACE is pinned by `_ref_canonical_sentinel_order`
    # rather than chosen. Both cases are therefore decided, and neither is decided by a tie.
    if probe is not None:
        probe.append(scores.detach())
    # `-103` r7 READ 2. Recomputed rather than captured inside `_ref_score`, so the value path above
    # is not touched by a diagnostic. Same inputs, same function, so it cannot disagree with the
    # scores the tie control reads.
    if probe_per_head is not None:
        probe_per_head.append(_ref_score_per_head(query, candidate_keys).detach())
    # inc-glm53f-103: bound, select, sentinelise, then pin the sentinel places -- the same four steps
    # in the same order as `Glm5NextDSAIndexer.select_bounded_pools` in `model_fp8.py` -- named by the
    # method rather than by a line, because this increment's own docstrings pushed those line numbers
    # twice already -- transcribed rather than
    # called. The order is the plan's declared order and is not free to vary: selecting on unbounded
    # scores and masking afterwards would answer a different question.
    bounded = _ref_causal_bound(scores, seq_lens, pool, candidates)
    pool_ids = _ref_topk(bounded, int(indexer.select_k()))
    # `candidates` IS the width, read from the same variable the bound was given, exactly as the
    # implementation reads `int(bounded.shape[1])` rather than a config field: the width the marker
    # screens against cannot then disagree with the width the selector was handed.
    pool_ids = _ref_causal_sentinel(bounded, pool_ids, candidates)
    pool_ids = _ref_canonical_sentinel_order(pool_ids)
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
    probe_per_head: list[torch.Tensor] | None = None,
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
        probe_per_head=probe_per_head,
    )
    attn_out = _ref_attend(
        attention, normed, latent_cache, int(start_position), topk_indices, float(softmax_scale)
    )
    return residual + attn_out


# =========================================================================== #
# THE FIXTURE. A materialised stack at :data:`TINY_GEOMETRY`.


#: The softmax scale, DERIVED THE WAY THE REFERENCE DERIVES IT: the inverse square root of the QUERY
#: head width ``qk_nope_head_dim + qk_rope_head_dim``. That width is the fork's own ``qk_head_dim``
#: (``model_fp8.py:4435``), summed rather than taken from the nope width alone because "the rotary
#: slice is 0 on this checkpoint and that 0 is a value" (``:415-418``), and the factor upstream folds
#: is ``head_dim ** -0.5`` on that same width (``:3333-3336``, quoted there).
#:
#: WHY NOT ``kv_lora_rank``, which this line used to read. Absorbing the KV projection makes the score
#: GEMM contract over the latent width, and that is what the retired comment argued from -- but the
#: contraction width is not the width the reference scales by. On both fixture geometries
#: ``kv_lora_rank`` is twice ``qk_nope_head_dim``, so the retired value was low by exactly ``sqrt(2)``
#: (0.0883883 against 0.125 here, 0.0441942 against 0.0625 on the published config). No landed result
#: moved when this changed, because :func:`_reference` applies the SAME scale to the torch reference
#: as the kernel receives (``:1577``) -- which is why the mis-derivation survived landing and why no
#: existing item can be pointed at as evidence that either value is right. Corrected under
#: ``inc-glm53f-109`` (DECISIONS section 362, open question ``oq-051-softmax-scale-sqrt2``).
#:
#: ``attend`` still takes the scale as a caller's argument on purpose and its own comment says why --
#: "no block registers a value for it ... deriving one here would mint a registered value this
#: increment has no authority to mint" (``model_fp8.py:4480-4482``). So this file derives one for its
#: own run and registers nothing.
SOFTMAX_SCALE = float(
    (TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]) ** -0.5
)


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


def build_layer_stack(*, layers: int = LAYERS, seed: int = 51_051_051, **cfg_overrides):
    """A stack of :class:`Glm5NextDSALayer` at the tiny geometry, every leaf materialised.

    The weight scaling mirrors the ``-042`` sibling's fixture (``test_mla_decode.py:139-143``):
    ``randn * in_features ** -0.5``, so activations stay order one through a 3-layer chain instead of
    growing and turning a tolerance comparison into a test of overflow.

    EVERY LAYER GETS ITS OWN SEED STREAM from one generator, so the three layers are genuinely
    different maps. A stack of three identical layers would let a per-layer indexing error pass.

    ``cfg_overrides`` REACHES :func:`_tiny_text_config` AND NOTHING ELSE, so the tiny geometry stays
    the one place a width is declared. It exists for one caller: the striking-``select_k`` item needs a
    larger ``index_topk``, which is the dial upstream itself expresses in TOKENS, and every reading it
    makes is then derived from the config it built rather than from a number typed twice.
    """
    model_fp8 = _impl()
    cfg = _tiny_text_config(**cfg_overrides)
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
    (``model_fp8.py:3569``), so a ``-1`` here is how a position says "my window is not a whole pool";
    those rows are steered to the trash row rather than dropped (``model_fp8.py:4272-4278``).
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


def say_tie_diagnostics(
    scores: torch.Tensor, k: int, label: str, per_head: torch.Tensor | None = None
) -> None:
    """`-103` r7 READ 2. Disclose the row that DECIDES the k-th-place gap, before any assertion.

    The r7 run read `prefill-layer2 kth_place_gap_min = 0.000e+00`, and an exact zero is not a near
    miss: `_ref_score` applies a ReLU (``clamp(min=0.0)``) before the head weights, so every candidate
    whose per-head scores are all non-positive collapses to the SAME exact `0.0`. Two candidates on
    that floor is one row's bad luck; several is structure, and only the count can tell them apart.
    So the count is printed rather than argued -- with the per-head numbers behind it when the caller
    has them, because the floor is reached in the per-head values and not in the weighted sum.

    The gap arithmetic is not respelled here: the row is chosen by the same top-``k+1`` difference
    :func:`kth_place_gap` minimises, so the row named below IS the row that produced the reading.
    """
    s = scores.float()
    top = torch.topk(s, int(k) + 1, dim=-1).values
    row_gaps = top[:, int(k) - 1] - top[:, int(k)]
    row = int(torch.argmin(row_gaps))
    row_values = s[row]
    say(
        label,
        "tie_read2",
        f"row={row}",
        f"gap={float(row_gaps[row]):.9e}",
        f"floor={TIE_FLOOR:.3e}",
        f"candidates={int(row_values.numel())}",
        f"at_exact_zero={int((row_values == 0.0).sum())}",
        f"distinct={len(set(row_values.tolist()))}",
        "values=" + ";".join(f"{float(v):.9e}" for v in row_values),
    )
    if per_head is not None:
        block = per_head.float()[row]
        say(
            label,
            "tie_read2_per_head",
            f"row={row}",
            f"heads={int(block.shape[0])}",
            f"nonpositive={int((block <= 0.0).sum())}",
            f"of={int(block.numel())}",
            "block=" + ";".join(
                "|".join(f"{float(x):.6e}" for x in head) for head in block
            ),
        )


def assert_selection_is_not_a_tie(
    scores: torch.Tensor, k: int, label: str, per_head: torch.Tensor | None = None
) -> None:
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
    say_tie_diagnostics(scores, int(k), label, per_head)
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
        # `-103` r7 READ 2 collects the per-head scores beside the weighted ones, so a tied row can be
        # read at the place the ReLU floor is actually reached.
        probe_per_head: list[torch.Tensor] = []
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
                probe_per_head=probe_per_head,
            )
        reference = ref_hidden

        # THE TIE CONTROL, on every layer's own selection. Run BEFORE the comparison so a tie is
        # reported as a fixture problem rather than surfacing as a numeric failure downstream.
        assert len(probe) == int(layers), (
            f"the probe collected {len(probe)} score tensors for {layers} layer(s); every layer "
            f"selects once per phase, so a short count means a layer was skipped"
        )
        assert len(probe_per_head) == len(probe), (
            f"the per-head probe collected {len(probe_per_head)} tensors against the score probe's "
            f"{len(probe)}; the two are appended in the same call, so a difference means one of them "
            f"was not threaded through every layer"
        )
        for idx, scores in enumerate(probe):
            assert_selection_is_not_a_tie(
                scores,
                int(stack[idx].attention.indexer.select_k()),
                f"{phase}-layer{idx}",
                probe_per_head[idx],
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

    # THE ORACLE MIRRORS THE PRODUCT'S FOUR-STEP CHAIN, and until rev 268 it did not (item 3, a
    # stale oracle). It selected straight off the UNBOUNDED scores and expanded, while
    # `select_bounded_pools` bounds, selects, marks and canonically orders -- three steps this arm
    # never applied. So the arm was comparing the packed implementation against a reference for an
    # older product, and any disagreement the bound or the marker introduced would have been read
    # here as a packing defect. The steps and their order are transcribed from `_ref_indexer` above,
    # which is the single place this file spells the chain; the order is not free to vary, because
    # selecting on unbounded scores and masking afterwards answers a different question.
    ref_rows = []
    offset = 0
    for n, scores in scored:
        own_lens = seq_lens[offset : offset + n]
        bounded = _ref_causal_bound(scores, own_lens, pool, candidates)
        pool_ids = _ref_topk(bounded, select_k)
        pool_ids = _ref_causal_sentinel(bounded, pool_ids, candidates)
        pool_ids = _ref_canonical_sentinel_order(pool_ids)
        ref_rows.append(_ref_expand(pool_ids, own_lens, pool))
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
        f"projections bit-for-bit (model_fp8.py:4437-4448); this is that claim failing, not a "
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


# =========================================================================== #
# THE SHORT-SEQUENCE CAUSAL BYPASS -- inc-glm53f-099's dispatch half.
#
# WHAT THE BYPASS IS, in one sentence. When a request is short enough that selecting the top
# select_k pools would take every candidate there is, there is nothing to select, so the indexer
# returns the plain causal index rows instead of running the score/select/expand chain -- which is
# what upstream does in the same regime (`sparse_attn_indexer_kpool.py:203-217`).
#
# WHAT USED TO BE HERE. `inc-glm53f-098` refused this regime by name and cited this increment as the
# owner of the route. That refusal and its two items are gone; the retirement note is in the refusal
# section above. The guarantee the refusal protected -- `dsa_topk_select` never sees `k == width`,
# because its gate returns False rather than raising (`topk_select.py:294`) and would take the torch
# route silently -- is protected here instead, by reading `topk_select`'s counter as a ZERO on the
# bypass case. That is a stronger reading than the refusal gave: the refusal proved the call never
# happened by never running at all, and this proves it never happened while the indexer ran to
# completion and returned a correct answer.
#
# THE BOUNDARY IS THE CASE, not a comfortable interior value. `BYPASS_SEQ_LEN` yields EXACTLY
# `select_k()` complete pools, which is the largest length the bypass serves. A non-strict bound
# would select here and a clamped `k` would too, so this is the length that tells the three
# candidate implementations apart.
#
# THREE ITEMS, NO parametrize, per plan section 6 rule 6 -- one item per counted conjunct. The two
# entry points are two conjuncts because they are two code paths that can drift apart, and the
# cross-entry-point comparison is a third because it fails for a reason neither of the first two can
# reach: both serving the regime, differently.

#: The largest length the bypass serves at this file's dials: exactly ``select_k()`` complete pools.
#: DERIVED from the two dials rather than typed, so a dial change moves the case instead of quietly
#: turning it into a selecting case that would pass for the wrong reason.
BYPASS_SEQ_LEN = TOPK_POOLS * POOL_SIZE

#: The five families the BYPASS DECISION governs. They must read ``(0, 0)`` on BOTH entry points.
#: Four are the selection chain the bypass skips. ``decode_tail_update`` is here because neither
#: case below is a decode step, so its zero is a PHASE reading rather than a bypass reading -- said
#: plainly so a reader does not count it as evidence about selection.
BYPASS_ZERO_FAMILIES: tuple[str, ...] = (
    "paged_gather", "score_gemm", "topk_select", "index_expand", "decode_tail_update",
)


def _ref_causal_rows(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """The bypass's answer computed by plain torch in THIS file: row ``i`` is ``0..seq_lens[i]-1``.

    Computed here rather than by calling ``dsa_causal_fill_torch_oracle``, deliberately. That oracle
    lives in the module under test, so using it would compare the module against itself; this file's
    ``_ref_*`` convention is that the two sides arrive by different means. Four lines of torch is
    the whole reference, which is also the point -- a reference a reader can check by eye is worth
    more here than a shared helper.
    """
    rows = int(seq_lens.shape[0])
    columns = torch.arange(int(width), dtype=torch.int32).expand(rows, int(width))
    positions = seq_lens.to(torch.int32).reshape(rows, 1) - 1
    return torch.where(columns <= positions, columns, torch.full_like(columns, -1))


def _bypass_width(indexer, pool: int) -> int:
    """The width the bypass must emit, read from the expansion seam's own helper, with three checks.

    THREE READINGS RATHER THAN ONE, the same shape the ragged arm's expansion check uses and for the
    same reason: if ``index_expand_width`` were itself wrong, a comparison against it alone would
    pass. So the multiple-of-``KEY_CHUNK`` arm and the not-below-raw arm are read too, with
    ``KEY_CHUNK`` taken from the module that DEFINES it (``mla_sparse.py:111``).
    """
    expand_mod = _seam_module("index_expand")
    select_k = int(indexer.select_k())
    raw = int(expand_mod.index_expand_raw_width(select_k, pool))
    emitted = int(expand_mod.index_expand_width(select_k, pool))
    key_chunk = int(
        importlib.import_module("vllm_neuron.functional.attention.mla_sparse").KEY_CHUNK
    )
    say("bypass", "width", emitted, "raw", raw, "key_chunk", key_chunk, "select_k", select_k)
    assert emitted % key_chunk == 0, (
        f"the emitted width {emitted} is not a whole multiple of KEY_CHUNK {key_chunk}, which is "
        f"the allocation rule the sparse kernel admits"
    )
    assert emitted >= raw, f"the emitted width {emitted} is below the raw expansion width {raw}"
    # THE ARITHMETIC THE BYPASS RESTS ON, read as a value rather than argued. The bypass holds only
    # while `max_seq_len // pool <= select_k`, so the largest position it can present is
    # `select_k * pool + pool - 2`, and that must sit strictly inside the emitted width -- otherwise
    # some admissible length would need a clamp that nothing implements.
    max_position = select_k * int(pool) + int(pool) - 2
    say("bypass", "max_position", max_position, "headroom_columns", emitted - 1 - max_position)
    assert max_position < emitted, (
        f"the widest bypass position {max_position} does not fit inside {emitted} column(s), so "
        f"some admissible length would write past the last column"
    )
    return emitted


def _causal_fill_api():
    """``(reset, read)`` for the bypass's own counter, discovered by the SAME rule as the seven.

    ``causal_fill`` is a dsa module and follows the naming convention, so the discovery helper
    reaches it unchanged. It is deliberately NOT folded into ``FAMILIES``: ``FAMILIES`` is the census
    the call spy attributes against and the declared per-layer tables are written against, and
    neither of those is about this seam.
    """
    return _discover_counter_api(_seam_module("causal_fill"))


def _declared_column(table: dict[str, tuple[int, int]], column: int) -> dict[str, int]:
    """One PHASE column of a declared table, folded onto families.

    ``declared_family_totals`` sums prefill AND decode, which is right for a layer case that runs
    both. The two cases below each run ONE phase, so each needs its own column: ``forward``'s
    prefill leg reads column 0 of :data:`DECLARED_PER_LAYER`, and the ragged arm is a decode step so
    it reads column 1 of :data:`DECLARED_PER_LAYER_RAGGED_ARM`. Deriving the two figures from the
    landed tables is the whole point -- typing "2" and "1" here would make the comparison below a
    claim about this file instead of a claim about the arms' declared design.
    """
    out = {family: 0 for family in FAMILIES}
    for entry, counts in table.items():
        out[ENTRY_POINTS[entry]] += int(counts[column])
    return out


def _bypass_forward_operands(cfg, *, seed: int) -> dict:
    """Operands for ``forward``'s prefill leg at the bypass boundary. One builder, two callers."""
    gen = torch.Generator().manual_seed(int(seed))
    return {
        "hidden": torch.randn(
            BYPASS_SEQ_LEN, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            BYPASS_SEQ_LEN, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.arange(1, BYPASS_SEQ_LEN + 1, dtype=torch.int32),
        "slot_mapping": prefill_slot_mapping(BYPASS_SEQ_LEN, int(cfg.index_kpool)),
    }


def _bypass_ragged_operands(cfg, *, seed: int) -> dict:
    """Operands for the NON-UNIFORM ragged arm at the bypass boundary.

    The arm refuses a uniform batch by name, so the two request lengths differ. It only READS the
    pool cache and on the bypass never reads it at all, but ``_require_serviceable`` still checks the
    cache's geometry, so it is shaped the way production shapes it.
    """
    lengths = [BYPASS_SEQ_LEN // 2, BYPASS_SEQ_LEN]
    max_len = max(lengths)
    gen = torch.Generator().manual_seed(int(seed))
    return {
        "hidden": torch.randn(
            len(lengths), max_len, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            len(lengths), max_len, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.cat([torch.arange(1, n + 1, dtype=torch.int32) for n in lengths]),
        "lengths": lengths,
    }


def _run_bypass(indexer, method: str, ops: dict, monkeypatch: pytest.MonkeyPatch):
    """Run ONE entry point on the short regime with both instruments installed and reset first.

    Returns ``(result, family_readings, causal_fill_reading, spy)``. One runner for all three items,
    so no item can differ from another by how it drove the call rather than by what it asserted.
    """
    reset_all_counters()
    _causal_fill_api()[0]()
    spy = SeamSpy()
    spy.install(monkeypatch)
    try:
        if method == "forward":
            result = indexer.forward(
                ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
                max_seq_len=BYPASS_SEQ_LEN, page_size=PAGE_SIZE,
                slot_mapping=ops["slot_mapping"],
            )
        else:
            result = indexer.forward_ragged(
                ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
                ops["lengths"], max_seq_len=BYPASS_SEQ_LEN, page_size=PAGE_SIZE,
            )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    fill = tuple(int(v) for v in _causal_fill_api()[1]())
    return result, readings, fill, spy


def _assert_regime_is_the_boundary(label: str, indexer, pool: int) -> int:
    """The case sits ON the strict bound, asserted from the closed form. Returns ``select_k``."""
    select_k = int(indexer.select_k())
    candidates = BYPASS_SEQ_LEN // int(pool)
    say(label, "regime", "seq_len", BYPASS_SEQ_LEN, "candidates", candidates, "select_k", select_k)
    assert candidates == select_k, (
        f"{BYPASS_SEQ_LEN} token(s) yields {candidates} complete pool(s) against select_k="
        f"{select_k}; these items exist to sit ON the boundary, where a non-strict bound and a "
        f"clamped k would both pass and a correct bypass is the only thing that ALSO reads zero on "
        f"topk_select"
    )
    return select_k


def _check_bypass_zeros(label: str, readings: dict[str, tuple[int, int]]) -> None:
    """The five zeros, plus the reading that makes them measurements rather than decoration.

    D1.5: a counted zero needs a control that fires on the same call. ``kpool_hadamard`` is that
    control and it costs nothing extra -- ``project_stage`` rotates the indexer query on EVERY
    indexer call, so the family is non-zero on both entry points and in both regimes. If the counter
    instrument were not reading this call at all, that figure would read zero too and this assertion
    fails before the five below can pass for the wrong reason.
    """
    control = readings["kpool_hadamard"]
    say(label, "D1.5_control", "kpool_hadamard", control)
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the "
        f"counter instrument is not reading this call and the five zeros below measure nothing"
    )
    for family in BYPASS_ZERO_FAMILIES:
        assert readings[family] == (0, 0), (
            f"{family} read {readings[family]} on the bypass; the bypass exists to skip the "
            f"selection chain, so anything it dispatched is work done for an answer it discards"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the bypass path: {readings}"
    )


def _check_the_kernel_ran(label: str, fill: tuple[int, int]) -> None:
    """The route predicate, form R-1, plus the identity read THROUGH the seam (D13.1)."""
    identity = _seam_module("causal_fill").causal_fill_kernel_identity()
    say(label, "causal_fill", "nki_dispatch", fill[0], "torch_fallback", fill[1],
        "kernel", identity)
    assert fill == (1, 0), (
        f"the bypass read {fill} on its own seam and owes exactly one NKI dispatch with no "
        f"fallback; a torch-level fill would read (0, 1) here and P13 forbids it"
    )
    assert identity is not None, "no kernel identity was recorded, so nothing certifies what ran"
    assert identity[1].endswith("_causal_fill_nki"), (
        f"the seam dispatched {identity}, not the NKI kernel this increment landed"
    )


def _check_two_instruments_agree(label: str, spy: SeamSpy, readings: dict) -> None:
    """The spy and the seven counters over the same events. One zero is not a reading; two are."""
    for family, (spy_sum, counter_sum) in agreement(spy, readings).items():
        say(label, "agreement", family, "spy", spy_sum, "counter", counter_sum)
        assert spy_sum == counter_sum, (
            f"the spy counted {spy_sum} {family} call(s) where the counter read {counter_sum}; two "
            f"instruments over the same events must agree, or neither zero is a reading"
        )


def _check_rows_are_exact(label: str, got: torch.Tensor, reference: torch.Tensor,
                          want_shape: tuple[int, int]) -> None:
    """Exact int32 equality, its shape and dtype, and a doctored control that must fail."""
    say(label, "shape", tuple(got.shape), "want", want_shape, "dtype", str(got.dtype))
    assert got.dtype == torch.int32, f"the bypass returned {got.dtype}, not int32"
    assert tuple(got.shape) == want_shape, (
        f"the bypass emitted {tuple(got.shape)} where it owes {want_shape}: it must emit the SAME "
        f"shape selection emits, or every consumer would have to branch on the regime"
    )
    diff = int((got.to(torch.int64) - reference.to(torch.int64)).abs().max())
    say(label, "max_abs_diff", diff, "entries", got.numel(), "sentinels", int((got == -1).sum()))
    assert diff == 0, (
        f"the bypass rows differ from this file's torch reference by up to {diff}. No tolerance is "
        f"used and none is admissible: these are indices, and a tolerance would hide an off-by-one"
    )
    # THE COMPARISON MUST BE ABLE TO FAIL, or `max_abs_diff == 0` is decoration. One entry of the
    # reference is moved by one and the same reader runs again over the same population.
    doctored = reference.clone()
    doctored[0, 0] = int(doctored[0, 0]) + 1
    differing = int((got != doctored).sum())
    say(label, "doctored_control", "differing", differing, "population", got.numel())
    assert differing == 1, (
        f"the element-wise reader found {differing} differing entries against a reference with "
        f"exactly one entry moved, so it is not reading the tensors it claims to compare"
    )


def test_forward_SERVES_the_short_regime_with_exact_causal_rows_and_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward``: at the bypass boundary the pool write still lands and selection never runs.

    THREE READINGS, and they fail for three different reasons.
      1. THE ANSWER IS EXACT. The returned int32 rows equal this file's own torch reference element
         for element, with no tolerance -- exact integer equality is the comparator this increment
         registered, and an index compared with a tolerance would hide an off-by-one.
      2. THE WRITE STAGE RAN. This is the whole reason the bypass sits where it does instead of
         inside ``_require_serviceable``: both entry points call that helper BEFORE they write the
         pooled-key store, so an early return from there would have skipped the write silently.
         Read as a before-and-after ROW COUNT, never as a boolean.
      3. SELECTION NEVER DISPATCHED, read by two independent instruments -- the seven family
         counters and the call spy -- because one instrument reading zero cannot tell "it did not
         happen" from "I was not looking".
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("D1", indexer, pool)
    width = _bypass_width(indexer, pool)

    ops = _bypass_forward_operands(cfg, seed=9_099_001)
    pool_cache = ops["pool_cache"]

    def written_rows() -> int:
        return int((pool_cache.to(torch.float32).abs().sum(dim=1) != 0).sum())

    before = written_rows()
    reference = _ref_causal_rows(ops["seq_lens"], width)
    got, readings, fill, spy = _run_bypass(indexer, "forward", ops, monkeypatch)

    # 2. THE WRITE STAGE RAN, as a value and not a boolean.
    after = written_rows()
    say("D1", "pool_rows_written", "before", before, "after", after,
        "of", int(pool_cache.shape[0]))
    assert before == 0, f"the fixture handed a pre-populated pool_cache ({before} row(s))"
    assert after > before, (
        "the pooled-key store is untouched after a prefill call, so the bypass returned BEFORE the "
        "write instead of after it -- the exact defect the placement exists to prevent"
    )

    # 3. SELECTION NEVER DISPATCHED.
    for family in FAMILIES:
        say("D1", "counter", family, readings[family])
    _check_bypass_zeros("D1", readings)
    spy.report("D1")
    _check_two_instruments_agree("D1", spy, readings)
    _check_the_kernel_ran("D1", fill)

    # 1. THE ANSWER IS EXACT.
    _check_rows_are_exact("D1", got, reference, (BYPASS_SEQ_LEN, width))


def test_forward_ragged_SERVES_the_short_regime_with_exact_causal_rows_and_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward_ragged``: the same bypass on the non-uniform arm, and the pack still runs.

    THE PLACEMENT IS THE EXTRA READING HERE. This arm writes no cache and advances no ring, so the
    only thing its bypass placement decides is whether the pack still happens -- and the pack is
    this arm's whole reason for existing (``design-20260905-aa``'s seventh family). The bypass
    therefore sits AFTER the pack, and this item reads ``ragged_pack`` as NON-ZERO to say so.
    Bypassing before the pack would zero the seventh family on the short regime and nothing else in
    this file would notice.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("D2", indexer, pool)
    width = _bypass_width(indexer, pool)

    ops = _bypass_ragged_operands(cfg, seed=9_099_002)
    tokens = int(ops["seq_lens"].shape[0])
    assert tokens == sum(ops["lengths"]), "one seq_len per PACKED row"
    assert len(set(ops["lengths"])) > 1, "a uniform batch is refused by the arm, and rightly"

    reference = _ref_causal_rows(ops["seq_lens"], width)
    got, readings, fill, spy = _run_bypass(indexer, "forward_ragged", ops, monkeypatch)

    for family in FAMILIES:
        say("D2", "counter", family, readings[family])
    _check_bypass_zeros("D2", readings)
    pack = readings["ragged_pack"]
    say("D2", "pack_still_ran", pack)
    assert pack[0] > 0, (
        f"ragged_pack read {pack} on the bypass, so the bypass returned BEFORE the pack and zeroed "
        f"the seventh counter family on this regime. The placement is after the pack on purpose"
    )
    spy.report("D2")
    _check_two_instruments_agree("D2", spy, readings)
    _check_the_kernel_ran("D2", fill)
    _check_rows_are_exact("D2", got, reference, (tokens, width))


def test_the_two_entry_points_read_the_SAME_bypass_governed_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass is ONE decision, so the families it governs must read the same on both calls.

    WHAT THIS CATCHES that neither item above can: both entry points serving the short regime
    DIFFERENTLY. Each item above reads its own call in isolation, so a change that made one entry
    point bypass and the other select would fail only the one item that happened to be looking --
    and if a future edit relaxed that item, nothing would compare the two paths at all.

    FIVE OF THE SEVEN FAMILIES MUST MATCH, AND TWO MUST NOT, and the two that must not are read
    against the arms' own landed tables rather than against each other. That split is a measurement
    this seat took, not a softening of the claim: ``forward``'s prefill leg writes the pooled-key
    store and the ragged arm declares that it does not, while the arm packs and the layer case
    declares that it does not (``DECLARED_PER_LAYER`` against
    ``DECLARED_PER_LAYER_RAGGED_ARM``). So ``kpool_hadamard`` and ``ragged_pack`` differ by design at
    ANY bypass placement, and making them equal would mean authoring a dispatch to move a counter --
    which this file has refused by name three times. Ruled at DECISIONS section 97 (v).
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("D3", indexer, pool)

    # Each call gets its OWN operands, so neither call can see the other's pool write.
    _got_f, forward_readings, forward_fill, _spy_f = _run_bypass(
        indexer, "forward", _bypass_forward_operands(cfg, seed=9_099_003), monkeypatch
    )
    _got_r, ragged_readings, ragged_fill, _spy_r = _run_bypass(
        indexer, "forward_ragged", _bypass_ragged_operands(cfg, seed=9_099_004), monkeypatch
    )

    for family in FAMILIES:
        say("D3", "family", family, "forward", forward_readings[family],
            "forward_ragged", ragged_readings[family],
            "same" if forward_readings[family] == ragged_readings[family] else "DIFFERS")

    # THE FIVE THAT MUST MATCH, and both claims are made: equal to each other, and equal to zero.
    # Equality alone would pass if both entry points selected identically.
    for family in BYPASS_ZERO_FAMILIES:
        assert forward_readings[family] == ragged_readings[family], (
            f"{family} read {forward_readings[family]} on forward and {ragged_readings[family]} on "
            f"forward_ragged; the bypass is one decision and the families it governs cannot differ"
        )
        assert forward_readings[family] == (0, 0), (
            f"{family} read {forward_readings[family]} on BOTH entry points, equally and non-zero, "
            f"so the two agree while both are still running selection"
        )
    say("D3", "matched_families", len(BYPASS_ZERO_FAMILIES), "of", len(FAMILIES))

    # THE FILL RAN ON BOTH, which is the positive half of the same claim.
    say("D3", "causal_fill", "forward", forward_fill, "forward_ragged", ragged_fill)
    assert forward_fill == ragged_fill == (1, 0), (
        f"the bypass seam read {forward_fill} on forward and {ragged_fill} on forward_ragged; both "
        f"owe exactly one NKI dispatch and no fallback"
    )

    # THE TWO THAT DIFFER, each against its OWN arm's declared column, derived not typed.
    declared_forward = _declared_column(DECLARED_PER_LAYER, 0)
    declared_ragged = _declared_column(DECLARED_PER_LAYER_RAGGED_ARM, 1)
    for family in ("kpool_hadamard", "ragged_pack"):
        say("D3", "declared_difference", family,
            "forward", forward_readings[family], "declares", declared_forward[family],
            "forward_ragged", ragged_readings[family], "declares", declared_ragged[family])
        assert forward_readings[family][0] == declared_forward[family], (
            f"{family} read {forward_readings[family][0]} on forward's prefill leg where "
            f"DECLARED_PER_LAYER's prefill column declares {declared_forward[family]}"
        )
        assert ragged_readings[family][0] == declared_ragged[family], (
            f"{family} read {ragged_readings[family][0]} on the ragged arm where "
            f"DECLARED_PER_LAYER_RAGGED_ARM's decode column declares {declared_ragged[family]}"
        )
    differing = sorted(
        f for f in FAMILIES if forward_readings[f] != ragged_readings[f]
    )
    say("D3", "families_that_differ", differing)
    assert differing == ["kpool_hadamard", "ragged_pack"], (
        f"the two entry points differ on {differing}; exactly two families may differ on the short "
        f"regime and both are named by the arms' landed tables, so a third difference is either a "
        f"new dispatch on one path or a lost one on the other"
    )


# =========================================================================== #
# inc-glm53f-109. THE SOFTMAX SCALE, AGAINST THE REFERENCE'S OWN DERIVATION.
#
# Why these two items exist when no landed result moved. :func:`_reference` at
# ``:1577`` applies the SAME scale to the torch reference as ``attend`` receives, so
# every landed item here measures agreement AT a chosen scale and is blind to which
# scale was chosen. That blindness is what let ``kv_lora_rank ** -0.5`` land and is
# why no existing item can be cited as evidence for either value. These two items
# read the constant against the reference's FORMULA instead.


def test_softmaxscale_is_the_reference_derivation_and_NOT_the_latent_rank() -> None:
    """Item (a): the constant is the QUERY head width's inverse square root.

    The formula is written out again here from this file's own geometry dict, and
    deliberately not factored into a helper the constant also calls -- a shared helper
    would make this item compare the constant to itself and pass at any scale.
    """
    width = TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]
    reference = float(width**-0.5)
    say("D109", "query_head_width", width, "reference", repr(reference),
        "constant", repr(SOFTMAX_SCALE))
    assert SOFTMAX_SCALE == reference, (
        f"SOFTMAX_SCALE is {SOFTMAX_SCALE!r}; the reference's derivation over this file's "
        f"own geometry is {reference!r}. The scale is the inverse square root of the QUERY "
        f"head width qk_nope_head_dim + qk_rope_head_dim = {width}, which is the fork's own "
        f"qk_head_dim (model_fp8.py:4435), summed for the reason :415-418 gives"
    )


def test_softmaxscale_control_the_retired_latent_rank_value_fails_that_item() -> None:
    """Item (b), THE CONTROL, and item (a) is only evidence because this one passes.

    Item (a) would also pass on a geometry where the two derivations coincide, and this
    fixture is not such a geometry: ``kv_lora_rank`` is twice ``qk_nope_head_dim`` here,
    so the retired value is low by exactly ``sqrt(2)``. This item measures that gap
    rather than asserting it, and it fails if a future geometry edit collapses it.
    """
    retired = float(TINY_GEOMETRY["kv_lora_rank"] ** -0.5)
    reference = float(
        (TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]) ** -0.5
    )
    ratio = reference / retired
    say("D109", "retired", repr(retired), "reference", repr(reference),
        "ratio", repr(ratio), "sqrt2", repr(2.0**0.5))
    assert retired != reference, (
        f"the retired derivation kv_lora_rank ** -0.5 and the reference's derivation both "
        f"read {reference!r} on this geometry, so item (a) cannot tell them apart and is "
        f"vacuous here; the two must differ for that item to mean anything"
    )
    assert abs(ratio - 2.0**0.5) <= 1e-12, (
        f"the reference is {ratio!r} times the retired value; the block's ground is that "
        f"kv_lora_rank is twice qk_nope_head_dim on this fixture, so the ratio is sqrt(2) "
        f"= {2.0**0.5!r} to within 1e-12. A different ratio means the geometry moved"
    )
    assert SOFTMAX_SCALE != retired, (
        f"SOFTMAX_SCALE still reads the retired latent-rank value {retired!r}"
    )


# =============================================================================================== #
# THE SELECTING-REGIME CAUSAL BOUND -- inc-glm53f-103's dispatch half.
#
# WHAT THE BOUND IS, in one sentence. Above the bypass bound the selector runs, and a query row must
# not select a key pool whose tokens finish AFTER the row's own position -- so every such pool's
# score is pushed to `-inf` before selection and any selection that comes back holding `-inf` is
# replaced by the `-1` sentinel.
#
# WHY THE READING BELONGS IN THIS FILE and not only in `test_causal_bound.py`. That file measures
# the two kernels against their oracles on a synthetic shape. This item measures the DISPATCH SITE:
# that the indexer's own forward composes the bound, the landed selector and the sentinel in the
# right order, on operands the indexer built itself, with the counters that prove which route ran.
# A kernel that is correct and never called would pass there and fail here.
#
# TWO ITEMS IN THIS FILE, AND THE SECOND ONE IS A REPAIR (`103r5`). The block declared one, and the
# one below was it. Review finding F1 at `reviews/glm-5.3-flash-port/bless-103-code-6874a0f5-findings.md`
# measured what that leaves unread: this item pins `select_k == 2`, where the vendored selector takes
# the NON-striking `max8` + `nc_find_index8` branch, so the parent commit `c81113a2` -- whose dispatch
# site gathered the bounded scores instead of reading the selector's values -- passes every test the
# range shipped. The lead ruled a second model-level item at a STRIKING `select_k`
# (`approvals/LEAD-LOG.md` §750 ruling 2, §752); it is
# :func:`test_forward_at_a_striking_select_k_sentinelises_every_pad_the_selector_returns` at the end of
# this block. The plan revision that records two items is the lead's write, not this file's claim.
#
# THE FIGURES THIS ITEM READS WERE CORRECTED BEFORE IT WAS WRITTEN. The plan's first spelling of
# them at rev 256 named rows 4-7 as the rows reading one sentinel. That does not follow from the
# block's own predicate, `causal_len = position + 1` -- which is upstream's formula and which this
# file's own fixtures already use, `seq_lens = arange(1, n + 1)`. Under it the rows reading one are
# 3, 4, 5 and 6. Ruled at DECISIONS section 332, plan rev 257: ":889 becomes 'rows 3-6 one each'".
# CARRIED FORWARD UNCHANGED to the 32-token case at plan rev 260 (entry `design-20260907-bd`): rows 3
# to 6 still read one, rows 0 to 2 still read two, and the rows the longer case adds -- 7 through 31 --
# all read zero, because every one of them completes at least `select_k` pools. The ruling's own words
# still describe the vector; the case moved because the selector could not run the shorter one.
# NOTHING HERE IS TYPED FROM THAT RULING: the vector is computed from this item's own inputs by
# :func:`_expected_sentinel_counts`, and the superseded reading is PLANTED as the control, so an
# implementation that used `position` instead of `position + 1` fails this item rather than passing
# it.

#: The 8 fixed lanes ``nisa.max8`` emits. THIS FILE ALREADY ARGUES THIS FLOOR at :138-166, where it
#: is why ``PREFILL_TOKENS`` is 35 and not 19; it is named here so the tiny case and the boundary case
#: below cite ONE floor instead of two, and so the reason a shorter case cannot run is readable in the
#: file rather than only in a run record.
MAX8_LANES = 8

#: The shortest length that SELECTS at this file's dials AND that the selector can actually run.
#: DERIVED from the dials and the lane floor, so a dial change moves the case instead of quietly
#: turning it back into a bypass case that would read every counter below as a zero and pass for the
#: wrong reason.
#:
#: WHY 32 AND NOT 12, which is what this constant read until plan revision 260. Twelve is one pool
#: past the bypass boundary and is the shorter, sharper case -- and it CANNOT EXECUTE. At
#: ``POOL_SIZE`` 4 it gives 3 candidate pools, and the vendored selector asserts at least 8 on the
#: non-partition axis: ``rotational_topk.py:180`` -> ``rotational_topk_utils.py:994``
#: ``naive_scanning_topk`` -> ``:1036`` ``nisa.max8`` -> ``nki/isa/_validation.py:1608``
#: ``assert n >= 8``. Measured on the host, not inferred: the item raised
#: ``AssertionError: max8 requires at least 8 elements per partition, got 3`` before it reached its
#: own subject. 32 tokens give 8 pools, so one ``nisa.max8`` can run, and 32 is the SMALLEST length
#: that clears the floor -- 28 gives 7 and misses it by one.
#:
#: THE CLAIM DID NOT MOVE WITH THE CASE. The formula below is unchanged, the vector is still computed
#: from this item's own inputs, its first eight entries are what the 12-token case read, every row 32
#: adds is a zero, and the superseded control still differs at rows 3 and 7. Ruled at plan revision
#: 260, entry ``design-20260907-bd``.
SELECTING_SEQ_LEN = POOL_SIZE * max(MAX8_LANES, TOPK_POOLS + 1)

#: The families the SELECTING decision governs, each owing exactly one dispatch on one prefill leg.
#: `causal_fill` is deliberately NOT here: it is inc-glm53f-099's bypass seam and owes a ZERO on
#: this case, which is a different claim and is read as such below.
SELECTING_ONE_FAMILIES: tuple[str, ...] = (
    "paged_gather", "score_gemm", "topk_select", "index_expand",
)


def _causal_bound_apis():
    """``{"bound": (reset, read), "sentinel": (reset, read)}`` for inc-glm53f-103's two seams.

    A SIBLING OF :func:`_discover_counter_api`, NOT A SECOND CONVENTION. Same suffix rule, same
    `reset_` prefix rule; the only difference is the arity it admits. It exists because
    `causal_bound.py` carries TWO pairs -- the block requires "accessors and reset per entry point"
    so that "exactly 1 per call, per entry point" is readable at all, where one summed pair could not
    tell a bound call and a sentinel call apart from two bound calls.

    THE SINGLE-PAIR HELPER'S REFUSAL IS ASSERTED HERE rather than described, because that helper
    documents itself as failing loudly "if a module ever grows a second pair" and this is the first
    module that has. If it ever stopped refusing, this rule would be redundant and a reader should
    find that out from a failure, not from reading two helpers side by side.
    """
    module = _seam_module("causal_bound")
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    resets = sorted(n for n in names if n.startswith("reset_"))
    reads = sorted(n for n in names if not n.startswith("reset_"))
    assert len(resets) == 2 and len(reads) == 2, (module.__name__, resets, reads)
    with pytest.raises(AssertionError):
        _discover_counter_api(module)
    out = {}
    for key, token in (("bound", "causal_bound"), ("sentinel", "causal_sentinel")):
        reset = [n for n in resets if token in n]
        read = [n for n in reads if token in n]
        assert len(reset) == 1 and len(read) == 1, (key, reset, read)
        out[key] = (getattr(module, reset[0]), getattr(module, read[0]))
    return out


def _expected_sentinel_counts(seq_lens: torch.Tensor, select_k: int, pool: int,
                              width: int, *, offset: int = 0) -> list[int]:
    """Sentinels per row, COMPUTED from the predicate, the dials and this case's own lengths.

    ``offset`` exists ONLY so the superseded reading can be produced by the same function as the
    ruled one: ``offset=0`` is ``causal_len = seq_len = position + 1``, the block's predicate and
    upstream's; ``offset=-1`` is ``causal_len = position``, the reading DECISIONS section 332 struck.
    Producing both from one function is the point -- if the control came from a second expression the
    two could drift and the control would stop being a control.

    A row completes ``causal_len // pool`` pools, capped by the candidate width, and the selector
    takes ``select_k`` of them, so the selections that reached no complete pool number
    ``max(0, select_k - min(complete, width))``.
    """
    return [
        max(0, int(select_k) - min((int(s) + int(offset)) // int(pool), int(width)))
        for s in seq_lens
    ]


def _selecting_operands(cfg, *, seed: int, tokens: int = SELECTING_SEQ_LEN) -> dict:
    """Operands for ``forward``'s prefill leg one pool ABOVE the bypass boundary.

    ``seq_lens = arange(1, n + 1)`` is this file's own prefill convention, unchanged -- which is also
    the evidence that ``causal_len`` is ``position + 1`` here and not ``position``.

    ``tokens`` DEFAULTS TO THE DECLARED CASE and is a parameter only because the striking-``select_k``
    item below needs a longer leg for the same shape of operand. It is one length, used for the row
    count, the lengths and the slot mapping alike, so the two legs cannot disagree about it.
    """
    gen = torch.Generator().manual_seed(int(seed))
    rows = int(tokens)
    return {
        "hidden": torch.randn(
            rows, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            rows, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.arange(1, rows + 1, dtype=torch.int32),
        "slot_mapping": prefill_slot_mapping(rows, int(cfg.index_kpool)),
    }


def test_forward_BOUNDS_the_selecting_regime_to_each_rows_own_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward`` above the bypass bound: every row selects only pools it completes.

    FOUR READINGS, and they fail for four different reasons.
      1. THE REGIME SELECTS. Read first, because every reading below is about the selecting chain and
         would pass vacuously on the bypass -- where the two new counters, like every other one,
         read zero.
      2. THE POOL IDS CARRY THE RIGHT SENTINELS, per row, against a vector computed from this case's
         own lengths and dials. This is the block's declared figure and the reason the item exists.
      3. THE SUPERSEDED PREDICATE FAILS. The struck reading (`causal_len = position`) is computed by
         the same function and asserted to DISAGREE, naming the rows where it differs -- so a
         regression to it fails here instead of passing.
      4. THE ROUTE RAN IN NKI. Each of this block's two entry points reads exactly one dispatch and
         no fallback, `-047`'s selector reads exactly one, and `-099`'s bypass seam reads ZERO --
         which is what says the two regimes are exclusive rather than both firing.

    Certifying component (D1.4): `Glm5NextDSAIndexer.select_bounded_pools`, composing
    `dsa_causal_bound`, the landed `dsa_topk_select` and `dsa_causal_sentinel`.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    select_k = int(indexer.select_k())

    # 1. THE REGIME SELECTS, read from the closed form before anything else.
    candidates = SELECTING_SEQ_LEN // pool
    say("D4", "regime", "seq_len", SELECTING_SEQ_LEN, "candidates", candidates,
        "select_k", select_k, "bypass_bound", BYPASS_SEQ_LEN)
    assert candidates > select_k, (
        f"{SELECTING_SEQ_LEN} token(s) yields {candidates} complete pool(s) against select_k="
        f"{select_k}; this item must sit ABOVE the strict bound or the bypass serves it and every "
        f"counter below reads zero for a reason that has nothing to do with the causal bound"
    )
    # THE LANE FLOOR, READ BEFORE THE CASE IDENTITY. This is the reading whose ABSENCE let the
    # 12-token case be ruled and land: the item asserted `candidates > select_k` and never that the
    # selector could run the width it was handed, so it raised inside `nisa.max8` before reaching its
    # own subject and reported nothing about the causal bound at all.
    assert candidates >= MAX8_LANES, (
        f"{SELECTING_SEQ_LEN} token(s) yields {candidates} candidate pool(s), under the "
        f"{MAX8_LANES} lanes nisa.max8 emits; the selector refuses with 'max8 requires at least 8 "
        f"elements per partition' (rotational_topk_utils.py:1036 -> nki/isa/_validation.py:1608) and "
        f"this item never reaches the bound it exists to read"
    )
    assert SELECTING_SEQ_LEN == 32, (
        f"the block declares a 32-token case and this file's dials now derive "
        f"{SELECTING_SEQ_LEN}; the figures below are computed, but the case identity is declared"
    )
    assert select_k == TOPK_POOLS == 2 and pool == POOL_SIZE == 4, (select_k, pool)

    bound_api, sentinel_api = _causal_bound_apis()["bound"], _causal_bound_apis()["sentinel"]
    fill_reset, fill_read = _causal_fill_api()
    ops = _selecting_operands(cfg, seed=9_103_001)

    # The pool ids are the block's subject and `forward` returns the EXPANDED token indices, so the
    # sentinel seam's own output is recorded as it passes. The recorder DELEGATES to the real seam,
    # so it moves no counter of its own -- which the (1, 0) reading below then confirms.
    recorded: list[torch.Tensor] = []
    recorded_widths: list[int] = []
    causal_bound_mod = _seam_module("causal_bound")
    real_sentinel = causal_bound_mod.dsa_causal_sentinel

    def recording_sentinel(values, indices, width):
        # `width` IS FORWARDED AND NEVER DEFAULTED. The seam took two operands before `103r5` and
        # takes three now; a spy that swallowed the third would keep passing while the dispatch site
        # stopped screening pads, which is the defect this repair exists to remove.
        out = real_sentinel(values, indices, width)
        recorded.append(out.clone())
        recorded_widths.append(int(width))
        return out

    reset_all_counters()
    bound_api[0]()
    sentinel_api[0]()
    fill_reset()
    spy = SeamSpy()
    spy.install(monkeypatch)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_sentinel", recording_sentinel)
    try:
        got = indexer.forward(
            ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
            max_seq_len=SELECTING_SEQ_LEN, page_size=PAGE_SIZE,
            slot_mapping=ops["slot_mapping"],
        )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    bound_count = tuple(int(v) for v in bound_api[1]())
    sentinel_count = tuple(int(v) for v in sentinel_api[1]())
    fill_count = tuple(int(v) for v in fill_read())

    assert len(recorded) == 1, (
        f"the sentinel seam ran {len(recorded)} time(s) on one prefill leg; the dispatch site "
        f"composes it exactly once"
    )
    # AND THE WIDTH IT WAS GIVEN IS THE ONE THE SELECTOR SAW. `103r5`: the marker screens an index at
    # or past `width`, so a `width` wider than the real pool columns disables that arm silently. The
    # dispatch site reads it off the bounded tensor; this reads that the number arriving is the
    # candidate count and not, say, the pool-cache row count or `select_k`.
    say("D4", "sentinel_widths", recorded_widths, "candidates", candidates)
    assert recorded_widths == [candidates], (
        f"the dispatch site handed the marker width {recorded_widths} where the bounded tensor has "
        f"{candidates} real pool column(s); a wider width turns the pad arm off without failing"
    )
    pool_ids = recorded[0]
    assert pool_ids.dtype == torch.int32, pool_ids.dtype
    assert tuple(pool_ids.shape) == (SELECTING_SEQ_LEN, select_k), tuple(pool_ids.shape)

    # 2. THE POOL IDS CARRY THE RIGHT SENTINELS, per row, computed and not typed.
    want = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates)
    per_row = (pool_ids == -1).sum(dim=1).to(torch.int64).tolist()
    say("D4", "sentinels_per_row", per_row, "computed", want,
        "lengths", ops["seq_lens"].tolist())
    assert per_row == want, (
        f"the bounded chain read {per_row} sentinel(s) per row where the predicate computes {want}"
    )
    ones = [r for r, n in enumerate(want) if n == 1]
    twos = [r for r, n in enumerate(want) if n == 2]
    zeros = [r for r, n in enumerate(want) if n == 0]
    say("D4", "rows_with_two", twos, "rows_with_one", ones, "rows_with_zero", zeros)
    assert twos[0] == 0 and per_row[0] == 2, (twos, per_row[0])
    assert ones == [3, 4, 5, 6], (
        f"the rows reading one sentinel computed to {ones}; DECISIONS section 332 ruled the plan's "
        f"figure to 'rows 3-6 one each' on exactly this arithmetic"
    )
    assert per_row[-1] == 0 and (SELECTING_SEQ_LEN - 1) in zeros, (per_row[-1], zeros)

    # EVERY NON-SENTINEL POOL ID IS ONE THE ROW COMPLETES, which is inc-glm53f-048's precondition
    # restated per row -- the property the whole increment exists to restore, read directly.
    complete = [min(int(s) // pool, candidates) for s in ops["seq_lens"]]
    illegal = [
        (r, int(p)) for r in range(SELECTING_SEQ_LEN) for p in pool_ids[r]
        if int(p) != -1 and not (0 <= int(p) < complete[r])
    ]
    say("D4", "illegal_pool_ids", len(illegal), "detail", illegal, "complete_pools", complete)
    assert illegal == [], f"a row selected a pool it does not complete: {illegal}"

    # 3. THE SUPERSEDED PREDICATE FAILS, from the same function at offset -1.
    struck = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates, offset=-1)
    differing = [r for r in range(SELECTING_SEQ_LEN) if struck[r] != want[r]]
    say("D4", "struck_predicate", struck, "differs_at_rows", differing)
    assert struck != want, (
        "the struck reading (causal_len = position) agrees with the ruled one on this case, so this "
        "item cannot tell them apart and the control is not a control"
    )
    assert differing == [3, 7], (
        f"the two predicates differ at rows {differing}; on a 32-token case at these dials they "
        f"differ at exactly rows 3 and 7 -- the two pool boundaries the off-by-one moves -- which is "
        f"what makes this control able to fire. The wider case did not weaken it: the width cap does "
        f"not reach these rows, so the differing set is the one the 12-token case read"
    )
    assert per_row != struck, (
        f"the chain reproduced the STRUCK predicate {struck} rather than the ruled one {want}: the "
        f"dispatch site is passing position where it owes position + 1"
    )

    # 4. THE ROUTE RAN IN NKI, form R-1, per entry point.
    for family in FAMILIES:
        say("D4", "counter", family, readings[family])
    say("D4", "causal_bound", bound_count, "causal_sentinel", sentinel_count,
        "causal_fill_099", fill_count)
    assert bound_count == (1, 0), (
        f"the bound seam read {bound_count} and owes exactly one NKI dispatch with no fallback; a "
        f"torch masked_fill would read (0, 1) here and P13 forbids it"
    )
    assert sentinel_count == (1, 0), (
        f"the sentinel seam read {sentinel_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert fill_count == (0, 0), (
        f"inc-glm53f-099's bypass seam read {fill_count} on a SELECTING case; the two regimes are "
        f"exclusive and a non-zero here means both fired"
    )
    for family in SELECTING_ONE_FAMILIES:
        assert readings[family] == (1, 0), (
            f"{family} read {readings[family]} where the selecting chain owes exactly one dispatch "
            f"and no fallback on one prefill leg"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the selecting path: {readings}"
    )
    control = readings["kpool_hadamard"]
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the counter "
        f"instrument is not reading this call and the zeros above measure nothing"
    )
    spy.report("D4")
    _check_two_instruments_agree("D4", spy, readings)
    identity = causal_bound_mod.causal_bound_kernel_identity()
    sentinel_identity = causal_bound_mod.causal_sentinel_kernel_identity()
    say("D4", "kernel", identity, "sentinel_kernel", sentinel_identity)
    assert identity is not None and identity[1].endswith("_causal_bound_nki"), identity
    assert sentinel_identity is not None and sentinel_identity[1].endswith(
        "_causal_sentinel_nki"
    ), sentinel_identity

    # AND THE EXPANSION STILL EMITS THE SHAPE EVERY CONSUMER READS, unchanged by the bound.
    width = _bypass_width(indexer, pool)
    say("D4", "shape", tuple(got.shape), "want", (SELECTING_SEQ_LEN, width),
        "dtype", str(got.dtype))
    assert got.dtype == torch.int32, got.dtype
    assert tuple(got.shape) == (SELECTING_SEQ_LEN, width), tuple(got.shape)
    live = got >= 0
    limit = ops["seq_lens"].to(torch.int64).reshape(SELECTING_SEQ_LEN, 1).expand_as(got)
    out_of_row = int((live & (got.to(torch.int64) >= limit)).sum())
    say("D4", "live_tokens", int(live.sum()), "out_of_row", out_of_row,
        "max_token", int(got.max()))
    assert out_of_row == 0, (
        f"{out_of_row} live token index(es) point past their own row's causal length, which is the "
        f"defect the bound removes seen at the token level"
    )


# =========================================================================== #
# THE SAME BOUND, AT A `select_k` WHERE THE SELECTOR STRIKES AND PADS -- repair `103r5`.
#
# WHAT THIS ITEM ADDS, in one sentence. The item above runs the dispatch site at `select_k = 2`, where
# the vendored selector takes its non-striking, non-folding branch; this one runs the SAME dispatch
# site at a `select_k` where the selector folds its input, PADS the short fold with a finite value,
# strikes what it takes, and moves selected values across partitions with a matmul -- the four
# behaviours the `103r5` repair exists to survive.
#
# WHY IT IS A SEPARATE ITEM AND NOT A PARAMETRISATION of the one above. The two read different
# claims. That one reads the PREDICATE (`causal_len = position + 1`) against a superseded control and
# owns this file's declared 32-token case; this one reads the MARKER (both arms) against the selector's
# own returned pads. Parametrising would make one failure message answer for two claims.
#
# EVERY NUMBER BELOW IS DERIVED OR READ, AND THE PREMISES ARE READ OFF THE KERNEL'S OWN CONFIG. The
# case needs three properties, and no comment in this file can promise them -- the kernel's factories
# decide them from `(rows, width, k, dtype)`. So the item asks
# `create_rotational_topk_config` (through the seam's own `_nki_config`) for `n_stages`,
# `local_top_k_per_stage` and `padded_vocab_size`, and a failure of any premise is a DIAL finding for
# the lead, not a finding about the module under test. That is the shape review finding R2 asked for:
# the earlier `k = 16` case NAMED a branch instead of reading one.

#: The striking case's ``select_k``, and why it is 16 rather than 8 or 2.
#:
#: `topk_core` strikes each value it takes -- `nc_match_replace8` with `imm=float("-inf")` -- on every
#: fold whose `k` is a whole number of 8 (`rotational_topk_utils.py:1053-1066`), and takes the
#: non-writing `max8` + `nc_find_index8` path only on a last fold with `k % 8 != 0` (`:1028-1039`).
#: WHICH `k` REACHES `topk_core` DEPENDS ON THE STAGE COUNT, and that is the part the earlier round of
#: this increment got wrong: at `n_stages == 1` the kernel calls `naive_scanning_topk` with the
#: ORIGINAL `k` (`rotational_topk.py:163-183`, `rotational_topk_utils.py:994`), and at `n_stages >= 2`
#: it calls `topk_core` per stage with `k = local_top_k_per_stage`, which the factory always aligns UP
#: to 8 (`:428-430`) -- so the rotational path always strikes.
#:
#: 8 WOULD STRIKE AND WOULD STILL PROVE NOTHING. The factory's stage count is
#: `div_ceil(min(k, vocab), 8)` (`:417`), so `k = 8` gives ONE stage: no folding, hence no pad column,
#: and no rotation, hence no matmul.
#:
#: 16 IS NOT THE SMALLEST SUCH `k`, AND THIS FILE WILL NOT CLAIM IT IS. That claim is exactly what
#: review finding R2 struck from the sibling test file, and the arithmetic here says the same thing:
#: the stage count reaches 2 as soon as `min(k, vocab) >= 9`, so `k = 9` over a width of 11 is already
#: rotational, already striking (`local_top_k_per_stage` aligns up to 8), and already padded. It would
#: run on a 44-token leg instead of a 68-token one.
#:
#: 16 IS CHOSEN FOR A DIFFERENT AND SMALLER REASON: it is the smallest ROTATIONAL `k` for which
#: `padded_k == orig_k` -- `local_top_k_per_stage * n_stages` is `8 * 2` (`:428-431`) -- which is also
#: true of production's `k = 512`. (`k = 8` also has `padded_k == orig_k` and also strikes, but on the
#: scanning branch, which is the branch this case exists to leave.) At `k = 9` the kernel would return its padded 16 and trim, adding one
#: behaviour between the selector and the marker that production does not have. The cost is 24 extra
#: token rows, and every property the case needs is READ from the config below either way, so a future
#: lap may move this dial down without touching a single reading.
STRIKING_SELECT_K = 16

#: The candidate width: strictly above ``select_k`` -- `can_run_dsa_topk_select` refuses `k == width`
#: (`topk_select.py:294`) -- and ODD, so the fold cannot divide it and a pad column must exist. The
#: pad's own existence is read from `padded_vocab_size` below, not from this comment.
STRIKING_CANDIDATES = STRIKING_SELECT_K + 1

#: The prefill length that yields those candidates, and the ``index_topk`` that yields that
#: ``select_k``. Both derived: ``select_k()`` is ``index_topk // index_kpool``
#: (``model_fp8.py:3712``), and ``candidates`` is ``max_seq_len // index_kpool``
#: (``model_fp8.py:4096``).
STRIKING_SEQ_LEN = STRIKING_CANDIDATES * POOL_SIZE
STRIKING_INDEX_TOPK = STRIKING_SELECT_K * POOL_SIZE


def test_forward_at_a_striking_select_k_sentinelises_every_pad_the_selector_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward`` at a striking ``select_k``: no pad column survives into a row's pool ids.

    SEVEN READINGS, each failing for its own reason.
      1. THE CASE IS THE ONE IT CLAIMS -- dials first, from the built config.
      2. THE GEOMETRY IS THE STRIKING, FOLDING, PADDING ONE -- read off the kernel's own factories:
         at least two stages, a per-stage ``k`` that is a whole number of 8, and at least one pad
         column. A failure here is a DIAL finding: the case did not reach its own subject.
      3. THE SELECTOR REALLY RETURNED A PAD, and the pad's VALUE is above the marker's threshold --
         the two facts that make the index arm load-bearing rather than decorative. Measured from the
         operands the seam was handed, not argued.
      4. NO SELECTED VALUE IS NaN. The finite ``BOUND_FILL`` exists so that the rotation matmul --
         ``nisa.nc_matmul`` with a 0/1 permutation, where ``0 * -inf`` is NaN -- has no infinity to
         destroy. A NaN here is a reading about the selector for the lead, and the record
         (``increments/contradiction-103-selector-pad-6874a0f5.md``) already names it as unmeasured.
      5. THE OUTPUT IS LEGAL: every id a row keeps is a pool that row completes, and no id is at or
         past the width. This is the reading the parent commit fails.
      6. THE ONE-ARM MARKER DISAGREES, computed from this run's own recorded values and indices. It is
         asserted RED, so the pad arm is known to be doing work on this case rather than assumed to.
      7. THE ROUTE RAN IN NKI, per entry point, with no torch fallback anywhere.

    Certifying component (D1.4): ``Glm5NextDSAIndexer.select_bounded_pools`` -- the same component the
    item above certifies, at the branch that item cannot reach (review finding F1).
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    fill, mark = _fill_constants()
    stack, cfg, _gen = build_layer_stack(
        layers=1, seed=9_103_016, index_topk=STRIKING_INDEX_TOPK
    )
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    select_k = int(indexer.select_k())
    candidates = STRIKING_SEQ_LEN // pool

    # 1. THE CASE IS THE ONE IT CLAIMS. Read before the run: a wrong dial here spends nothing.
    say("D5", "seq_len", STRIKING_SEQ_LEN, "candidates", candidates, "select_k", select_k,
        "index_topk", int(cfg.index_topk), "fill", fill, "mark", mark)
    assert select_k == STRIKING_SELECT_K, (
        f"the config built select_k={select_k} where this case declares {STRIKING_SELECT_K}; "
        f"index_topk={int(cfg.index_topk)} over index_kpool={pool} is the only route to it"
    )
    assert candidates == STRIKING_CANDIDATES and candidates > select_k, (
        f"{STRIKING_SEQ_LEN} token(s) yields {candidates} candidate pool(s) against select_k="
        f"{select_k}; the selector refuses k == width and the bypass serves anything below it"
    )
    assert candidates >= MAX8_LANES, (candidates, MAX8_LANES)

    bound_api, sentinel_api = _causal_bound_apis()["bound"], _causal_bound_apis()["sentinel"]
    fill_reset, fill_read = _causal_fill_api()
    ops = _selecting_operands(cfg, seed=9_103_016, tokens=STRIKING_SEQ_LEN)

    # BOTH SEAMS ARE RECORDED, and each recorder DELEGATES to the real seam, so it moves no counter of
    # its own -- which the (1, 0) readings below then confirm. The bound's output is recorded because
    # it is the exact tensor the selector was handed: the width, the row count and the dtype the
    # kernel's config is built from all come from it rather than from a number typed here.
    bounded_seen: list[torch.Tensor] = []
    values_seen: list[torch.Tensor] = []
    indices_seen: list[torch.Tensor] = []
    widths_seen: list[int] = []
    marked_seen: list[torch.Tensor] = []
    causal_bound_mod = _seam_module("causal_bound")
    real_bound = causal_bound_mod.dsa_causal_bound
    real_sentinel = causal_bound_mod.dsa_causal_sentinel

    def recording_bound(scores, causal_len, pool_size):
        out = real_bound(scores, causal_len, pool_size)
        bounded_seen.append(out.clone())
        return out

    def recording_sentinel(values, indices, width):
        values_seen.append(values.clone())
        indices_seen.append(indices.clone())
        widths_seen.append(int(width))
        out = real_sentinel(values, indices, width)
        marked_seen.append(out.clone())
        return out

    reset_all_counters()
    bound_api[0]()
    sentinel_api[0]()
    fill_reset()
    spy = SeamSpy()
    spy.install(monkeypatch)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_bound", recording_bound)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_sentinel", recording_sentinel)
    try:
        got = indexer.forward(
            ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
            max_seq_len=STRIKING_SEQ_LEN, page_size=PAGE_SIZE,
            slot_mapping=ops["slot_mapping"],
        )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    bound_count = tuple(int(v) for v in bound_api[1]())
    sentinel_count = tuple(int(v) for v in sentinel_api[1]())
    fill_count = tuple(int(v) for v in fill_read())

    assert len(bounded_seen) == len(marked_seen) == 1, (len(bounded_seen), len(marked_seen))
    bounded, values, raw_ids, pool_ids = (
        bounded_seen[0], values_seen[0], indices_seen[0], marked_seen[0]
    )
    say("D5", "bounded", tuple(bounded.shape), str(bounded.dtype), "values",
        tuple(values.shape), str(values.dtype), "widths", widths_seen)
    assert tuple(bounded.shape) == (STRIKING_SEQ_LEN, candidates), tuple(bounded.shape)
    assert widths_seen == [candidates], (
        f"the marker was handed width {widths_seen} where the tensor the selector saw has "
        f"{candidates} real column(s); a wider width switches the index arm off silently"
    )
    assert tuple(pool_ids.shape) == (STRIKING_SEQ_LEN, select_k), tuple(pool_ids.shape)
    assert pool_ids.dtype == torch.int32, pool_ids.dtype

    # 2. THE GEOMETRY IS THE STRIKING, FOLDING, PADDING ONE, asked of the kernel's own factories with
    # the run's own operands. A failure of any of these three is a DIAL finding.
    from vllm_neuron.functional.dsa.topk_select import _nki_config, _nki_dtype_of

    kernel_cfg = _nki_config(
        int(bounded.shape[0]), int(bounded.shape[1]), select_k, _nki_dtype_of(bounded)
    )
    n_stages = int(kernel_cfg.n_stages)
    local_k = int(kernel_cfg.local_top_k_per_stage)
    pad_columns = int(kernel_cfg.padded_vocab_size) - int(kernel_cfg.vocab_size)
    say("D5", "n_stages", n_stages, "local_top_k_per_stage", local_k, "stage_free_size",
        int(kernel_cfg.stage_free_size), "vocab", int(kernel_cfg.vocab_size),
        "padded_vocab", int(kernel_cfg.padded_vocab_size), "pad_columns", pad_columns)
    assert int(kernel_cfg.vocab_size) == candidates, (int(kernel_cfg.vocab_size), candidates)
    assert n_stages >= 2, (
        f"the factory chose n_stages={n_stages}, so the kernel takes naive_scanning_topk with the "
        f"ORIGINAL k and neither folds nor rotates; this case then reads no pad and no matmul. A "
        f"DIAL finding: raise select_k or the width until the factory chooses two stages"
    )
    assert local_k % MAX8_LANES == 0, (
        f"the per-stage k is {local_k}, not a whole number of {MAX8_LANES}, so topk_core takes the "
        f"non-striking max8 + nc_find_index8 branch and this case is the item above with a longer leg"
    )
    assert pad_columns >= 1, (
        f"padded_vocab_size equals vocab_size, so the fold divides the width evenly and the selector "
        f"pads NOTHING; the index arm cannot be read on this case. A DIAL finding: the width must not "
        f"be a multiple of n_stages={n_stages}"
    )

    # 3. THE SELECTOR REALLY RETURNED A PAD, AND ITS VALUE IS ABOVE THE MARK. The first fact makes the
    # index arm reachable; the second is why the value arm alone cannot stand in for it.
    out_of_range = raw_ids.to(torch.int64) >= candidates
    pad_slots = int(out_of_range.sum())
    pad_values = values[out_of_range]
    distinct = sorted({float(v) for v in pad_values.to(torch.float32).flatten()})
    say("D5", "out_of_range_selections", pad_slots, "rows_touched",
        int(out_of_range.any(dim=1).sum()), "max_id", int(raw_ids.max()),
        "pad_values", distinct[:8], "pad_value_count", len(distinct))
    assert pad_slots >= 1, (
        f"the selector returned no index at or past the width on a geometry the factory says has "
        f"{pad_columns} pad column(s), so this case does not read the index arm at all. A DIAL "
        f"finding about the case, not a pass"
    )
    assert bool((pad_values > mark).all()), (
        f"every out-of-range slot came back at or below the mark {mark}, so the VALUE arm alone would "
        f"have caught them and this case still does not read the index arm. The pad values seen were "
        f"{distinct[:8]}"
    )

    # 4. NO SELECTED VALUE IS NaN, which is what the finite fill was chosen for.
    nan_slots = int(torch.isnan(values).sum())
    say("D5", "nan_values", nan_slots, "min_value", float(values.min()),
        "values_at_or_below_mark", int(torch.le(values, mark).sum()))
    assert nan_slots == 0, (
        f"{nan_slots} selected value(s) came back NaN. The bound writes the FINITE {fill} so that the "
        f"rotation matmul has no infinity to turn into one; a NaN here is a reading about the "
        f"selector for the lead -- the marker still fires on it, so the ids stay legal, but the "
        f"per-row count below is then measuring something this case did not declare"
    )

    # 5. THE OUTPUT IS LEGAL, which is the reading the parent commit fails.
    complete = [min(int(s) // pool, candidates) for s in ops["seq_lens"]]
    illegal = [
        (r, int(p)) for r in range(STRIKING_SEQ_LEN) for p in pool_ids[r]
        if int(p) != -1 and not (0 <= int(p) < complete[r])
    ]
    say("D5", "illegal_pool_ids", len(illegal), "detail", illegal[:12],
        "still_out_of_range", int((pool_ids.to(torch.int64) >= candidates).sum()))
    assert int((pool_ids.to(torch.int64) >= candidates).sum()) == 0, (
        f"a pad column survived into the pool ids: {illegal[:12]}. This is the `103r5` defect at the "
        f"dispatch site -- a finite pad outranks a bounded column, so it wins a slot, and only the "
        f"index arm can tell it from a real selection"
    )
    assert illegal == [], f"a row selected a pool it does not complete: {illegal[:12]}"

    want = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates)
    per_row = (pool_ids == -1).sum(dim=1).to(torch.int64).tolist()
    say("D5", "sentinels_per_row", per_row, "computed", want)
    assert per_row == want, (
        f"the bounded chain read {per_row} sentinel(s) per row where the predicate computes {want}. "
        f"Fewer means a fill or a pad survived; more means a slot was marked that carried a real "
        f"selection -- a NaN value is the one known way that happens and reading 4 covers it"
    )
    assert want[0] == select_k and want[-1] == 0, (
        f"the computed vector is {want[:3]}..{want[-3:]}; row 0 completes no pool so it owes "
        f"{select_k} sentinels and the longest row completes {complete[-1]} so it owes none. Without "
        f"both ends this case reads one regime only"
    )

    # 6. THE ONE-ARM MARKER DISAGREES, computed from this run's own recorded operands. Asserted RED:
    # if the value arm alone reached the same answer, the index arm would be decoration on this case.
    value_only = torch.where(
        values > mark, raw_ids, torch.full_like(raw_ids, -1)
    )
    value_only_out_of_range = int((value_only.to(torch.int64) >= candidates).sum())
    value_only_counts = (value_only == -1).sum(dim=1).to(torch.int64).tolist()
    say("D5", "value_only_out_of_range", value_only_out_of_range,
        "value_only_counts", value_only_counts[:8])
    assert value_only_out_of_range >= 1, (
        f"the value arm ALONE left no out-of-range id, so this case cannot tell the two-arm marker "
        f"from the one-arm form the repair replaced"
    )
    assert value_only_counts != per_row, (
        f"the one-arm form read the same per-row counts as the two-arm marker, so the index arm "
        f"changed nothing measurable here"
    )

    # 7. THE ROUTE RAN IN NKI, form R-1, per entry point.
    for family in FAMILIES:
        say("D5", "counter", family, readings[family])
    say("D5", "causal_bound", bound_count, "causal_sentinel", sentinel_count,
        "causal_fill_099", fill_count)
    assert bound_count == (1, 0), (
        f"the bound seam read {bound_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert sentinel_count == (1, 0), (
        f"the sentinel seam read {sentinel_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert fill_count == (0, 0), (
        f"inc-glm53f-099's bypass seam read {fill_count} on a SELECTING case"
    )
    for family in SELECTING_ONE_FAMILIES:
        assert readings[family] == (1, 0), (
            f"{family} read {readings[family]} where the selecting chain owes exactly one dispatch "
            f"and no fallback on one prefill leg"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the selecting path: {readings}"
    )
    control = readings["kpool_hadamard"]
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the counter "
        f"instrument is not reading this call and the zeros above measure nothing"
    )
    spy.report("D5")
    _check_two_instruments_agree("D5", spy, readings)

    # AND THE EXPANSION CARRIES NOTHING PAST A ROW'S OWN LENGTH, the same defect at the token level.
    width = _bypass_width(indexer, pool)
    live = got >= 0
    limit = ops["seq_lens"].to(torch.int64).reshape(STRIKING_SEQ_LEN, 1).expand_as(got)
    out_of_row = int((live & (got.to(torch.int64) >= limit)).sum())
    say("D5", "shape", tuple(got.shape), "want", (STRIKING_SEQ_LEN, width),
        "live_tokens", int(live.sum()), "out_of_row", out_of_row, "max_token", int(got.max()))
    assert got.dtype == torch.int32, got.dtype
    assert tuple(got.shape) == (STRIKING_SEQ_LEN, width), tuple(got.shape)
    assert out_of_row == 0, (
        f"{out_of_row} live token index(es) point past their own row's causal length. A surviving pad "
        f"id expands to tokens at or past {candidates * pool}, which no row of this case can see"
    )
