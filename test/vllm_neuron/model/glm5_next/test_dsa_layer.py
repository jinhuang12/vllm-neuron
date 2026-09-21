"""The DSA decoder layer, its indexer chain and its runner integration.

Numeric agreement against a torch reference, the traced-graph contracts the
indexer's operands have to keep, and the refusals each stage raises by name.
"""

from __future__ import annotations

import functools
import importlib
from collections import Counter

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_mla_decode import paged_operands

# --------------------------------------------------------------------------- #
# the declared geometry. Every value is the checkpoint's or is derived from a gate this file
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

#: The tiny case: eight complete pools plus a tail of ``POOL_SIZE - 1``, so the first decode step
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

#: ``head_size``, derived from the two config fields the way the implementation derives it
#: (``model_fp8.py``) rather than typed, so a config change reaches this file.
TINY_HEAD_SIZE = TINY_GEOMETRY["kv_lora_rank"] + TINY_GEOMETRY["qk_rope_head_dim"]

RTOL = 5e-3
ATOL = 1e-5


# --------------------------------------------------------------------------- #
# the nine entry points and their seven counter FAMILIES.
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

#: Calls per layer, per phase, per entry point in the layer case: ``(prefill, decode)``.
#: Each figure's reason is in a separate measurement; the three zeros are the two
#: phase-exclusive entries plus the pack pair upstream does not reach at batch one.
DECLARED_PER_LAYER: dict[str, tuple[int, int]] = {
    "dsa_decode_tail_update": (0, 1),   # decode-only: the tail ring advances one token per step
    "dsa_kpool_hadamard": (1, 0),       # prefill-only: prefill sees whole pools
    "dsa_hadamard128": (1, 1),          # the indexer query is rotated every step; never pooled
    "dsa_paged_gather": (1, 1),         # the selected rows are read out of paged storage
    "dsa_ragged_pack": (0, 0),          # Declared zero: batch one is uniform, so no padding
    "dsa_ragged_unpack": (0, 0),        # and nothing to unpack
    "dsa_score_gemm": (1, 1),           # the indexer scores candidates
    "dsa_topk_select": (1, 1),          # and selects
    "dsa_index_expand": (1, 1),         # pool ids become token indices
}

#: Calls per layer in the ragged decode arm, one non-uniform decode step, ``(prefill, decode)``.
#: These are ``Glm5NextDSAIndexer.forward_ragged``'s own declared readings, which its
#: docstring states entry point by entry point.
#:
#: They are not upstream's figures. Upstream packs the quantised query, packs the query
#: scale only when one exists, packs the weights, and unpacks once. Two of those three
#: figures do not survive the fork's own gates. The query scale never existed here, which was
#: already known. The weights cannot pack, because they are fp32 by ``dsa_score_gemm``'s gate and
#: ``dsa_ragged_pack`` admits bf16 alone -- so upstream's pair is a single pack here. And the
#: unpack has no consumer, because ``mla_sparse_attention`` refuses a rank-3 index tensor by name.
#: The round-trip test covers that collision.
DECLARED_PER_LAYER_RAGGED_ARM: dict[str, tuple[int, int]] = {
    "dsa_decode_tail_update": (0, 0),   # Declared zero: the arm selects; the ring is the
                                        # uniform case's business and advances there
    "dsa_kpool_hadamard": (0, 0),       # no prefill in the arm, so no whole pools to pool
    "dsa_hadamard128": (0, 1),          # the query is still rotated
    "dsa_paged_gather": (0, 1),
    "dsa_ragged_pack": (0, 1),          # The query alone: the fp32 weights cannot pack
    "dsa_ragged_unpack": (0, 0),        # Declared zero; the round-trip test covers it
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

#: The layer case's declared readings, spelled out so a reader sees them without running the
DECLARED_FAMILY_TOTALS: dict[str, int] = {
    "decode_tail_update": 3,
    "index_expand": 6,
    "kpool_hadamard": 9,
    "paged_gather": 6,
    "ragged_pack": 0,
    "score_gemm": 6,
    "topk_select": 6,
}

#: The ragged arm's declared readings at three layers. Its one-third control is the 1-layer arm.
#: ``kpool_hadamard`` reads 3 on the rotation half alone, which is why the entry-point table above
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
    """``(reset, read)`` on one module, discovered rather than spelled out. """
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    reset = [n for n in names if n.startswith("reset_")]
    read = [n for n in names if not n.startswith("reset_")]
    assert len(reset) == 1 and len(read) == 1, (module.__name__, reset, read)
    return getattr(module, reset[0]), getattr(module, read[0])


def _counter_api(family: str):
    """``(reset, read)`` for one of the seven dsa families."""
    return _discover_counter_api(_seam_module(family))


def reset_all_counters() -> None:
    for family in FAMILIES:
        _counter_api(family)[0]()


def read_all_counters() -> dict[str, tuple[int, int]]:
    """``{family: (nki_dispatch, torch_fallback)}`` since the last reset."""
    return {family: tuple(int(v) for v in _counter_api(family)[1]()) for family in FAMILIES}

#: The module that holds the projection seam. Not under ``functional.dsa``; see above.
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

#: The closed form, term by term: the projection dispatches one layer owes for one phase,
#: each term with the caller that makes it. The total is the sum and is never typed on its own.
#:
#: The same nine arrive from the reference side. The torch reference in this file reaches
#: ``_ref_projection`` nine times per layer per phase over a call graph written independently
#: of the production one: ``_ref_layer`` projects once itself (the indexer's latent), calls
#: ``_ref_indexer`` -> ``_ref_project_stage`` which projects four times, and calls
#: ``_ref_attend`` which projects four times (q_a_proj, q_b_proj, kv_a_proj_with_mqa,
#: o_proj). Production splits that last four as 3 + 1 across ``project_query_and_latent``
#: and ``project_output``, over the same sites.
#:
#: The third term is three and not two because of composition, not arithmetic:
#: ``project_query_and_latent`` calls ``project_query_latent`` as its own first statement, so
#: the nested dispatch belongs to it. Counting the direct ``mla_projection(`` lines in that
#: method reads two, which is the wrong reading.
#:
#: Each row is ``(key, what calls it, dispatches, where the figure comes from)``. The
#: arithmetic below selects on the key, so rewording a label cannot move a number.
DECLARED_PROJECTION_TERMS: tuple[tuple[str, str, int, str], ...] = (
    (
        "layer_query_latent",
        "layer.forward -> attention.project_query_latent",
        1,
        "layer.forward calls project_query_latent, whose one dispatch is q_a_proj",
    ),
    (
        "indexer",
        "indexer.forward -> project_stage",
        4,
        "indexer.forward calls project_stage, whose four dispatches are wq_b, wk, "
        "weights_proj and index_kpool_compress_gate",
    ),
    (
        "attend_query_and_latent",
        "attention.attend -> project_query_and_latent",
        3,
        "attend calls project_query_and_latent: the nested project_query_latent "
        "(q_a_proj) plus q_b_proj plus kv_a_proj_with_mqa",
    ),
    (
        "attend_output",
        "attention.attend -> project_output",
        1,
        "attend calls project_output, whose one dispatch is o_proj",
    ),
)

#: The arm's closed form. It calls the indexer directly, with no layer and no ``attend``
#: involved, so the indexer term is the whole of it; it projects the padded grid once rather
#: than once per request, which is the commutation claim the arm exists to test.
#: The arm builds one layer and calls ``forward_ragged`` once, so the figure it owes is four.
DECLARED_PROJECTION_TERMS_RAGGED_ARM: tuple[tuple[str, str, int, str], ...] = (
    (
        "indexer",
        "indexer.forward_ragged -> project_stage",
        4,
        "indexer.forward_ragged calls project_stage once on the flattened padded grid, "
        "four dispatches as above",
    ),
)


def projection_terms(table: tuple[tuple[str, str, int, str], ...]) -> dict[str, int]:
    """``{key: dispatches}``, and a duplicate key is a failure rather than a silent overwrite."""
    out: dict[str, int] = {}
    for key, _label, count, _cite in table:
        assert key not in out, f"duplicate projection term key {key!r}"
        out[key] = count
    return out

#: Dispatches one completing indexer call owes. Read out of the terms table rather than typed a
#: second time, so the per-call reading and the per-case total rest on one declaration.
PROJECTION_PER_INDEXER_CALL = projection_terms(DECLARED_PROJECTION_TERMS)["indexer"]

PROJECTION_PER_LAYER_PER_PHASE = sum(projection_terms(DECLARED_PROJECTION_TERMS).values())
PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE = (
    PROJECTION_PER_LAYER_PER_PHASE - PROJECTION_PER_INDEXER_CALL
)
PROJECTION_PER_ARM_CASE = sum(projection_terms(DECLARED_PROJECTION_TERMS_RAGGED_ARM).values())

# What the closed form deliberately excludes, named so the total is not silently wrong later.
# ``Glm5NextMLAAttention.project_qkv`` (``model_fp8.py``) holds four more dispatch sites
#  and has no caller anywhere in the tree -- it is dead on every path
# this file exercises, so the layer's reading is 9 and not 13. ``Glm5NextMLAAttention.forward``
# is still a stub (``model_fp8.py``) and is likewise never on the path; that stub is one of
# the arms ``test_kv_spec.py`` keeps. Neither exclusion is asserted by a source scan here: the
# measured per-case total is the guard, because wiring either one in would move it.
#
# What the fallback zero is worth. This family's ``torch_fallback`` cannot be
# incremented by any code path: the module has no torch
# projection route and an inadmissible geometry raises instead (``mla_projections.py``).
# So the zero is a statement that stays true by construction, and it is asserted only because a
# torch route added later would make it a real reading. The reading with teeth is
# ``nki_dispatch``: it falls short if a dispatch is missed, rises if one is added, and falls short
# if a call is served by ``mla_projection_torch_oracle`` (``mla_projections.py``), which moves
# no counter at all.


# --------------------------------------------------------------------------- #
# the test-side call spy.


class SeamSpy:
    """Counts every seam call and attributes it to the layer whose forward is running.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self._open: list[int | None] = []
        # The dtype of each call's first tensor argument. Kept in its own list so `calls`
        # still holds `(entry, layer)` and every count above is computed from that alone.
        self.first_arg_dtypes: list[tuple[str, torch.dtype]] = []
        # The kernel's selected pool ids, per layer, so the reference can adopt them. Its own
        # list for the same reason as above. The
        # tensor is detached and cloned, because the buffer the product returns is free to be reused
        # and a live view would let the implementation edit the reference's input after the fact.
        self.selected_pool_ids: list[tuple[int | None, torch.Tensor]] = []

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
        """Every dtype this entry's first tensor argument arrived as, on the production
        path.
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
        # Record the selected pool ids so the reference can adopt them. This is the method
        # both entry points call, and its return is the final set --
        # bounded, selected, marked and canonically ordered -- which is exactly the set the tie
        # comparison is about. Recorded only; the real value is returned unchanged, so this patch
        # cannot alter what the implementation computes.
        indexer_cls = model_fp8.Glm5NextDSAIndexer
        monkeypatch.setattr(
            indexer_cls,
            "select_bounded_pools",
            self._record_pool_ids(indexer_cls.select_bounded_pools),
        )

    def _record_pool_ids(self, real):
        @functools.wraps(real)
        def wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                self.selected_pool_ids.append((self.current_layer, out.detach().clone()))
            return out

        return wrapper

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
        """Wrap the layer's forward so every seam call lands inside exactly one layer
        bracket.
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
def agreement(spy: SeamSpy, counters: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """``{family: (spy_sum, counter_delta)}`` -- two instruments counting the same events."""
    per_family = spy.per_family()
    return {family: (per_family[family], counters[family][0]) for family in FAMILIES}


# --------------------------------------------------------------------------- #
# the per-indexer-call projection reading.


class IndexerProjectionProbe:
    """Reads the projection counter's delta across every indexer call that completes.
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
        """Every recorded call read exactly ``(4, 0)``, and the call count is itself a reading."""
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
            assert nki == PROJECTION_PER_INDEXER_CALL, (
                f"{label}: indexer call {idx} ({method}) dispatched {nki} projections where "
                f"project_stage owes exactly {PROJECTION_PER_INDEXER_CALL} -- one per site at "
                f"model_fp8.py. A short count means a site was served by "
                f"mla_projection_torch_oracle (which moves no counter) or was not reached at all; "
                f"a long one means a site projects more than once"
            )
            assert fallback == 0, (
                f"{label}: indexer call {idx} ({method}) recorded {fallback} torch fallbacks. This "
                f"counter has no code path that raises it today, "
                f"so a non-zero reading here means a torch projection route was added and the "
                f"expectation has to be re-derived, not this number"
            )


def projection_closed_form(
    label: str, table: tuple[tuple[str, str, int, str], ...], *, layers: int, phases: int
) -> int:
    """Sum the closed form term by term and return the total it
    declares.
    """
    total = 0
    for _key, _what, count, _cite in table:
        owed = count * layers * phases
        total += owed
    return total


# --------------------------------------------------------------------------- #
# the gate precondition.


def gate_live() -> bool:
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return bool(can_run_kernel())


def test_the_declared_table_is_internally_consistent() -> None:
    """The declared readings must cover the census, reach every family, and keep the
    one-third form.
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

    # 7/7 is read over the union, which is what the two-case shape of part 1 is for.
    union_zero = sorted(f for f in FAMILIES if totals[f] == 0 and arm[f] == 0)
    assert union_zero == [], f"families no case reaches: {union_zero}"

    # The layer case's one zero is declared, and naming it here is what stops a silent miss
    # from passing as a declaration later.
    assert sorted(f for f, v in totals.items() if v == 0) == ["ragged_pack"]
    assert arm["ragged_pack"] > 0, "the arm exists to reach the family the layer case declares zero"

    # The one-third moving control holds separately on each case, over the families that case
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

    RULED_DECLARED_ZERO = {
        "dsa_ragged_unpack": (
            "no admissible "
            "payload -- the fp32 gate weights cannot enter a bf16-only pack, and "
            "mla_sparse_attention refuses a rank-3 index tensor by name. The direction's own "
            "coverage is the declared bit-identical round-trip test"
        )
    }
    unreached = sorted(
        e
        for e in ENTRY_POINTS
        if sum(DECLARED_PER_LAYER[e]) + sum(DECLARED_PER_LAYER_RAGGED_ARM[e]) < 1
    )
    assert unreached == sorted(RULED_DECLARED_ZERO), (
        f"entry points no case calls: {unreached}; ruled declared zeros: "
        f"{sorted(RULED_DECLARED_ZERO)}"
    )
    reached = len(ENTRY_POINTS) - len(unreached)
    assert reached == 8, reached

    # The arm's own declared zero is a family, and the union is what covers it: the arm never
    # advances the ring, and the layer case's decode step does it three times.
    assert DECLARED_FAMILY_TOTALS_RAGGED_ARM["decode_tail_update"] == 0
    assert DECLARED_FAMILY_TOTALS["decode_tail_update"] > 0

    # The phase-exclusive pair is declared as such, so a one-phase case cannot pass unnoticed.
    assert DECLARED_PER_LAYER["dsa_decode_tail_update"] == (0, 1)
    assert DECLARED_PER_LAYER["dsa_kpool_hadamard"] == (1, 0)

    # The arm is decode-only, so its prefill column is entirely zero. A stray prefill figure here
    # would mean the arm had quietly become a second layer case.
    assert all(p == 0 for p, _ in DECLARED_PER_LAYER_RAGGED_ARM.values())

    # The projection closed form, checked here as a declaration and measured in the two runs
    # below: nine per layer per phase, four in the indexer's project_stage, one in the layer's
    # own project_query_latent, three inside project_query_and_latent and one in
    # project_output. Editing a term without re-deriving the closed form fails here, before
    # any run reads a counter.
    terms = projection_terms(DECLARED_PROJECTION_TERMS)
    arm_terms = projection_terms(DECLARED_PROJECTION_TERMS_RAGGED_ARM)
    assert sorted(terms) == [
        "attend_output", "attend_query_and_latent", "indexer", "layer_query_latent"
    ], sorted(terms)
    assert PROJECTION_PER_LAYER_PER_PHASE == 9, PROJECTION_PER_LAYER_PER_PHASE
    assert PROJECTION_PER_INDEXER_CALL == 4, PROJECTION_PER_INDEXER_CALL
    assert PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE == 5
    assert len(PHASES) == 2, PHASES

    # The arm's only term is the indexer's, at the same figure, because the arm calls the indexer
    # directly with no layer and no attend around it. So the arm reads four for its one call.
    assert sorted(arm_terms) == ["indexer"], sorted(arm_terms)
    assert arm_terms["indexer"] == PROJECTION_PER_INDEXER_CALL
    assert PROJECTION_PER_ARM_CASE == PROJECTION_PER_INDEXER_CALL


# --------------------------------------------------------------------------- #
# the refusals, and why they are tests rather than defensive habits.
#
# Each refusal below is a design decision the lead ruled, so each one is worth a test that
# fails if the refusal is ever softened into a branch. Two of them share one property that
# the tests assert directly: they fire before the indexer touches a weight. That is what
# lets an unmaterialised indexer exercise them, and it is also what stops a load error from
# masking a config error.
#
# These tests are not among the measured run. Each one resets the counters, expects a raise,
# and then asserts every counter still reads zero -- a refusal that dispatched first would be
# a refusal that already did the wrong thing.

ENTRY_METHODS: tuple[str, ...] = ("forward", "forward_ragged")


def _impl():
    """Import the implementation module inside a test body, never at import. """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _tiny_text_config(**overrides):
    """The checkpoint's own config, narrowed to :data:`TINY_GEOMETRY` and the tiny
    ``select_k``.
    """
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    fields = dict(TINY_GEOMETRY)
    fields["index_topk"] = TOPK_POOLS * POOL_SIZE
    fields.update(overrides)
    return replace(Glm5NextTextConfig(), **fields)


def _bare_indexer(**overrides):
    """An indexer with no weight materialised, which is enough to reach both preconditions."""
    return _impl().Glm5NextDSAIndexer(_tiny_text_config(**overrides))


def _reach(indexer, method: str, *, max_seq_len: int, pool_rows: int | None = None):
    """Call one entry point with operands shaped to reach its preconditions and no further.
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
    """Run one entry point, require a named refusal, and require that nothing dispatched.
    """
    reset_all_counters()
    reset_projection_counter()
    with pytest.raises(_impl().Glm5NextDSAIndexerError) as caught:
        _reach(indexer, method, max_seq_len=max_seq_len, pool_rows=pool_rows)
    readings = read_all_counters()
    projection = read_projection_counter()
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
    """The precondition: both dials are read, and neither is a switch. """
    indexer = _bare_indexer(**{dial: False})
    assert getattr(indexer, dial) is False
    message = _refuses(method, indexer, max_seq_len=PREFILL_TOKENS)
    assert dial in message, f"the refusal must NAME the dial it refuses; got: {message}"
    assert "glm5_next.py" in message, "and cite the upstream refusal it mirrors"


def test_the_two_entry_points_refuse_in_the_same_order() -> None:
    """A config that breaks both preconditions must report the dial, on both entry points.
    """
    # Above the bypass bound, so the regime still selects and the trash row is still addressed:
    # This test is about refusal order, not about the short regime.
    long_enough = PREFILL_TOKENS
    assert long_enough // POOL_SIZE > TOPK_POOLS, (
        "the ordering case must sit above the strict selection bound, or the indexer would take "
        "the bypass and never reach the trash-row check at all"
    )
    # Fewer pool_cache rows than there are addressable candidate pools, so `candidates > trash`.
    starved_rows = 2
    assert long_enough // POOL_SIZE > starved_rows - 1, (
        f"{starved_rows} pool_cache row(s) must leave no trash row above the "
        f"{long_enough // POOL_SIZE} candidate pool(s), or case B refuses nothing"
    )
    TRASH_WORDING = "leaves no trash row above"

    for method in ENTRY_METHODS:
        # Case A: both faults present. The dial must be the one reported.
        both_broken = _bare_indexer(index_kpool_compress=False)
        message = _refuses(
            method, both_broken, max_seq_len=long_enough, pool_rows=starved_rows
        )
        assert "index_kpool_compress" in message
        assert TRASH_WORDING not in message, (
            "the dial must be reported first: it is the cheaper, earlier and more likely "
            "operator error, and reporting the pool_cache geometry would send a reader to "
            "resize a cache that is not the problem"
        )

        # Case B, the control: the dial repaired, the same starved cache. The other refusal
        # must now appear, or case A's absence measured nothing.
        dial_ok = _bare_indexer()
        control = _refuses(
            method, dial_ok, max_seq_len=long_enough, pool_rows=starved_rows
        )
        assert TRASH_WORDING in control, (
            f"the trash-row refusal did not fire on {starved_rows} row(s) with the dial "
            f"repaired, so case A's absence of that wording is not a reading. Got: {control}"
        )
        assert "index_kpool_compress" not in control


def test_the_indexer_hands_the_pooling_seam_bf16_keys() -> None:
    """The indexer's key and gate reach ``dsa_kpool_hadamard`` as bf16, or
    production runs torch.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import _SUPPORTED_DTYPES

    assert _SUPPORTED_DTYPES == (torch.bfloat16,), (
        f"the seam's admitted dtypes are {_SUPPORTED_DTYPES}, not bf16 alone, so this check's premise "
        f"has changed and the finding needs re-reading rather than this assertion relaxing"
    )

    gen = torch.Generator().manual_seed(9_051_071)
    indexer = _bare_indexer()
    _materialise_indexer(indexer, gen)
    tokens = PREFILL_TOKENS
    hidden = torch.randn(tokens, int(indexer.hidden_size), generator=gen, dtype=torch.float32)
    q_latent = torch.randn(tokens, int(indexer.q_lora_rank), generator=gen, dtype=torch.float32)

    # The reset/read pair around this call. It is the third class of indexer call in this file and
    # the only one that reaches ``project_stage`` directly, so the pair sits here literally rather
    # than through the probe the two measured runs install on ``forward``/``forward_ragged``. Four
    # dispatches for four sites, and this check's own claim depends on it: the dtypes below are what
    # the kernels returned only if the kernels ran, and a torch oracle serving all four would return
    # the same shapes with no other symptom.
    reset_projection_counter()
    query, key, weights, gate_score = indexer.project_stage(hidden, q_latent)
    projection = read_projection_counter()
    assert projection == (PROJECTION_PER_INDEXER_CALL, 0), (
        f"project_stage read {projection} where it owes "
        f"({PROJECTION_PER_INDEXER_CALL}, 0) -- one dispatch per site at model_fp8.py, "
        f":3113. Without this the dtypes asserted below could be a torch oracle's, and this "
        f"check is about the route the seam takes"
    )
    for name, tensor, want in (
        ("query", query, torch.bfloat16),
        ("key", key, torch.bfloat16),
        ("gate_score", gate_score, torch.bfloat16),
        ("weights", weights, torch.float32),
    ):
        assert tensor.dtype == want, (
            f"project_stage returned {name} as {tensor.dtype}, not {want}. A {name} of the wrong "
            f"dtype passes every shape check and sends the pooling or scoring seam to its torch "
            f"oracle, which shows up only as a counter reading"
        )


def test_the_ragged_pack_admits_bf16_only_and_preserves_it() -> None:
    """The pack's output dtype is its input's, and bf16 is the only dtype it
    admits.
    """
    from vllm_neuron.functional.dsa import ragged_pack as rp

    assert rp._SUPPORTED_DTYPES == (torch.bfloat16,), (
        f"the pack admits {rp._SUPPORTED_DTYPES}; this check is written against bf16 alone"
    )
    lengths = [PREFILL_TOKENS // 2, PREFILL_TOKENS]
    max_len, width = max(lengths), INDEX_HEAD_DIM
    gen = torch.Generator().manual_seed(9_051_067)
    padded = torch.randn(
        len(lengths), max_len, width, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)

    packed = rp.dsa_ragged_pack(padded, lengths)
    assert packed.dtype == padded.dtype, (
        f"the pack returned {packed.dtype} from a {padded.dtype} input; the seam contracts to return "
        f"the input's dtype and a silent widening would change what the indexer scores"
    )
    assert tuple(packed.shape) == (sum(lengths), width), (
        f"the packed shape {tuple(packed.shape)} is not the closed form {(sum(lengths), width)} -- "
        f"the caller never states the packed length, so this is a measurement and not an echo"
    )


# =========================================================================== #
# the torch reference for check 1.
#
# What is independent here and what is not, because the comparison is only worth running if the
# answer is not smuggled in from the code under test.
#
#   * The orchestration is written again below from the implementation's declared contracts --
#     four projections, the pooling window, the tail ring, the cache writes, the gather, the
#     score, the selection, the expansion, the attention. A reference that shared the
#     orchestration would cancel a composition error out and pass vacuously.
#   * The seam mathematics is written again too, and not for purity: the ``torch_fallback``
#     counter bump sits inside the private reference for three of the five seams
#     (kpool_hadamard.py, topk_select.py, decode_tail_update.py) and in the public wrapper for
#     the other two (score_gemm.py, index_expand.py). Calling the shipped references would make
#     ``torch_fallback`` nonzero for reasons unrelated to the production path, and the standing
#     ``torch_fallback == 0`` reading -- the instrument that proves the kernels ran -- would read
#     a number this test itself put there.
#   * Two things are imported, and both are data rather than algorithm: ``hadamard_matrix`` and
#     ``HADAMARD_SCALE``. The matrix fixes the transform's basis and sign convention, and
#     hand-building a second Sylvester matrix would risk a convention mismatch that reads as a
#     numeric failure of a correct kernel; the scale's own docstring records that the obvious
#     spelling ``1.0 / math.sqrt(128)`` is one ulp low (kpool_hadamard.py). Importing both and
#     multiplying by hand still compares the kernel's in-place butterfly against a plain matrix
#     multiply, which is the comparison that has content.
#
# Every function below names the contract it mirrors.


def _ref_scale(indexer) -> float:
    """``projection_scale``: exactly two folded factors (``model_fp8.py``). """
    return float(int(indexer.index_head_dim) ** -0.5) * float(int(indexer.index_n_heads) ** -0.5)


def _ref_projection(x: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    """``mla_projection``: a plain contraction, float32 in and out. """
    return x.to(torch.float32) @ weight_out_in.to(torch.float32).t()


def _ref_absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``mla_absorb``: ``[S,H,K] x [H,K,N] -> [S,H,N]`` (``mla_absorb.py``)."""
    return torch.einsum("shk,hkn->shn", x.to(torch.float32), w.to(torch.float32))


def _ref_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """``_key_norm``: normalise in float32, cast back to the input's dtype
    (``model_fp8.py``).
    """
    width = int(x.shape[1])
    normed = torch.nn.functional.layer_norm(
        x.float(), (width,), weight.float(), bias.float(), float(eps)
    )
    return normed.to(x.dtype)


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Glm5NextDSALayer._input_norm`` (``model_fp8.py``)."""
    f = x.to(torch.float32)
    normed = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    return (normed * weight.to(torch.float32)).to(x.dtype)


def _ref_hadamard_matrix(width: int) -> torch.Tensor:
    """The transform's basis, taken from the seam so the sign convention cannot drift.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import hadamard_matrix

    return hadamard_matrix(int(width), dtype=torch.float32)


def _ref_hadamard_scale() -> float:
    """``HADAMARD_SCALE`` (``kpool_hadamard.py``), imported and never re-derived.
    """
    from vllm_neuron.functional.dsa.kpool_hadamard import HADAMARD_SCALE

    return float(HADAMARD_SCALE)


def _ref_rotate(x: torch.Tensor) -> torch.Tensor:
    """``dsa_hadamard128``: rotate rows in float32, scale, cast back (``kpool_hadamard.py``)."""
    rotated = x.float() @ _ref_hadamard_matrix(int(x.shape[1])).t()
    return (rotated * _ref_hadamard_scale()).to(x.dtype)


def _ref_compress_prefill(
    slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """``dsa_kpool_hadamard``: pool then rotate, one cast at the end. """
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
    """``_compress_pool_torch``: the same pooling, plus one extra intermediate cast. """
    dtype = pool_key.dtype
    weights = torch.softmax(pool_score.float() + ape.float(), dim=0)
    pooled = (weights * pool_key.float()).sum(dim=0, keepdim=True)
    pooled = pooled.to(dtype).float()  # the extra round trip: decode only
    rotated = pooled @ _ref_hadamard_matrix(int(pool_key.shape[1])).t()
    return (rotated * _ref_hadamard_scale()).to(dtype)


def _ref_score(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``dsa_score_gemm``: per-head scores, rectified, then head-weighted and summed.
    """
    return (
        _ref_score_per_head(q, k).clamp(min=0.0) * weights.float().unsqueeze(-1)
    ).sum(dim=1)


def _ref_score_per_head(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The per-head scores before the ReLU and before the head weights. """
    return torch.einsum("mhd,nd->mhn", q.float(), k.float())


def _ref_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``dsa_topk_select`` then ``select_pools``' cast (``topk_select.py``, ``model_fp8.py``)."""
    return torch.topk(scores, int(k), dim=-1).indices.to(torch.int32)


def _fill_constants() -> tuple[float, float]:
    """``(BOUND_FILL, BOUND_FILL_MARK)``, asked of the product module rather than typed
    here.
    """
    module = _seam_module("causal_bound")
    return float(module.BOUND_FILL), float(module.BOUND_FILL_MARK)


def _ref_causal_bound(
    scores: torch.Tensor, seq_lens: torch.Tensor, pool_size: int, width: int
) -> torch.Tensor:
    """``dsa_causal_bound``: ``BOUND_FILL`` at every pool column the row does not complete.
    """
    fill, _mark = _fill_constants()
    causal = seq_lens.to(torch.int64).reshape(-1, 1)
    columns = torch.arange(int(width), dtype=torch.int64).reshape(1, -1)
    completes = (columns + 1) * int(pool_size) - 1 < causal
    return scores.masked_fill(~completes, fill)


def _ref_causal_sentinel(
    bounded: torch.Tensor, pool_ids: torch.Tensor, width: int
) -> torch.Tensor:
    """``dsa_causal_sentinel``: ``-1`` at every selection the row may not see. """
    _fill, mark = _fill_constants()
    selected = bounded.gather(1, pool_ids.to(torch.int64))
    struck = torch.le(selected, mark) | torch.isnan(selected)
    padded = torch.ge(pool_ids.to(torch.int64), int(width))
    return torch.where(struck | padded, torch.full_like(pool_ids, -1), pool_ids)


def _ref_canonical_sentinel_order(pool_ids: torch.Tensor) -> torch.Tensor:
    """Sentinels to the trailing columns; real ids keep their relative order. """
    k = int(pool_ids.shape[1])
    position = torch.arange(k, dtype=torch.int64)
    key = (pool_ids < 0).to(torch.int64) * k + position
    return pool_ids.gather(1, key.argsort(dim=1))


def _ref_expand(pool_ids: torch.Tensor, seq_lens: torch.Tensor, pool_size: int) -> torch.Tensor:
    """``dsa_index_expand``: pool ids become token indices, with a raw-token tail appended.
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
    """Sparse attention over each row's non-sentinel columns only. """
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
            f"{int(cache.shape[0])}; the seam refuses this at mla_sparse.py and the "
            f"reference must not paper over it"
        )
        gathered = cache[live]
        weights = torch.softmax((q[s] @ gathered.t()) * float(softmax_scale), dim=-1)
        out[s] = weights @ gathered
    return out


# --------------------------------------------------------------------------- #
# the orchestration, re-written. This is the part test (1) tests.


def _raw(module, name: str) -> torch.Tensor:
    """The raw ``[out, in]`` checkpoint leaf, read from the module by its declared
    attribute name.
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
    """``Glm5NextDSAIndexer.project_stage`` (``model_fp8.py``), step for step.
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
    """``_gather_candidates`` collapses to a prefix slice, and the collapse is proven not
    assumed.
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
    adopt_pool_ids: torch.Tensor | None = None,
    tie_label: str | None = None,
) -> torch.Tensor:
    """``Glm5NextDSAIndexer.forward``, both legs (``model_fp8.py``). """
    pool = int(indexer.index_kpool)
    is_decode = tail is not None
    query, key, weights, gate_score = _ref_project_stage(indexer, hidden, q_latent)
    ape = indexer.index_kpool_compress_ape.to(torch.float32)

    if is_decode:
        # ``tail_step`` -> ``dsa_decode_tail_update`` (``model_fp8.py``,
        # ``decode_tail_update.py``). ``slot_of`` is ``position % pool_size``.
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
        # ``pool_window`` (``model_fp8.py``): a sliding window of ``pool`` positions
        # ending at each token, clamped at the start; a row is written only where the slot is real
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
    # The probe keeps the unbounded scores on purpose, and feeds a non-gating reading only. A
    # tie at the k-th place among real candidates is a legal product state -- the product's own
    # ReLU floors every all-non-positive candidate to an identical `0.0` -- so no criterion may
    # be stricter than tie-equivalence, and a reseed moves such a tie without removing it. The
    # bound writes a finite `BOUND_FILL`, not `-inf`.
    if probe is not None:
        probe.append(scores.detach())
    # Recomputed rather than captured inside `_ref_score`, so the value path above
    # is not touched by a diagnostic. Same inputs, same function, so it cannot disagree with the
    # scores the tie control reads.
    if probe_per_head is not None:
        probe_per_head.append(_ref_score_per_head(query, candidate_keys).detach())
    bounded = _ref_causal_bound(scores, seq_lens, pool, candidates)
    pool_ids = _ref_topk(bounded, int(indexer.select_k()))
    # `candidates` is the width, read from the same variable the bound was given, exactly as the
    # implementation reads `int(bounded.shape[1])` rather than a config field: the width the marker
    # screens against cannot then disagree with the width the selector was handed.
    pool_ids = _ref_causal_sentinel(bounded, pool_ids, candidates)
    pool_ids = _ref_canonical_sentinel_order(pool_ids)
    # Adoption, and it is conditional. The reference has computed its own selection
    # above, which is what the comparison below is against; adoption never replaces that step. When
    # the kernel's set differs, it is adopted only if it is itself a legal top-k of the same bounded
    # scores -- checked, not assumed -- so a genuinely wrong selection cannot be laundered into
    # agreement by handing it to the reference.
    if adopt_pool_ids is not None:
        assert tuple(adopt_pool_ids.shape) == tuple(pool_ids.shape), (
            f"{tie_label or 'adopt'}: the kernel returned selected pool ids of shape "
            f"{tuple(adopt_pool_ids.shape)} where the reference computed "
            f"{tuple(pool_ids.shape)}. Adoption compares the two sets row by row, so a different "
            f"shape is a finding about the seam's contract and not a shape to broadcast past"
        )
        check_tie_equivalent_selection(
            bounded, pool_ids, adopt_pool_ids, tie_label or "adopt"
        )
        pool_ids = adopt_pool_ids.to(pool_ids.dtype)
    return _ref_expand(pool_ids, seq_lens, pool)


def _ref_attend(
    attention,
    hidden: torch.Tensor,
    latent_cache: torch.Tensor,
    start_position: int,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """``Glm5NextMLAAttention.attend`` (``model_fp8.py``). """
    tokens = int(hidden.shape[0])
    heads = int(attention.num_attention_heads)
    eps = float(attention.rms_norm_eps)
    x = hidden.to(torch.float32)

    # project_query_and_latent , which routes through project_query_latent .
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
    """``_latent_norm`` (``model_fp8.py``): an RMS norm with a gain and no cast back."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + float(eps)) * gain.to(torch.float32)


def _ref_absorb_operands(attention) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the raw ``kv_b_proj`` into ``W_UK`` and ``W_UV`` (``model_fp8.py``).
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
    adopt_pool_ids: torch.Tensor | None = None,
    tie_label: str | None = None,
) -> torch.Tensor:
    """``Glm5NextDSALayer.forward`` (``model_fp8.py``). """
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
        adopt_pool_ids=adopt_pool_ids,
        tie_label=tie_label,
    )
    attn_out = _ref_attend(
        attention, normed, latent_cache, int(start_position), topk_indices, float(softmax_scale)
    )
    return residual + attn_out


# =========================================================================== #
# the fixture. A materialised stack at :data:`TINY_GEOMETRY`.


#: The softmax scale, derived the way the reference derives it: the inverse square root of the query
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
    # (``model_fp8.py``), so the leaf is stored in the checkpoint's dtype here rather than
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
    return stack, cfg, gen


def prefill_slot_mapping(tokens: int, pool: int) -> torch.Tensor:
    """The pool-granular slot per position: the pool's own id where a pool completes, else
    ``-1``.
    """
    slots = torch.full((int(tokens),), -1, dtype=torch.int32)
    for p in range(int(tokens)):
        if (p + 1) % int(pool) == 0:
            slots[p] = p // int(pool)
    return slots


def case_operands(cfg, *, tokens: int = PREFILL_TOKENS):
    """Every operand both runs need, plus the two derived counts the indexer will
    re-derive.
    """
    pool = int(cfg.index_kpool)
    rows = PAGES * PAGE_SIZE
    candidates = int(tokens) // pool
    trash = rows - 1
    assert candidates > TOPK_POOLS, (
        f"{candidates} candidate pool(s) at {tokens} token(s) is not more than the "
        f"{TOPK_POOLS} selected, and the indexer refuses that (model_fp8.py)"
    )
    assert candidates <= trash, (
        f"{candidates} candidates leaves no trash row above them in {rows} pool_cache row(s)"
    )
    return {
        "slot_mapping": prefill_slot_mapping(int(tokens), pool),
        # One length per row of the scores, and the rows are tokens -- so this is each token's own
        # context length. That makes the tail each token's own incomplete pool.
        "seq_lens": torch.arange(1, int(tokens) + 1, dtype=torch.int32),
        "candidates": candidates,
        "trash": trash,
        "rows": rows,
    }


def per_layer_caches(cfg, layers: int, *, tokens: int = PREFILL_TOKENS) -> list[dict]:
    """one set of caches per layer, which is not a detail. """
    rows = PAGES * PAGE_SIZE
    head_dim = int(cfg.index_head_dim)
    pool = int(cfg.index_kpool)
    return [
        {
            "pool_cache": torch.zeros(rows, head_dim, dtype=torch.bfloat16),
            # Whole blocks: the layer takes the bank and a table naming its pages, and a
            # bank no block divides is refused by name.
            "latent_cache": torch.zeros(
                -(-(int(tokens) + DECODE_STEPS) // PAGE_SIZE) * PAGE_SIZE,
                1, TINY_HEAD_SIZE, dtype=torch.float32,
            ),
            "tail": torch.zeros(2, pool, head_dim, dtype=torch.bfloat16),
        }
        for _ in range(int(layers))
    ]


def assert_close_within_tolerance(label: str, got: torch.Tensor, reference: torch.Tensor) -> None:
    """Assert the achieved error is inside tolerance, and name it if it is not."""
    assert got.shape == reference.shape, f"{label}: {tuple(got.shape)} vs {tuple(reference.shape)}"
    torch.testing.assert_close(got.float(), reference.float(), rtol=RTOL, atol=ATOL)


def kth_place_gap(scores: torch.Tensor, k: int) -> float:
    """The smallest gap at the k-th place, measured and returned rather than asserted on.
    """
    top = torch.topk(scores.float(), int(k) + 1, dim=-1).values
    return float((top[:, int(k) - 1] - top[:, int(k)]).min())


def say_gaps(gaps: list[tuple[str, float]]) -> None:
    """Check every label's k-th-place gap."""


def check_tie_equivalent_selection(
    bounded: torch.Tensor, ref_ids: torch.Tensor, kernel_ids: torch.Tensor, label: str
) -> None:
    """The kernel's selected set must equal the reference's up to substitution among exact
    ties.
    """
    scores = bounded.float()
    rows, width = int(scores.shape[0]), int(scores.shape[1])
    reported = False
    for r in range(rows):
        ref_row = [int(v) for v in ref_ids[r].tolist()]
        ker_row = [int(v) for v in kernel_ids[r].tolist()]
        ref_real = sorted(v for v in ref_row if v >= 0)
        ker_real = sorted(v for v in ker_row if v >= 0)

        # The distinctness reading, before any set is taken. Both lists are already sorted, so a
        # repeat is an equal neighbour. The repeated id is named because "a duplicate exists" sends
        # the next reader back to this row, and the id says which pool the row lost.
        ker_repeats = sorted({v for i, v in enumerate(ker_real[1:]) if v == ker_real[i]})
        ref_repeats = sorted({v for i, v in enumerate(ref_real[1:]) if v == ref_real[i]})
        assert not ker_repeats, (
            f"{label} row {r}: the kernel returned pool id(s) "
            f"{';'.join(str(v) for v in ker_repeats)} more than once, so this row attends fewer "
            f"pools than it claims. A repeat is an ILLEGAL selection and not a tie substitution -- "
            f"kernel={ker_row} reference={ref_row}"
        )
        assert not ref_repeats, (
            f"{label} row {r}: the REFERENCE returned pool id(s) "
            f"{';'.join(str(v) for v in ref_repeats)} more than once. The reference is the oracle, "
            f"so this is a fault in the test's own scorer and not a finding about the kernel -- "
            f"reference={ref_row} kernel={ker_row}"
        )

        assert len(ker_real) == len(ref_real), (
            f"{label} row {r}: the kernel returned {len(ker_real)} real pool ids where the "
            f"reference returned {len(ref_real)}. A different COUNT is not a tie substitution -- "
            f"kernel={ker_row} reference={ref_row}"
        )
        if ker_real == ref_real:
            continue

        chosen = set(ker_real)
        lo = min(float(scores[r, c]) for c in chosen)
        unchosen = [c for c in range(width) if c not in chosen]
        hi = max(float(scores[r, c]) for c in unchosen) if unchosen else float("-inf")
        # One row per leg, and it is the row that actually differs -- the interesting one.
        if not reported:
            reported = True
        assert lo >= hi, (
            f"{label} row {r}: the kernel's selection is not a legal top-k, so it is a WRONG "
            f"selection and not a tie substitution. Its lowest chosen score is {lo:.9e} while an "
            f"unchosen column scores {hi:.9e}, which is strictly higher. kernel={ker_real} "
            f"reference={ref_real}"
        )
    if not reported:
        # The row that came closest to a tie, so the leg always emits exactly one row.
        torch.topk(scores, min(width, 2), dim=-1).values


def say_tie_diagnostics(
    scores: torch.Tensor, k: int, label: str, per_head: torch.Tensor | None = None
) -> None:
    """The row that decides the k-th-place gap, read before any assertion."""
    s = scores.float()
    top = torch.topk(s, int(k) + 1, dim=-1).values
    row_gaps = top[:, int(k) - 1] - top[:, int(k)]
    row = int(torch.argmin(row_gaps))
    if per_head is not None:
        per_head.float()[row]


def say_selection_tie_readings(
    scores: torch.Tensor, k: int, label: str, per_head: torch.Tensor | None = None
) -> None:
    """Read the selection's tie structure. """
    say_tie_diagnostics(scores, int(k), label, per_head)
    say_gaps([(label, kth_place_gap(scores, int(k)))])


# =========================================================================== #
# run 1 of 2 -- the layer case. Acceptance test (1) and route-predicate part 1.


@pytest.mark.parametrize("layers", [LAYERS, CONTROL_LAYERS])


def test_run_1_a_dsa_stack_matches_the_torch_reference_and_moves_every_seam(
    monkeypatch: pytest.MonkeyPatch, layers: int
) -> None:
    """A DSA stack matches the reference at ``rtol=5e-3``/``atol=1e-5``."""
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=int(layers))
    ops = case_operands(cfg)
    pool = int(cfg.index_kpool)
    hidden = torch.randn(
        PREFILL_TOKENS, int(cfg.hidden_size),
        generator=torch.Generator().manual_seed(9_051_001), dtype=torch.float32,
    )
    # Two independent sets, one per side, each with one cache set per layer. The reference carries
    # its own state from prefill into decode rather than cloning the implementation's -- otherwise
    # the decode comparison would run both sides against a cache only one of them computed, and a
    # wrong prefill cache write would make the decode leg agree with itself.
    impl_caches = per_layer_caches(cfg, int(layers))
    ref_caches = per_layer_caches(cfg, int(layers))

    spy = SeamSpy()
    # The projection counter is reset once for the whole case, not once per phase, and that is the
    # difference between one reading and two instruments. The probe below reads each indexer call's
    # delta; this running total reads the case; after the loop the two are checked against each
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

        # The running projection total at the start of this phase, so both the reference check and
        # the phase delta below are differences rather than absolute numbers that would each have to
        # know what the previous phase left behind.
        projection_mark = read_projection_counter()

        # (2) Reset first, and prove the reset took effect before anything is measured against
        # it. This sits ahead of both sides: the reference adopts the implementation's selected
        # pool ids, so the implementation has to run first and a reset cannot come between the
        # two. Ahead of both is strictly stronger -- the
        # readings below are the implementation's alone because nothing ran before it, rather than
        # because a reference that already ran was cleared out behind them.
        reset_all_counters()
        after_reset = read_all_counters()
        assert all(v == (0, 0) for v in after_reset.values()), (
            f"the counters did not reset to zero: {after_reset}. Every reading below would then be "
            f"partly an earlier leg's dispatches, which is exactly the confusion this order avoids"
        )

        # (3) the implementation, on the originals, with the spy installed.
        spy_here = SeamSpy()
        spy_here.install(monkeypatch)
        # The projection probe is installed in the same window as the spy and is undone by the same
        # `monkeypatch.undo`, so it can only ever see the implementation's calls. The probe object
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
                slot_mapping=kwargs.get("slot_mapping"),
                tail=None if phase == "prefill" else caches["tail"],
                position=kwargs.get("position"),
                **paged_operands(
                    caches["latent_cache"], start, int(got.shape[0]), page=PAGE_SIZE
                ),
            )
        monkeypatch.undo()
        spy.calls.extend(spy_here.calls)
        spy.first_arg_dtypes.extend(spy_here.first_arg_dtypes)
        spy.selected_pool_ids.extend(spy_here.selected_pool_ids)

        # (4) the captured selection, checked before anything is allowed to use it. The ids the
        # reference adopts come from the spy's recorded return values and from nowhere else -- the
        # product is not called a second time and nothing here recomputes them -- so this leg must
        # hold exactly one recorded set per layer, in layer order. A repeat means a layer ran twice
        # and a gap means one did not run; either way the adopted set would belong to a different
        # layer than the reference row it is handed to, which no downstream assertion could name.
        captured_layers = [int(idx) for idx, _ids in spy_here.selected_pool_ids]
        assert captured_layers == list(range(int(layers))), (
            f"the {phase} leg recorded selected pool ids for layers {captured_layers}, where "
            f"{layers} layer(s) each select exactly once per leg and the expected record is "
            f"therefore {list(range(int(layers)))}. `build_layer_stack` numbers the stack "
            f"0..n-1 and the spy reads that same `layer_idx`, so this compares the recording "
            f"against the stack rather than against a typed count"
        )
        kernel_pool_ids = [ids for _idx, ids in spy_here.selected_pool_ids]

        # (5) the readings -- the implementation's alone, taken before the reference runs at all.
        readings = read_all_counters()
        assert all(v[1] == 0 for v in readings.values()), (
            f"a torch fallback ran on the {phase} leg: {readings}. Every DSA seam here is "
            f"work the kernel owns, so a fallback is a route failure and not a slow path"
        )
        # The phase's projection delta, against the closed form for this many layers and one phase.
        phase_reading = read_projection_counter()
        phase_nki = phase_reading[0] - projection_mark[0]
        phase_fallback = phase_reading[1] - projection_mark[1]
        want_phase = PROJECTION_PER_LAYER_PER_PHASE * int(layers)
        assert phase_nki == want_phase, (
            f"the {phase} leg dispatched {phase_nki} projections where {layers} layer(s) owe "
            f"{want_phase}, at {PROJECTION_PER_LAYER_PER_PHASE} per layer per phase. The terms and "
            f"the closed form is summed after the loop; a miss is a finding about the closed "
            f"form or about the layer, and not a number to edit"
        )
        assert phase_fallback == 0, (
            f"the {phase} leg recorded {phase_fallback} projection torch fallbacks, which no code "
            f"path can produce today (mla_projections.py), so the substrate declaration for "
            f"expectation has to be re-derived rather than this number relaxed"
        )

        # (6) the reference, on its own per-layer caches, adopting the implementation's selection
        # where that selection is tie-equivalent. It runs last because adoption needs the recorded
        # ids, and it runs after every counter reading above so it cannot touch one of them.
        #
        # The reference is bracketed by its own projection-counter snapshot, because the projection
        # counter is the one counter not reset per phase -- the case total after the loop is the
        # second instrument the per-call probe is checked against, so a per-phase reset would leave
        # nothing to check it with. That is only sound if the reference dispatches nothing, and here
        # that is measured rather than trusted: the reference projects through `_ref_projection`, a
        # plain matmul in this file, and reaches no seam. If it ever did, the case total below would
        # be part reference and would read high for a reason no assertion could name.
        candidates = (PREFILL_TOKENS if phase == "prefill" else PREFILL_TOKENS + 1) // pool
        ref_before = read_projection_counter()
        ref_hidden = step_hidden
        probe: list[torch.Tensor] = []
        # The tie diagnostics collect the per-head scores beside the weighted ones, so a tied row can be
        # read at the place the ReLU floor is actually reached.
        probe_per_head: list[torch.Tensor] = []
        for idx, (layer, caches) in enumerate(zip(stack, ref_caches)):
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
                # The adoption, one layer's recorded set per layer, positionally matched to the stack
                # by the check at step (4) rather than by assumption.
                adopt_pool_ids=kernel_pool_ids[idx],
                tie_label=f"{phase}-layer{idx}",
            )
        reference = ref_hidden

        # The tie readings, on every layer's own selection, and they are non-gating: a
        # k-th-place tie among real candidates is a legal product state, so the numbers are
        # only reported and the legality of the adopted set is what is asserted, inside
        # `_ref_indexer`.
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
            say_selection_tie_readings(
                scores,
                int(stack[idx].attention.indexer.select_k()),
                f"{phase}-layer{idx}",
                probe_per_head[idx],
            )

        ref_after = read_projection_counter()
        assert ref_after == ref_before, (
            f"the torch reference moved the projection counter from {ref_before} to {ref_after} on "
            f"the {phase} leg. The reference and the implementation have to be computed by different "
            f"means, and a reference that dispatches the seam under test is comparing the seam "
            f"against itself"
        )

        # (7) the comparison.
        assert_close_within_tolerance(f"check-1-{phase}-L{layers}", got, reference)

    # The adoption's own count, over the whole case rather than per leg. Every layer selects once per
    # leg, so the case owes exactly layers x legs recorded sets. The per-leg check above already
    # proved each leg's record is `0..n-1` with no repeat; this one proves no leg was skipped and no
    # leg recorded a second time, which is the other way the adopted ids could have come from
    # somewhere other than the run they are compared against.
    want_pool_id_sets = int(layers) * len(PHASES)
    assert len(spy.selected_pool_ids) == want_pool_id_sets, (
        f"the case recorded {len(spy.selected_pool_ids)} selected-pool-id sets where {layers} "
        f"layer(s) across {len(PHASES)} leg(s) owe {want_pool_id_sets}, at one per layer per leg. "
        f"The reference adopted from this record, so a wrong count means it adopted a set from a "
        f"call it was not compared against"
    )

    # Part 1 of the route predicate, over this case: six of the seven families move here and the
    # pack pair is the declared zero entry `aa` ruled. The seventh is run 2's.
    per_family = spy.per_family()
    expected = declared_family_totals(int(layers))
    assert per_family == expected, (
        f"the spy's per-family totals {per_family} do not match the declared table {expected}. "
        f"The table is the declaration and the spy is the measurement, so a mismatch is a finding "
        f"about one of them and not a number to edit"
    )

    # The projection substrate reading for this case, with its terms -- the reading this file did not
    # take for twelve attempts. `mla_projection` is the eighth counter family and the seven-family
    # walk above structurally cannot reach it, because it lives under `functional.attention` while
    # `FAMILIES` is built from entry points under `functional.dsa`.
    label = f"check-1-L{layers}"
    want_total = projection_closed_form(
        label, DECLARED_PROJECTION_TERMS, layers=int(layers), phases=len(PHASES)
    )
    total_nki, total_fallback = read_projection_counter()
    assert total_nki == want_total, (
        f"the case dispatched {total_nki} projections where the closed form owes {want_total}. Every "
        f"every term is declared above, so a mismatch names itself: a short count is "
        f"a call served by mla_projection_torch_oracle or not made at all, and a long one is a term "
        f"the closed form does not know about -- Glm5NextMLAAttention.project_qkv holds four more "
        f"dispatch sites and is dead on this path, so wiring it in would read here first"
    )
    assert total_fallback == 0, f"the case recorded {total_fallback} projection torch fallbacks"

    # Two instruments over the same events, and the residual is the check. The probe read each
    # indexer call's delta; the counter read the whole case. So the case total minus the indexer's
    # share must be exactly the non-indexer terms. A probe that double-counted, or an indexer that
    # projected three times while something else projected five, agrees with neither.
    projection_probe.check(label, want_calls=int(layers) * len(PHASES))
    indexer_share = projection_probe.dispatched
    expected_indexer = PROJECTION_PER_INDEXER_CALL * int(layers) * len(PHASES)
    want_rest = PROJECTION_NON_INDEXER_PER_LAYER_PER_PHASE * int(layers) * len(PHASES)
    assert indexer_share == expected_indexer, (
        f"the probe read {indexer_share} projections across the indexer calls where "
        f"{int(layers) * len(PHASES)} calls at {PROJECTION_PER_INDEXER_CALL} each owe {expected_indexer}"
    )
    assert total_nki - indexer_share == want_rest, (
        f"the case total {total_nki} less the indexer's {indexer_share} leaves "
        f"{total_nki - indexer_share} for the layer's own projections, where the non-indexer terms "
        f"owe {want_rest}. The two instruments read the same events by different means, so this is "
        f"where one of them being wrong shows up"
    )

    # NOTE, read on the production path rather than reconstructed. The standalone check
    # reads what `project_stage` returns; this reads what the seams were actually handed while the
    # layer ran, which is the thing the finding is about. `torch_fallback == 0` above already proves
    # the kernels served these calls, so the two readings together say the bf16 reached the gate and
    # the gate admitted it.
    for entry in ("dsa_kpool_hadamard", "dsa_hadamard128", "dsa_decode_tail_update"):
        seen = spy.dtypes_for(entry)
        assert seen, (
            f"{entry} has no recorded production dtype, so the dtype check below would assert a subset "
            f"of nothing and pass while measuring nothing. The per-entry counts say "
            f"whether the seam was called; if it was, the spy's dtype log did not reach this spy"
        )
        assert seen == {torch.bfloat16}, (
            f"{entry} was handed {sorted(str(d) for d in seen)} on the production path; anything but "
            f"bf16 fails the seam's dtype gate and takes the torch oracle while every shape still "
            f"checks out (kpool_hadamard.py)"
        )


# =========================================================================== #
# run 2 of 2 -- the ragged decode arm. The seventh counter family.


def test_run_2_the_ragged_arm_packs_and_each_request_matches_itself_run_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seventh family, and a correctness reading that is exact rather than a tolerance.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    hidden_size = int(cfg.hidden_size)
    q_lora = int(cfg.q_lora_rank)

    # A non-uniform batch: the arm refuses a uniform one by name (``model_fp8.py``).
    lengths = [PREFILL_TOKENS // 2, PREFILL_TOKENS]
    max_len = max(lengths)
    tokens = sum(lengths)
    assert len(set(lengths)) > 1, "a uniform batch is refused by the arm, and rightly"

    gen = torch.Generator().manual_seed(9_051_003)
    hidden = torch.randn(len(lengths), max_len, hidden_size, generator=gen, dtype=torch.float32)
    q_latent = torch.randn(len(lengths), max_len, q_lora, generator=gen, dtype=torch.float32)

    # The arm only reads the pool cache -- it writes no pool -- so the cache is seeded directly
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
    assert int(seq_lens.shape[0]) == tokens, "one seq_len per PACKED row (model_fp8.py)"

    reset_projection_counter()
    projection_probe = IndexerProjectionProbe()

    select_k = int(indexer.select_k())
    scored: list[tuple[int, torch.Tensor]] = []
    for b, n in enumerate(lengths):
        q_own, _k, w_own, _g = _ref_project_stage(
            indexer, hidden[b, :n], q_latent[b, :n]
        )
        scored.append((n, _ref_score(q_own, _ref_candidate_keys(pool_cache, candidates), w_own)))

    say_gaps(
        [
            (f"arm-request-{b}", kth_place_gap(scores, select_k))
            for b, (_n, scores) in enumerate(scored)
        ]
    )

    # The oracle mirrors the product's four-step chain. An oracle that selected straight off
    # the unbounded scores and expanded would miss three steps `select_bounded_pools` takes --
    # it bounds, selects, marks and canonically orders -- so the arm would compare the packed
    # implementation against a reference for a different product, and any disagreement the
    # bound or the marker introduced would be read
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

    # The reference dispatched nothing, read rather than assumed -- the same claim run 1 makes on
    # each of its legs, for the same reason: the two sides have to be computed by different means.
    after_reference = read_projection_counter()
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
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the ragged arm: {readings}"
    )
    assert readings["ragged_pack"][0] > 0, (
        "the ragged arm did not move the pack family, which is the ONLY reason this second run "
        "exists -- part 1 reads 7/7 over the union of the two runs and this is the seventh"
    )

    # The arm's projection reading, with its terms. One term, because the arm calls the indexer
    # directly, and the term is the indexer's four. The probe's per-call reading and this total are
    # the same number here, and that is not a duplication: they arrive by different means, so a
    # probe that missed the call reads zero calls while the total still reads four.
    want_arm = projection_closed_form(
        "arm", DECLARED_PROJECTION_TERMS_RAGGED_ARM, layers=1, phases=1
    )
    arm_nki, arm_fallback = read_projection_counter()
    assert arm_nki == want_arm, (
        f"the arm dispatched {arm_nki} projections where it owes {want_arm}. The arm projects the "
        f"PADDED GRID ONCE rather than once per request -- that commutation is the claim this run "
        f"exists to test (model_fp8.py) -- so a reading of "
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

    expand_mod = _seam_module("index_expand")
    raw_want = int(expand_mod.index_expand_raw_width(select_k, pool))
    emitted_want = int(expand_mod.index_expand_width(select_k, pool))
    # KEY_CHUNK read from the module that defines it (mla_sparse.py), which is where index_expand
    # imports it from too (index_expand.py). Reading it here is what keeps claim 1 from resting on
    # index_expand_width alone: if that helper were wrong, the multiple-of-KEY_CHUNK arm and the
    # not-below-raw arm would still catch a width that cannot be what the sparse kernel admits.
    key_chunk = int(importlib.import_module("vllm_neuron.functional.attention.mla_sparse").KEY_CHUNK)
    raw_got = int(reference.shape[1])
    assert raw_got == raw_want, (
        f"the reference emitted {raw_got} columns where index_expand_raw_width({select_k}, {pool}) "
        f"says {raw_want}; the reference's own width is a reading before it is a yardstick"
    )

    # Claim 1, the emitted width.
    assert int(got.shape[0]) == int(reference.shape[0]), (
        f"the arm returned {int(got.shape[0])} rows against the reference's "
        f"{int(reference.shape[0])}; the row count is the packed token count and must agree exactly"
    )
    assert int(got.shape[1]) == emitted_want, (
        f"the arm emitted {int(got.shape[1])} columns where index_expand_width({select_k}, {pool}) "
        f"says {emitted_want} (index_expand.py, allocated at :527)"
    )
    assert int(got.shape[1]) % key_chunk == 0, (
        f"the emitted width {int(got.shape[1])} is not a whole multiple of KEY_CHUNK={key_chunk}, so "
        f"mla_sparse_attention would refuse it (mla_sparse.py) whatever index_expand_width "
        f"returns -- this arm holds even if that helper is wrong"
    )
    assert int(got.shape[1]) >= raw_want, (
        f"the emitted width {int(got.shape[1])} is below the raw width {raw_want}, so meaningful "
        f"columns were dropped by the allocation itself"
    )

    # Claim 2, the meaningful columns.
    head = got[:, :raw_got]
    mismatches = int((head.to(torch.int64) != reference.to(torch.int64)).sum())
    assert mismatches == 0, (
        f"{mismatches} of {int(reference.numel())} meaningful expanded indices differ over the first "
        f"{raw_got} columns. The implementation claims the pack COMMUTES with the row-wise "
        f"projections bit-for-bit (model_fp8.py); this is that claim failing, not a "
        f"tolerance to widen"
    )

    # Claim 3, the padding. Its own claim because a padding column carrying a real token index is a
    # different fault from a wrong content column: mla_sparse drops `-1` and attends anything else
    # (mla_sparse.py), so meaning written here would be silently attended.
    tail = got[:, raw_got:]
    pad_cols = int(tail.shape[1])
    bad_pad = int((tail.to(torch.int64) != -1).sum())
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
        f"every column that is not -1 (mla_sparse.py) and attends it, so a real token "
        f"index written past the raw width is extra attention with no other symptom"
    )

    per_family = spy.per_family()
    expected = declared_family_totals(1, DECLARED_PER_LAYER_RAGGED_ARM)
    assert per_family == expected, (
        f"the arm's per-family totals {per_family} do not match its declared table {expected}"
    )

    # NOTE on the NKI path. The standalone check reads the pack's dtype behaviour through
    # torch; this reads it here, where `ragged_pack` moved a counter and `torch_fallback` is 0 above --
    # So the dtype the seam was handed is the dtype the kernel admitted, which torch alone cannot say.
    seen = spy.dtypes_for("dsa_ragged_pack")
    assert seen == {torch.bfloat16}, (
        f"the pack was handed {sorted(str(d) for d in seen)}; the module admits bf16 alone "
        f"(ragged_pack.py) and any other dtype routes to the torch path with no other symptom"
    )


# =========================================================================== #
# the readers the two regime files share: the emitted width, the bypass seam's own
# counter, and the two-instrument agreement check.


def _bypass_width(indexer, pool: int) -> int:
    """The width the bypass must emit, read from the expansion seam's own helper, with
    three checks.
    """
    expand_mod = _seam_module("index_expand")
    select_k = int(indexer.select_k())
    raw = int(expand_mod.index_expand_raw_width(select_k, pool))
    emitted = int(expand_mod.index_expand_width(select_k, pool))
    key_chunk = int(
        importlib.import_module("vllm_neuron.functional.attention.mla_sparse").KEY_CHUNK
    )
    assert emitted % key_chunk == 0, (
        f"the emitted width {emitted} is not a whole multiple of KEY_CHUNK {key_chunk}, which is "
        f"the allocation rule the sparse kernel admits"
    )
    assert emitted >= raw, f"the emitted width {emitted} is below the raw expansion width {raw}"
    # The arithmetic the bypass rests on, read as a value rather than argued. The bypass holds only
    # while `max_seq_len // pool <= select_k`, so the largest position it can present is
    # `select_k * pool + pool - 2`, and that must sit strictly inside the emitted width -- otherwise
    # some admissible length would need a clamp that nothing implements.
    max_position = select_k * int(pool) + int(pool) - 2
    assert max_position < emitted, (
        f"the widest bypass position {max_position} does not fit inside {emitted} column(s), so "
        f"some admissible length would write past the last column"
    )
    return emitted


def _causal_fill_api():
    """``(reset, read)`` for the bypass's own counter, discovered by the same rule as the
    seven.
    """
    return _discover_counter_api(_seam_module("causal_fill"))


def _check_two_instruments_agree(label: str, spy: SeamSpy, readings: dict) -> None:
    """The spy and the seven counters over the same events. One zero is not a reading; two are."""
    for family, (spy_sum, counter_sum) in agreement(spy, readings).items():
        assert spy_sum == counter_sum, (
            f"the spy counted {spy_sum} {family} call(s) where the counter read {counter_sum}; two "
            f"instruments over the same events must agree, or neither zero is a reading"
        )


# =========================================================================== #
# the softmax scale, against the reference's own derivation.
#
# Every other test here applies the same scale to the torch reference as ``attend``
# receives, so each one measures agreement at a chosen scale and is blind to which
# scale was chosen -- which is how ``kv_lora_rank ** -0.5`` could pass. This test
# reads the constant against the reference's formula instead.


def test_softmax_scale_is_the_reference_derivation_not_the_latent_rank() -> None:
    """The constant is the query head width's inverse square root."""
    width = TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]
    reference = float(width**-0.5)
    assert SOFTMAX_SCALE == reference, (
        f"SOFTMAX_SCALE is {SOFTMAX_SCALE!r}; the reference's derivation over this file's "
        f"own geometry is {reference!r}. The scale is the inverse square root of the QUERY "
        f"head width qk_nope_head_dim + qk_rope_head_dim = {width}, which is the fork's own "
        f"qk_head_dim (model_fp8.py), summed because the query carries both halves"
    )
