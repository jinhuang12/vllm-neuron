"""Acceptance for `inc-glm53f-024` -- the WP6 retile mapping PRODUCER.

WHAT IS UNDER TEST
------------------
``vllm_neuron.functional.moe.blockwise_fp8_retile`` -- the host-side producer that
maps a blockwise-fp8 checkpoint's ``[128, 128]`` fp32 block scales onto the
consumer kernel's ``256 x 256`` granularity. Five counted parts, plus a negative
case and a detector arm.

WHY THE REFERENT IS IMPORTED RATHER THAN RE-TRANSCRIBED
------------------------------------------------------
Part 2's declared expectation is that the producer *"reproduces ``-071``'s
reference mapping exactly"*. ``-071``'s measurement is landed in-tree as
``test_blockwise_fp8_retile_losslessness.py``, so that module is imported here and
used **read-only** as the referent: its ``build_checkpoint`` supplies the fixtures
its own reference mapping was measured against, and its ``retile`` /
``dequant_path_o`` / ``dequant_path_k`` are the reference. Re-transcribing the
reference into this file instead would compare the producer against a copy this
file wrote -- the tautology ``-071`` section 8 warns about. Nothing in the referent
is modified.

THE FALSE-PASS DOORS THIS FILE CLOSES
-------------------------------------
Measured at the unmodified parent ``382091c`` by
``increments/probe-024-parent-readings.py`` before any of this existed:

* ``R3`` -- the referent compared **against itself** is bit-equal and scores
  ``16/16`` at the parent. So part 2 puts the PRODUCER on one side; the producer
  did not exist at the parent (``R1``: ``find_spec`` -> ``False``).
* ``R2`` -- ``is_pow2_exact`` already existed at the parent, in the referent
  (1 file). So every power-of-two reading here calls the PRODUCER's predicate, and
  the two are additionally cross-checked against each other.
* ``R7`` -- the referent already raises ``AssertionError`` on a non-256-divisible
  shape. So the negative case asserts the producer's own NAMED error type, which
  did not exist at the parent (``R2``: 0 occurrences).
* ``R4`` -- part 1's expected tuple is written here as an independent literal
  formula transcribed from ``-070``, and its values ``(2, 512)`` / ``(2, 2048)``
  were recorded by the parent probe before the producer was authored.
* ``R6`` -- the referent's default fixture already has conjunct 1 holding while
  conjunct 2 fails on ``4/4`` blocks. That makes it the DETECTOR fixture, and it
  is why part 5's ``k/N == 100%`` needs a fixture built to satisfy BOTH conjuncts.

DECLARED CONSTRUCTION -- stated before any reading, so no arm is tuned to pass
-----------------------------------------------------------------------------
* Part 2 uses the referent's own ``build_checkpoint`` verbatim: fp8 weights on the
  fp8 grid, and four scales per ``256`` block that are mutually power-of-two
  related over the fixed shift table ``((0, 1), (2, -1))`` with a per-block base
  that is **deliberately not** a power of two.
* Part 5 uses :func:`build_pow2_checkpoint`, which differs in exactly one respect:
  the per-block base **is** a power of two. Both conjuncts then hold, which is
  what makes ``k/N == 100%`` the declared reading on synthetic fixtures.
* Every generator is seeded. Every bar is exact equality on IEEE-754 fp32 or an
  integer count, so no reading depends on a library's reduction order and no
  tolerance pair is owed (D3 applies to inexact comparisons; there are none here).

NON-SCOPE, DECLARED
-------------------
The satisfying fraction over REAL checkpoint weights is not measured here and no
synthetic stand-in is substituted for it. Real scales are not generally powers of
two, so that fraction is expected to be materially below 100% -- which is not a
defect of this producer. It belongs to the first attempt that loads a real
checkpoint, at the bar it already carries.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import pytest
import torch

from vllm_neuron.functional.moe import blockwise_fp8_retile as producer

# The referent: -071's landed measurement. READ-ONLY -- never edited, never
# monkeypatched.
from test.vllm_neuron.functional.moe import (  # noqa: E402
    test_blockwise_fp8_retile_losslessness as referent,
)

FP32 = torch.float32
FP8 = torch.float8_e4m3fn

PRODUCER_PATH = pathlib.Path(producer.__file__)
REFERENT_PATH = pathlib.Path(referent.__file__)

# Part 1's expected shapes, transcribed INDEPENDENTLY from `-070`
# (`increments/wp6-scale-consumer-geometry.md`, shapes table):
#   down_proj_scale    (dims.E, I_blocks_total * (dims.H // BLOCK_QUANT_SIZE)
#                       * TILE_SIZE)                            bwmm_shard_on_I.py:1995
#   gate_up_proj_scale (dims.E, H_blocks * 2 * I_blocks_total
#                       * TILE_SIZE)                            bwmm_shard_on_I.py:1127
# Written as arithmetic on the extents, not by calling the producer, so the two
# sides of the comparison are independent expressions.
_TILE_SIZE = 128
_BLOCK = 256


def expected_down_shape(experts: int, rows: int, cols: int) -> tuple[int, int]:
    return (experts, (cols // _BLOCK) * (rows // _BLOCK) * _TILE_SIZE)


def expected_gate_up_shape(experts: int, rows: int, cols: int) -> tuple[int, int]:
    return (experts, (rows // _BLOCK) * 2 * (cols // _BLOCK) * _TILE_SIZE)


# --------------------------------------------------------------------------- #
# Fixtures.                                                                    #
# --------------------------------------------------------------------------- #
def referent_fixture(
    rows: int, cols: int, perturb: tuple | None = None
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """The referent's own fixture, lifted to the producer's ``(E, H, I)`` layout.

    The referent is 2-D and single-expert; the producer takes an expert bank. The
    lift is ``unsqueeze(0)`` and nothing else, so part 2 compares the same numbers
    the referent measured.
    """
    weights, scales, dims = referent.build_checkpoint(rows, cols, perturb=perturb)
    return weights.unsqueeze(0), scales.unsqueeze(0), dims


def build_pow2_checkpoint(
    rows: int, cols: int, experts: int = 1, first_exponent: int = -6
) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixture satisfying BOTH losslessness conjuncts.

    Identical in construction to the referent's ``build_checkpoint`` except that
    the per-block base is ``2 ** k`` rather than a non-power-of-two, so the
    retained ``256``-block scale is itself a power of two (conjunct 2) while the
    four constituent scales stay mutually power-of-two related over the referent's
    own shift table (conjunct 1).
    """
    h_tiles, i_tiles = rows // _TILE_SIZE, cols // _TILE_SIZE
    h_256, i_256 = rows // _BLOCK, cols // _BLOCK
    scales = torch.empty((experts, h_tiles, i_tiles), dtype=FP32)
    for expert in range(experts):
        for h_block in range(h_256):
            for i_block in range(i_256):
                base = referent.fp32(
                    2.0 ** (first_exponent + expert + h_block * i_256 + i_block)
                )
                for h_off in range(2):
                    for i_off in range(2):
                        scales[expert, h_block * 2 + h_off, i_block * 2 + i_off] = (
                            referent.fp32(
                                base * 2.0 ** referent.SHIFT_TABLE[h_off][i_off]
                            )
                        )
    generator = torch.Generator().manual_seed(20260901)
    raw = torch.randint(
        -120, 121, (experts, rows, cols), generator=generator
    ).to(FP32) / 16.0
    return raw.to(FP8).to(FP32), scales


def transposed_index(h_256: int, i_256: int):
    """A DELIBERATELY WRONG flattening: h-major instead of i-major.

    Never used to measure anything. It exists only so the emitted-slot counted
    zero can be shown to move (D1.5). On a non-square block grid every wrong index
    still lands inside the tensor, which is exactly the failure a range check
    cannot see.
    """

    def index(h_tile: int, i_tile: int) -> int:
        return (h_tile // 2) * i_256 + (i_tile // 2)

    return index


# --------------------------------------------------------------------------- #
# Provenance and instrument guards.                                            #
# --------------------------------------------------------------------------- #
def test_referent_identity_is_the_landed_071_module_and_is_not_modified() -> None:
    """Records WHICH bytes part 2 is measured against.

    The digest is REPORTED, not gated: gating it would make this file fail on a
    legitimate future edit to the referent, which is not this increment's to
    adjudicate. What is asserted is only that the referent resolved to the
    in-tree overlay path.
    """
    raw = REFERENT_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    print(
        f"[provenance] referent = {REFERENT_PATH.name}  sha256 = {digest}  "
        f"lines = {raw.count(chr(10).encode())}  "
        f"producer = {PRODUCER_PATH.name}"
    )
    assert REFERENT_PATH.name == "test_blockwise_fp8_retile_losslessness.py"
    assert REFERENT_PATH.parent == pathlib.Path(__file__).parent


def test_is_pow2_exact_is_a_bit_pattern_test_and_never_a_log() -> None:
    """The PRODUCER's predicate (door R2), pinned three ways.

    Certifying component: ``blockwise_fp8_retile.is_pow2_exact`` /
    ``blockwise_fp8_retile.pow2_exponent``.
    """
    table = {
        1.0: True,
        2.0: True,
        0.5: True,
        2.0**-30: True,
        2.0**20: True,
        1.5: False,
        3.0: False,
        0.75: False,
        1.0 + 2.0**-23: False,  # one fp32 ulp above a power of two
        0.0: False,
        -2.0: False,
        float("inf"): False,  # significand IS all zeros -- must still be False
        float("nan"): False,
    }
    wrong = {
        value: producer.is_pow2_exact(value)
        for value, want in table.items()
        if producer.is_pow2_exact(value) is not want
    }
    assert not wrong, f"producer.is_pow2_exact disagrees with the declared table at {wrong}"

    # Cross-check against the referent's independently written predicate.
    disagree = {
        value: (producer.is_pow2_exact(value), referent.is_pow2_exact(value))
        for value in table
        if producer.is_pow2_exact(value) != referent.is_pow2_exact(value)
    }
    assert not disagree, f"producer and referent predicates disagree at {disagree}"

    # The exponent inverse agrees with the predicate.
    for value, want in table.items():
        assert (producer.pow2_exponent(value) is not None) is want
    assert producer.pow2_exponent(0.25) == -2
    assert producer.pow2_exponent(8.0) == 3

    # No log-family call anywhere in the PRODUCER, screened over the parse tree
    # rather than the text: a textual screen would trip on the word "log2" in the
    # module's own prose and would miss an aliased import.
    tree = ast.parse(PRODUCER_PATH.read_text(encoding="utf-8"))
    banned = {"log2", "log", "log10", "frexp"}

    def log_calls(parsed: ast.AST) -> list[tuple[str, int]]:
        found = []
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else getattr(target, "id", "")
            )
            if name in banned:
                found.append((name, node.lineno))
        return found

    hits = log_calls(tree)
    control = log_calls(ast.parse("import math\nx = math.log2(4.0)\n"))
    print(
        f"[guard] log-family calls in the producer = {len(hits)} {hits}  "
        f"screen liveness on a synthetic control = {len(control)}"
    )
    assert not hits, f"a log-based power-of-two test crept into the producer at {hits}"
    assert len(control) == 1, "the ast screen is dead -- it cannot see math.log2"


# --------------------------------------------------------------------------- #
# PART 1 -- consumer-shape equality, 1/1.                                      #
# --------------------------------------------------------------------------- #
def test_part1_consumer_scale_shape_equals_the_enumerated_shape_1_of_1() -> None:
    """Part 1: the emitted scale tensor's shape equals, element by element as a
    tuple, the shape ``-070`` enumerated from the consumer's own allocation.

    Certifying component: ``blockwise_fp8_retile.consumer_scale_shape`` and the
    ``consumer_scales`` field of ``retile_block_scales``'s result.
    """
    experts, rows, cols = 2, 512, 512
    want = expected_down_shape(experts, rows, cols)
    declared = producer.consumer_scale_shape(experts, rows, cols, producer.DOWN)

    weights, scales = build_pow2_checkpoint(rows, cols, experts=experts)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    emitted = tuple(int(extent) for extent in result.consumer_scales.shape)

    matched = int(
        len(emitted) == len(want)
        and all(left == right for left, right in zip(emitted, want))
    )
    print(
        f"[P1] emitted shape = {emitted}  declared = {declared}  "
        f"-070 enumerated = {want}  element-by-element equal = {matched}/1  "
        f"dtype = {result.consumer_scales.dtype}"
    )
    assert len(emitted) == len(want), f"rank {len(emitted)} vs {len(want)}"
    for axis, (left, right) in enumerate(zip(emitted, want)):
        assert left == right, f"axis {axis}: emitted {left} vs enumerated {right}"
    assert matched == 1, f"shape equality held in {matched}/1 cases"
    assert declared == want
    assert result.consumer_scales.dtype == FP32


def test_part1_gate_up_shape_also_matches_the_enumeration() -> None:
    """ADDITIONAL, not a declared conjunct: the second enumerated form.

    ``-070`` enumerated TWO allocations (:1127 gate/up, :1995 down). The declared
    part-1 count is ``1/1`` and is taken on the down form, which is the form part
    2's referent uses; this arm keeps the other enumerated shape from being
    silently dropped without widening the declared count.
    """
    want = expected_gate_up_shape(2, 512, 512)
    got = producer.consumer_scale_shape(2, 512, 512, producer.GATE_UP)
    weights, scales = build_pow2_checkpoint(512, 512, experts=2)
    result = producer.retile_block_scales(weights, scales, producer.GATE_UP)
    print(
        f"[P1-extra] gate_up declared = {got}  -070 enumerated = {want}  "
        f"emitted = {tuple(result.consumer_scales.shape)}  "
        f"emitted_unsupplied = {result.emitted_unsupplied}"
    )
    assert got == want
    assert tuple(int(e) for e in result.consumer_scales.shape) == want
    assert result.emitted_unsupplied == 0


# --------------------------------------------------------------------------- #
# PART 2 -- reproduces -071's reference mapping exactly.                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("side", "expected_blocks"), [(512, 16), (1024, 64)])
def test_part2_reproduces_071_reference_mapping_bit_exactly(
    side: int, expected_blocks: int
) -> None:
    """Part 2: bit-exact against ``-071``'s reference mapping, ``16/16`` blocks at
    ``[512,512]`` and ``64/64`` at ``[1024,1024]``.

    Certifying component: ``blockwise_fp8_retile.retile_block_scales`` --
    ``block_scales`` (the mapping) and ``retiled_weights`` -- measured against
    ``test_blockwise_fp8_retile_losslessness.retile`` and ``dequant_path_o``.
    """
    weights, scales, dims = referent_fixture(side, side)
    reference_flat, _, reference_weights, reference_inexact = referent.retile(
        weights[0], scales[0], dims
    )
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    mine_flat = result.block_scales[0].reshape(-1)
    mine_weights = result.retiled_weights[0].to(FP32)

    # The reference dequantisation (path O) is written straight from the
    # checkpoint's definition and never consults the consumer, which is what makes
    # the comparison able to fail.
    path_o = referent.dequant_path_o(weights[0], scales[0], dims)
    path_k = referent.dequant_path_k(mine_weights, mine_flat, dims)
    max_abs_diff = (path_o - path_k).abs().max().item()
    k, n = referent.bit_exact_block_count(path_o, path_k, _TILE_SIZE)

    print(
        f"[P2] [{side},{side}] mapping equal = "
        f"{bool(torch.equal(mine_flat, reference_flat))}  "
        f"retiled weights equal = {bool(torch.equal(mine_weights, reference_weights))}  "
        f"max abs diff = {max_abs_diff!r}  bit-exact blocks = {k}/{n}  "
        f"inexact rescales = {result.inexact_rescales} (reference "
        f"{reference_inexact})"
    )
    assert n == expected_blocks, f"[{side},{side}] gave {n} blocks, expected {expected_blocks}"
    assert torch.equal(mine_flat, reference_flat), (
        "the producer's retained-scale mapping differs from -071's reference"
    )
    assert torch.equal(mine_weights, reference_weights), (
        "the producer's retiled weights differ from -071's reference"
    )
    assert max_abs_diff == 0.0, f"max abs diff(O, K) = {max_abs_diff!r}, required exactly 0.0"
    assert k == n, f"bit-exact over {k}/{n} blocks, required {n}/{n}"
    assert result.inexact_rescales == reference_inexact == 0


def test_part2_the_comparison_can_fail_mutation_probe() -> None:
    """The declared bars discriminate the transcription from wrong arithmetic.

    Part 2 compares two independently written implementations, but a reviewer
    cannot tell from a green run whether the bar could have failed. Driving the
    producer with a deliberately wrong flattening answers that in-band.
    """
    weights, scales, dims = referent_fixture(512, 1024)
    reference_flat, _, _, _ = referent.retile(weights[0], scales[0], dims)
    mutant = producer.retile_block_scales(
        weights, scales, producer.DOWN, index_fn=transposed_index(2, 4)
    )
    path_o = referent.dequant_path_o(weights[0], scales[0], dims)
    mutant_flat = mutant.consumer_scales[0][:: _TILE_SIZE]
    path_k = referent.dequant_path_k(
        mutant.retiled_weights[0].to(FP32), mutant_flat, dims
    )
    max_abs_diff = (path_o - path_k).abs().max().item()
    k, n = referent.bit_exact_block_count(path_o, path_k, _TILE_SIZE)
    print(
        f"[P2-mutation] transposed flattening: max abs diff = {max_abs_diff!r}  "
        f"bit-exact blocks = {k}/{n}  emitted_unsupplied = "
        f"{mutant.emitted_unsupplied}  reference flat len = {reference_flat.numel()}"
    )
    assert max_abs_diff > 0.0, "a transposed flattening left path K bit-identical"
    assert k < n, "a transposed flattening scored bit-exact on every block"


# --------------------------------------------------------------------------- #
# PART 3 -- round trip through the producer's own layout.                      #
# --------------------------------------------------------------------------- #
def test_part3_round_trip_through_its_own_layout_is_bit_exact_16_of_16() -> None:
    """Part 3: re-expanding reproduces the original ``[128,128]`` fp32 scales
    bit-exactly -- ``max abs diff == 0.0`` over ``16/16`` blocks.

    EXPLICITLY NOT THE LOSSLESSNESS PROOF. Any invertible re-layout satisfies
    this; losslessness is part 5. It is here because a layout that is *not*
    invertible has silently thrown information away, and this is the arm that
    says so.

    Certifying component: ``blockwise_fp8_retile.expand_to_checkpoint_scales``
    over ``tile_exponent_shifts`` + ``block_scales``.
    """
    weights, scales, dims = referent_fixture(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    rebuilt = producer.expand_to_checkpoint_scales(result)

    max_abs_diff = (rebuilt - scales).abs().max().item()
    tiles = dims["h_tiles"] * dims["i_tiles"]
    exact = int((rebuilt == scales).sum().item())
    print(
        f"[P3] round trip max abs diff = {max_abs_diff!r}  "
        f"bit-exact [128,128] blocks = {exact}/{tiles}  "
        f"shifts = {sorted(set(result.tile_exponent_shifts.flatten().tolist()))}  "
        f"NOT the losslessness proof"
    )
    assert tiles == 16, f"[512,512] gave {tiles} [128,128] blocks, expected 16"
    assert max_abs_diff == 0.0, f"round trip max abs diff = {max_abs_diff!r}, required 0.0"
    assert exact == tiles, f"round trip bit-exact over {exact}/{tiles} blocks, required 16/16"


def test_part3_control_round_trip_breaks_when_the_layout_is_not_invertible() -> None:
    """D1.5 control for part 3: the round trip is not vacuously exact.

    A non-power-of-two ratio has no integer exponent shift, so that tile's scale
    is unrepresentable in the layout and the round trip must report it rather than
    round it.
    """
    weights, scales, dims = referent_fixture(512, 512, perturb=((1, 1), "x1.5"))
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    rebuilt = producer.expand_to_checkpoint_scales(result)
    unrepresentable = int(torch.isnan(rebuilt).sum().item())
    exact = int((rebuilt == scales).sum().item())
    print(
        f"[P3-control] x1.5 ratio: unrepresentable tiles = {unrepresentable}  "
        f"bit-exact tiles = {exact}/16  dropped = {result.input_scales_dropped}"
    )
    assert unrepresentable == 1, f"{unrepresentable} unrepresentable tiles, expected 1"
    assert exact == 15, f"{exact}/16 tiles round-tripped, expected exactly one to break"


# --------------------------------------------------------------------------- #
# PART 4 -- the two counted zeros, each with a control that MOVES.             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rows", "cols"), [(512, 512), (1024, 1024), (512, 1024)]
)
def test_part4_counted_zero_no_scale_emitted_that_the_input_did_not_supply(
    rows: int, cols: int
) -> None:
    """Part 4, first counted zero: ``0`` scales emitted that the input did not
    supply.

    Certifying component: ``blockwise_fp8_retile.count_emitted_unsupplied`` over
    the ``consumer_scales`` field. The non-square ``[512,1024]`` case is included
    because that is the grid on which a wrong flattening stays in range.
    """
    weights, scales = build_pow2_checkpoint(rows, cols)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    slots = result.consumer_scales.shape[1] // _TILE_SIZE
    print(
        f"[P4-emitted] [{rows},{cols}] emitted unsupplied = "
        f"{result.emitted_unsupplied} over {slots} slots  "
        f"NaN slots remaining = {int(torch.isnan(result.consumer_scales).sum().item())}"
    )
    assert result.emitted_unsupplied == 0, (
        f"{result.emitted_unsupplied} emitted slots carry a value the input never "
        f"supplied"
    )


@pytest.mark.parametrize(
    ("rows", "cols"), [(512, 512), (1024, 1024), (512, 1024)]
)
def test_part4_counted_zero_no_input_scale_dropped(rows: int, cols: int) -> None:
    """Part 4, second counted zero: ``0`` input scales dropped.

    Certifying component: ``blockwise_fp8_retile.count_input_scales_dropped``,
    which measures by re-expanding and comparing rather than by inferring.
    """
    weights, scales = build_pow2_checkpoint(rows, cols)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    print(
        f"[P4-dropped] [{rows},{cols}] input scales dropped = "
        f"{result.input_scales_dropped} over {scales.numel()} input scales"
    )
    assert result.input_scales_dropped == 0, (
        f"{result.input_scales_dropped} input scales cannot be recovered from the "
        f"emitted layout"
    )


def test_part4_control_emitted_counter_moves_under_a_transposed_flattening() -> None:
    """D1.5 control: the emitted counted zero can return non-zero.

    On a non-square block grid the transposed index stays inside the tensor's
    extent while reading the wrong block, so this also shows that a range check
    alone would have scored the wrong flattening clean.
    """
    weights, scales = build_pow2_checkpoint(512, 1024)
    correct = producer.retile_block_scales(weights, scales, producer.DOWN)
    h_256, i_256 = 512 // _BLOCK, 1024 // _BLOCK
    assert h_256 != i_256, "grid must be non-square for this control to bite"
    index = transposed_index(h_256, i_256)
    mutant = producer.retile_block_scales(
        weights, scales, producer.DOWN, index_fn=index
    )
    slots = correct.consumer_scales.shape[1] // _TILE_SIZE
    in_range = all(
        0 <= index(h, i) < slots
        for h in range(512 // _TILE_SIZE)
        for i in range(1024 // _TILE_SIZE)
    )
    print(
        f"[P4-control-A] grid h_256={h_256} i_256={i_256} slots={slots}  "
        f"correct = {correct.emitted_unsupplied}  "
        f"transposed = {mutant.emitted_unsupplied}  "
        f"transposed-stays-in-range = {in_range}"
    )
    assert correct.emitted_unsupplied == 0
    assert in_range, "the transposed index left the extent, so a range check would suffice"
    assert mutant.emitted_unsupplied > 0, (
        "the emitted-unsupplied counter returned 0 for a deliberately transposed "
        "flattening -- it cannot fail, so its zero measures nothing"
    )


def test_part4_control_dropped_counter_moves_under_a_non_pow2_ratio() -> None:
    """D1.5 control: the dropped counted zero can return non-zero."""
    weights, scales, _ = referent_fixture(512, 512, perturb=((1, 1), "x1.5"))
    mutant = producer.retile_block_scales(weights, scales, producer.DOWN)
    clean_w, clean_s = build_pow2_checkpoint(512, 512)
    clean = producer.retile_block_scales(clean_w, clean_s, producer.DOWN)
    print(
        f"[P4-control-B] clean dropped = {clean.input_scales_dropped}  "
        f"x1.5 dropped = {mutant.input_scales_dropped}  "
        f"x1.5 inexact rescales = {mutant.inexact_rescales}"
    )
    assert clean.input_scales_dropped == 0
    assert mutant.input_scales_dropped > 0, (
        "the dropped counter returned 0 for a scale the layout cannot represent"
    )


# --------------------------------------------------------------------------- #
# PART 5 -- the COMPLETE losslessness condition, k/N per declared shape.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("side", "expected_blocks"), [(512, 4), (1024, 16)])
def test_part5_complete_losslessness_condition_k_over_n(
    side: int, expected_blocks: int
) -> None:
    """Part 5: BOTH conjuncts, by the bit-pattern predicate, reported as ``k/N``.

    Conjunct 1 -- the four constituent ``[128,128]`` scales are mutually
    power-of-two related (three ratio constraints, in the declared order
    ``((0,1),(1,0),(1,1))``). Conjunct 2 -- the retained ``256``-block scale is
    ITSELF a power of two, which is owed because the consumer applies the scale
    after accumulating two ``i_tiles`` in PSUM rather than elementwise.

    On synthetic fixtures built to satisfy both, ``k/N == 100%``.

    Certifying component:
    ``blockwise_fp8_retile.RetiledBlockScales.losslessness_fraction`` over
    ``BlockLosslessnessRecord.quad_mutually_pow2`` and ``.block_scale_pow2``.
    """
    weights, scales = build_pow2_checkpoint(side, side)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    k, n = result.losslessness_fraction()
    conjunct1 = sum(1 for record in result.records if record.quad_mutually_pow2)
    conjunct2 = sum(1 for record in result.records if record.block_scale_pow2)
    ratios_checked = sum(len(record.ratios) for record in result.records)
    percent = 100.0 * k / n

    print(
        f"[P5] [{side},{side}] k = {k} (int)  N = {n}  k/N = {percent:.1f}%  "
        f"conjunct1 = {conjunct1}/{n}  conjunct2 = {conjunct2}/{n}  "
        f"ratio constraints checked = {ratios_checked}"
    )
    assert isinstance(k, int)
    assert n == expected_blocks, f"[{side},{side}] gave {n} 256-blocks, expected {expected_blocks}"
    assert ratios_checked == 3 * n, f"{ratios_checked} constraints over {n} blocks, expected {3 * n}"
    for record in result.records:
        assert record.ratios_pow2 == (True, True, True), (
            f"block {record.key} ratios {record.ratios} are not all powers of two"
        )
    assert conjunct1 == n, f"conjunct 1 holds on {conjunct1}/{n}, required {n}/{n}"
    assert conjunct2 == n, f"conjunct 2 holds on {conjunct2}/{n}, required {n}/{n}"
    assert k == n, f"k/N = {k}/{n} = {percent:.1f}%, required 100%"


def test_part5_ratio_order_is_the_declared_order() -> None:
    """The three ratio constraints are evaluated in the declared order."""
    weights, scales = build_pow2_checkpoint(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    print(
        f"[P5-order] RATIO_ORDER producer = {producer.RATIO_ORDER}  "
        f"referent = {referent.RATIO_ORDER}"
    )
    assert producer.RATIO_ORDER == ((0, 1), (1, 0), (1, 1))
    assert producer.RATIO_ORDER == referent.RATIO_ORDER
    assert all(len(record.ratios) == 3 for record in result.records)


def test_part5_counted_zero_no_unrecorded_conjunct2_failure() -> None:
    """Part 5's counted zero: ``0`` blocks where conjunct 1 holds while conjunct 2
    fails without being recorded.

    Certifying component:
    ``blockwise_fp8_retile.count_unrecorded_conjunct2_failures``, whose population
    is recomputed straight from ``scales`` independently of the record list.
    """
    weights, scales, _ = referent_fixture(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    unrecorded = producer.count_unrecorded_conjunct2_failures(scales, result.records)
    population = sum(
        1
        for record in result.records
        if record.quad_mutually_pow2 and not record.block_scale_pow2
    )
    print(
        f"[P5-zero] conjunct1-holds/conjunct2-fails blocks = {population}  "
        f"unrecorded = {unrecorded}"
    )
    assert population > 0, (
        "the fixture has no conjunct1-holds/conjunct2-fails block, so this zero "
        "would be measured over an empty population"
    )
    assert unrecorded == 0, f"{unrecorded} such blocks went unrecorded"


def test_part5_control_unrecorded_counter_moves_when_a_record_is_dropped() -> None:
    """D1.5 control: part 5's counted zero can return non-zero."""
    weights, scales, _ = referent_fixture(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    full = producer.count_unrecorded_conjunct2_failures(scales, result.records)
    truncated = producer.count_unrecorded_conjunct2_failures(
        scales, result.records[1:]
    )
    print(
        f"[P5-control] unrecorded with all {len(result.records)} records = {full}  "
        f"with one dropped = {truncated}"
    )
    assert full == 0
    assert truncated > 0, (
        "dropping a record left the unrecorded count at 0 -- the counter restates "
        "how the record list was built instead of measuring it"
    )


def test_part5_detector_conjunct1_holds_while_conjunct2_fails_counts_as_failing() -> None:
    """THE DETECTOR ARM. Without it part 5 is a tautology over scales this file
    chose.

    A ``256`` block whose four constituent scales are mutually power-of-two
    related while the retained block scale is NOT a power of two must be counted
    as FAILING part 5, in ``1/1`` cases. The referent's default fixture is exactly
    that block -- measured at the parent by ``probe-024-parent-readings.py`` R6
    (``conjunct1_holds_conjunct2_fails = 4``), so the subject pre-exists this
    increment and was not constructed to order.
    """
    weights, scales, _ = referent_fixture(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    block = next(record for record in result.records if record.key == (0, 0, 0))
    k, n = result.losslessness_fraction()
    rejected = int(
        block.quad_mutually_pow2 and not block.block_scale_pow2 and not block.lossless
    )
    print(
        f"[P5-detector] block {block.key}: S = {block.block_scale!r}  "
        f"conjunct1 = {block.quad_mutually_pow2}  "
        f"conjunct2 = {block.block_scale_pow2}  "
        f"ratios = {block.ratios}  lossless = {block.lossless}  "
        f"counted as failing = {rejected}/1  k/N over the fixture = {k}/{n}"
    )
    assert block.quad_mutually_pow2 is True, "conjunct 1 must HOLD for this to be the detector case"
    assert block.block_scale_pow2 is False, "conjunct 2 must FAIL for this to be the detector case"
    assert block.lossless is False, (
        "a block satisfying conjunct 1 alone was reported lossless -- part 5 "
        "reduces to conjunct 1 and the second conjunct is decoration"
    )
    assert rejected == 1, f"the detector block was counted as failing {rejected}/1 times, required 1/1"
    assert k == 0, f"k/N = {k}/{n} on a fixture where conjunct 2 fails on every block"


# --------------------------------------------------------------------------- #
# The declared negative case -- a NAMED refusal, not a silent truncation.      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rows", "cols"), [(512, 384), (384, 384), (384, 512), (512, 0)]
)
def test_negative_case_non_divisible_shape_raises_the_named_error(
    rows: int, cols: int
) -> None:
    """A shape that is not a whole number of ``256`` blocks RAISES rather than
    silently truncating -- and raises this producer's own NAMED error.

    Door R7: the referent already raises a bare ``AssertionError`` on the same
    shapes at the parent, so asserting on ``Exception`` would pass before this
    increment existed. The assertion is on ``BlockwiseFp8RetileError``, which had
    0 occurrences in the tree at ``382091c``.

    Certifying component: ``blockwise_fp8_retile._require_blocked`` reached through
    ``consumer_scale_shape`` and ``retile_block_scales``.
    """
    with pytest.raises(producer.BlockwiseFp8RetileError) as shape_error:
        producer.consumer_scale_shape(1, rows, cols, producer.DOWN)
    truncated_h, truncated_i = rows // _BLOCK, cols // _BLOCK
    print(
        f"[NEG] [{rows},{cols}] raised {type(shape_error.value).__name__}  "
        f"silent truncation would have emitted a "
        f"{truncated_h}x{truncated_i}-block layout  "
        f"message names the extent = {str(rows) in str(shape_error.value)}"
    )
    assert isinstance(shape_error.value, ValueError)
    assert str(BLOCK := producer.BLOCK_QUANT_SIZE) in str(shape_error.value)
    assert BLOCK == 256


def test_negative_case_control_a_divisible_shape_does_not_raise() -> None:
    """D1.5 control for the negative case: the refusal is conditional, not
    unconditional. Without this the raise arm passes for a producer that refuses
    everything."""
    shape = producer.consumer_scale_shape(1, 512, 512, producer.DOWN)
    weights, scales = build_pow2_checkpoint(512, 512)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    print(
        f"[NEG-control] [512,512] did NOT raise; shape = {shape}  "
        f"records = {len(result.records)}"
    )
    assert shape == expected_down_shape(1, 512, 512)
    assert len(result.records) == 4


def test_negative_case_a_mismatched_scale_grid_is_refused() -> None:
    """The same named refusal covers a scale grid that does not match the weight,
    which is the other way a caller can silently lose scales."""
    weights, scales = build_pow2_checkpoint(512, 512)
    with pytest.raises(producer.BlockwiseFp8RetileError):
        producer.retile_block_scales(weights, scales[:, :2, :], producer.DOWN)
    with pytest.raises(producer.BlockwiseFp8RetileError):
        producer.retile_block_scales(weights, scales.to(torch.float64), producer.DOWN)
    with pytest.raises(producer.BlockwiseFp8RetileError):
        producer.retile_block_scales(weights[0], scales[0], producer.DOWN)
    print("[NEG-grid] mismatched grid, wrong dtype and wrong rank all refused by name")


# --------------------------------------------------------------------------- #
# ADDITIONAL -- the expert axis, so the bank is not silently single-expert.     #
# --------------------------------------------------------------------------- #
def test_expert_axis_is_retiled_independently_per_expert() -> None:
    """ADDITIONAL, not a declared conjunct. Distinct per-expert scale sets must
    produce distinct emitted columns, so a producer that broadcast expert 0 over
    the bank would fail here."""
    weights, scales = build_pow2_checkpoint(512, 512, experts=3)
    result = producer.retile_block_scales(weights, scales, producer.DOWN)
    k, n = result.losslessness_fraction()
    per_expert = {
        expert: tuple(result.block_scales[expert].flatten().tolist())
        for expert in range(3)
    }
    distinct = len(set(per_expert.values()))
    print(
        f"[experts] N over the bank = {n} (4 per expert x 3)  k = {k}  "
        f"distinct per-expert mappings = {distinct}/3  "
        f"emitted_unsupplied = {result.emitted_unsupplied}  "
        f"dropped = {result.input_scales_dropped}"
    )
    assert n == 12, f"3 experts x 4 blocks gave {n} records"
    assert k == n
    assert distinct == 3, "the experts share a mapping -- expert 0 was broadcast"
    assert result.emitted_unsupplied == 0
    assert result.input_scales_dropped == 0
