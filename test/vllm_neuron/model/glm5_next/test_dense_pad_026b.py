# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-026b`` -- PAD-TO-128 and slice-back at the seam.

Acceptance command (plan block ``#### inc-glm53f-026b``, taken AS WRITTEN)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \
    python -m pytest test/vllm_neuron/model/glm5_next/test_dense_pad_026b.py \
      -s -rA -p no:cacheprovider

WHAT THE INCREMENT FIXES. The dense block-FP8 seam tiles ``M`` over the PSUM
partition axis and does not pad, so it refuses any token count that is not a
positive multiple of ``TILE_SIZE`` and says in its own message that padding is the
caller's (``functional/blockwise_fp8_mm.py:239-245``). No pad adapter existed
anywhere in the model package, so a ``[1, H]`` decode step could not run at all --
it raised. Both dense paths now pad up to a whole tile at their entry and slice the
result back at their single return.

**THE SEAM ITSELF IS NOT TOUCHED.** ``_require_blocked`` stays unconditional,
exactly as its own docstring argues: a geometry the kernel cannot serve must raise
rather than fall back, because falling back would ship a torch path for
kernel-class work (``blockwise_fp8_mm.py:386-397``, P13 and D6). This file asserts
that too -- the refusal still fires on a short count reaching the seam directly.

THE CRITERION IS SEAM-LOCAL, because the pad never leaves the MLP call: a
``[1, H]`` step reaches the seam and the sliced ``[1, H]`` result equals the
unpadded oracle's first row, the oracle being the ``[128, H]`` run's first row.

THE THREE ITEMS
1. **the dense MLP** -- ``Glm5NextDenseMLP.forward``, three dispatches.
2. **the shared expert** -- ``Glm5NextSharedExperts.shared_expert_mm``, three
   dispatches, and it reaches the seam through the prebuilt scale operand, which
   is a different call form and so a separate reading rather than a repeat.
3. **the seam's refusal, unweakened** -- a short count handed straight to the seam
   still raises on ``M``. Without this reading the increment could have been
   realised by weakening ``_require_blocked``, which is the one way to make items
   1 and 2 pass while shipping the defect P13 and D6 forbid.

EACH ITEM PRINTS, as counted readings rather than as prose:
  * the rows the seam SAW for each dispatch (``128``) against the rows the caller
    passed (``1``) -- the counted ``128`` the design asks for;
  * the pad rows carrying the sentinel EXACTLY, per dispatch. The first two
    dispatches consume the caller's activation and see ``127`` sentinel rows; the
    third consumes the SwiGLU intermediate, whose pad rows are the math's own and
    not the sentinel, so its reading is ``0`` by construction and is stated that
    way rather than hidden;
  * ``0`` sentinel-derived elements in the returned tensor, which is the whole
    claim of the slice: the return is byte-identical to the seam's own output
    restricted to the caller's rows;
  * the worst relative error against the unpadded oracle, and the count of
    elements outside the registered pair;
  * the route predicate, read from the seam's OWN counters: ``3`` dispatches per
    call and ``0`` torch fallbacks (D13 form R-1).

THE FAILING CONTROL, and it has a mechanism: ``_unpad_rows`` is replaced by a
passthrough -- the slice-back and nothing else is removed -- and the same call then
returns ``[128, H]`` where ``[1, H]`` was asked for, with ``127`` pad-derived rows
in the result. Both directions run in the same item and both are printed. The
control acts on a step this increment authors, which is why it can fail; the
counters the design review struck at revision 282 could not.

THE TOLERANCE IS THE CAMPAIGN'S REGISTERED fp8 PAIR AND IS NOT RE-REGISTERED:
``rtol=3e-2, atol=1e-5``, order named inline per design law D3, the pair
``inc-glm53f-005`` registered. The predicate is spelled out here rather than
delegated, because ``_DEFAULT_DTYPE_TOLERANCE`` has NO fp8 entry and an omitted
pair silently inherits the bf16 one (PIT-13). The worst relative error is REPORTED
as a number either way.

THE PLATFORM IS PRINTED AND GATES NOTHING. ``test/conftest.py`` carries the trn2
default for this suite; the acceptance command adds no platform override, per the
plan block as written, and this file reads the value only to record it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import torch

#: Asked of the one definition rather than retyped: the seam re-exports both.
from vllm_neuron.functional.blockwise_fp8_mm import BLOCK_QUANT_SIZE, TILE_SIZE

#: ``H`` and ``I``: two whole ``BLOCK_QUANT_SIZE`` columns each, the smallest
#: extents the public scale grid can express for both projection shapes.
HIDDEN = 2 * BLOCK_QUANT_SIZE
INTERMEDIATE = 2 * BLOCK_QUANT_SIZE

#: The count under test, and the count the oracle runs. ``1`` is the decode step
#: the seam refused; ``TILE_SIZE`` is the smallest count it ever accepted, so the
#: oracle is the shipped path on the shape it already ran.
ONE_TOKEN = 1
ORACLE_TOKENS = TILE_SIZE

#: The pad this increment writes, asked of the module rather than retyped, so a
#: changed sentinel cannot leave this file asserting the old one.
SENTINEL_ATTRIBUTE = "_TOKEN_PAD_SENTINEL"

RTOL = 3e-2
ATOL = 1e-5

SEED_HIDDEN, SEED_GATE, SEED_UP, SEED_DOWN = 2601, 2602, 2603, 2604

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

PROJECTIONS = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


class VacuousControlError(AssertionError):
    """A control that cannot fail, which makes the item it guards meaningless."""


def say(tag: str, *values) -> None:
    """One printed line per read value. ``-s`` is in the acceptance command."""
    print("DENSEPAD|" + tag + "|" + "|".join(str(v) for v in values))


def _impl():
    """Import the modeling module INSIDE a test body, this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _seam_module():
    """The seam MODULE, not the function.

    ``import_module`` rather than ``import ... as``: the package re-exports the
    seam FUNCTION under the submodule's own name, so the plain import binds the
    function and ``seam.blockwise_fp8_mm`` would raise ``AttributeError``. The
    module object is what both methods' function-local ``from ... import ...``
    reads its name out of, so patching it is what reaches the six call sites
    (``test_shared_expert_scale_prep.py`` records the same trap).
    """
    return importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")


def _pinned_raw_config() -> dict:
    """The pinned fixture config, digest-checked, so nothing here is hand-fed."""
    raw = FIXTURE_PATH.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise VacuousControlError(
            f"{FIXTURE_PATH} digests to {got}, not the campaign's registered "
            f"{FIXTURE_SHA256}; the quant config under test would not be the pin's"
        )
    return json.loads(raw.decode())


def _quant_config():
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _text_config():
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=INTERMEDIATE,
        n_shared_experts=1,
        num_key_value_heads=2,
    )


def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``.

    Unsigned and bf16-exact, the tiny suite's primitive, so the seam's cast of the
    activations introduces nothing this item would then have to explain away.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_grid(exponent: int, rows: int, cols: int) -> torch.Tensor:
    """A ``[rows, cols]`` fp32 grid of one exact power of two.

    Exact, so no scale can contribute a rounding difference between the padded run
    and the oracle -- the comparison is then about the pad and the slice, which is
    what this file is for.
    """
    return torch.full((rows, cols), float(2.0**exponent), dtype=torch.float32)


def _operands() -> dict:
    """The three weights, the three public grids, and one row of activations."""
    fp8 = torch.float8_e4m3fn
    k_parallel = HIDDEN // BLOCK_QUANT_SIZE
    n_parallel = INTERMEDIATE // BLOCK_QUANT_SIZE
    return {
        "row": _fp8_grid_values(SEED_HIDDEN, ONE_TOKEN, HIDDEN).to(torch.bfloat16),
        "gate_proj_weight": (
            _fp8_grid_values(SEED_GATE, HIDDEN, INTERMEDIATE).to(fp8),
            _pow2_grid(-1, k_parallel, n_parallel),
        ),
        "up_proj_weight": (
            _fp8_grid_values(SEED_UP, HIDDEN, INTERMEDIATE).to(fp8),
            _pow2_grid(0, k_parallel, n_parallel),
        ),
        "down_proj_weight": (
            _fp8_grid_values(SEED_DOWN, INTERMEDIATE, HIDDEN).to(fp8),
            _pow2_grid(-1, n_parallel, k_parallel),
        ),
    }


def _scale_grid_attribute(leaf: str) -> str:
    """The grid's attribute name, ASKED of its one definition rather than retyped."""
    return _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _dense_module(operands: dict):
    """A production ``Glm5NextDenseMLP`` carrying the three declared pairs."""
    module = _impl().Glm5NextDenseMLP(_text_config())
    for leaf in PROJECTIONS:
        weight, grid = operands[leaf]
        setattr(module, leaf, torch.nn.Parameter(weight.clone(), requires_grad=False))
        setattr(module, _scale_grid_attribute(leaf), grid.clone())
    return module


def _shared_module(operands: dict):
    """A production ``Glm5NextSharedExperts`` with its load-time operands built."""
    module = _impl().Glm5NextSharedExperts(_text_config())
    module.prepare_scale_operands(
        operands["gate_proj_weight"][0],
        operands["up_proj_weight"][0],
        operands["down_proj_weight"][0],
        operands["gate_proj_weight"][1],
        operands["up_proj_weight"][1],
        operands["down_proj_weight"][1],
    )
    return module


def _run_dense(module, hidden: torch.Tensor) -> torch.Tensor:
    return module.forward(hidden, quant_config=_quant_config())


def _run_shared(module, operands: dict, hidden: torch.Tensor) -> torch.Tensor:
    return module.shared_expert_mm(
        hidden_states=hidden,
        gate_proj_weight=operands["gate_proj_weight"][0],
        up_proj_weight=operands["up_proj_weight"][0],
        down_proj_weight=operands["down_proj_weight"][0],
        gate_proj_scale=operands["gate_proj_weight"][1],
        up_proj_scale=operands["up_proj_weight"][1],
        down_proj_scale=operands["down_proj_weight"][1],
        quant_config=_quant_config(),
    )


def _record_the_seam(monkeypatch: pytest.MonkeyPatch, sentinel: float) -> dict:
    """Watch every dispatch: the rows it saw, its sentinel rows, its raw output.

    A WATCHER, not a stand-in: it forwards to the real seam, so the kernel runs,
    the seam's own dispatch counters move, and the numbers below are the shipped
    path's. Both call forms are accepted -- the dense path passes three positional
    operands, the shared path adds ``prebuilt_scale_t`` by keyword.
    """
    seam = _seam_module()
    real = seam.blockwise_fp8_mm
    log: dict = {"rows": [], "sentinel_rows": [], "outputs": []}

    def watcher(x, weight, weight_scale, *, prebuilt_scale_t=None):
        log["rows"].append(int(x.shape[0]))
        log["sentinel_rows"].append(int((x == sentinel).all(dim=1).sum()))
        out = real(x, weight, weight_scale, prebuilt_scale_t=prebuilt_scale_t)
        log["outputs"].append(out)
        return out

    monkeypatch.setattr(seam, "blockwise_fp8_mm", watcher)
    return log


def _worst_relative_error(got: torch.Tensor, want: torch.Tensor) -> tuple[float, int]:
    """``(worst ratio, elements outside)`` against ``atol + rtol * |want|``.

    Spelled out because ``_DEFAULT_DTYPE_TOLERANCE`` has no fp8 entry (PIT-13): a
    delegated comparison would silently inherit the bf16 pair.
    """
    a = got.to(torch.float32)
    b = want.to(torch.float32)
    allowed = ATOL + RTOL * b.abs()
    gap = (a - b).abs()
    ratio = gap / allowed
    return float(ratio.max()), int((gap > allowed).sum())


def _platform_as_read() -> str:
    """The platform this process read. RECORDED, and it gates nothing here."""
    return os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "<unset-in-environ>")


def _one_item(
    monkeypatch: pytest.MonkeyPatch, label: str, run, module, hidden: torch.Tensor
) -> None:
    """The six readings and the control, for one of the two call sites."""
    model_fp8 = _impl()
    seam = _seam_module()
    sentinel = float(getattr(model_fp8, SENTINEL_ATTRIBUTE))
    say(label, "PLATFORM", _platform_as_read())
    say(label, "SENTINEL", sentinel)

    # ---- the unpadded ORACLE: the shipped path on the only count it ever ran.
    oracle_rows = hidden.repeat(ORACLE_TOKENS, 1)
    seam.reset_dispatch_counters()
    oracle = run(module, oracle_rows)
    oracle_dispatches, oracle_fallbacks = seam.dispatch_counters()
    say(label, "ORACLE", tuple(oracle.shape), str(oracle.dtype),
        oracle_dispatches, oracle_fallbacks)
    if tuple(oracle.shape) != (ORACLE_TOKENS, HIDDEN):
        raise VacuousControlError(
            f"the oracle returned {tuple(oracle.shape)}, not "
            f"({ORACLE_TOKENS}, {HIDDEN}); there would be no reference row"
        )

    # ---- THE ITEM: one token through the same path, watched.
    log = _record_the_seam(monkeypatch, sentinel)
    seam.reset_dispatch_counters()
    got = run(module, hidden)
    dispatches, fallbacks = seam.dispatch_counters()

    say(label, "SEAM_SAW", log["rows"], "caller_passed", int(hidden.shape[0]))
    say(label, "SENTINEL_ROWS", log["sentinel_rows"])
    if log["rows"] != [ORACLE_TOKENS] * 3:
        raise AssertionError(
            f"the seam saw {log['rows']} rows per dispatch; the pad must make "
            f"every one of the three a whole tile of {ORACLE_TOKENS}"
        )
    if log["sentinel_rows"][:2] != [ORACLE_TOKENS - ONE_TOKEN] * 2:
        raise AssertionError(
            f"the two projections that consume the caller's activation saw "
            f"{log['sentinel_rows'][:2]} sentinel rows, not "
            f"{[ORACLE_TOKENS - ONE_TOKEN] * 2}"
        )
    if log["sentinel_rows"][2] != 0:
        raise AssertionError(
            "the down projection consumes the SwiGLU intermediate, whose pad rows "
            f"are the math's own; it must see 0 sentinel rows, saw "
            f"{log['sentinel_rows'][2]}"
        )

    # ---- READING: no pad row survives into the result, and none is derived from
    # one -- the return is the seam's own output restricted to the caller's rows.
    raw = log["outputs"][-1]
    survivors = int(raw.shape[0]) - int(got.shape[0])
    identical = bool(torch.equal(got, raw[: int(got.shape[0])]))
    sentinel_derived = 0 if (identical and tuple(got.shape) == (ONE_TOKEN, HIDDEN)) else (
        survivors * HIDDEN
    )
    say(label, "SLICED", tuple(got.shape), "seam_returned", tuple(raw.shape),
        "sentinel_derived_elements", sentinel_derived, "byte_identical_prefix",
        identical)
    if tuple(got.shape) != (ONE_TOKEN, HIDDEN):
        raise AssertionError(
            f"the caller asked for {(ONE_TOKEN, HIDDEN)} and received "
            f"{tuple(got.shape)}; the pad left the call"
        )
    if not identical or sentinel_derived != 0:
        raise AssertionError(
            f"the returned rows are not the seam's own first {ONE_TOKEN} rows; "
            f"sentinel_derived_elements={sentinel_derived}"
        )

    # ---- READING: the criterion.
    worst, outside = _worst_relative_error(got[0], oracle[0])
    say(label, "AGREEMENT", "worst_ratio", "%.6f" % worst, "elements_outside",
        outside, "of", int(oracle[0].numel()), "rtol", RTOL, "atol", ATOL)
    if outside != 0:
        raise AssertionError(
            f"{outside} elements of the padded one-token result differ from the "
            f"unpadded oracle's first row by more than atol + rtol * |want| "
            f"(rtol={RTOL}, atol={ATOL})"
        )

    # ---- READING: the route predicate, from the seam's own counters (D13, R-1).
    say(label, "ROUTE", "nki_dispatch", dispatches, "torch_fallback", fallbacks)
    if (dispatches, fallbacks) != (3, 0):
        raise AssertionError(
            f"the route predicate reads (nki_dispatch={dispatches}, "
            f"torch_fallback={fallbacks}); this call declares (3, 0)"
        )

    # ---- THE CONTROL: remove the slice-back and nothing else.
    monkeypatch.setattr(model_fp8, "_unpad_rows", lambda out, tokens: out)
    unsliced = run(module, hidden)
    pad_rows_returned = int(unsliced.shape[0]) - ONE_TOKEN
    say(label, "CONTROL_SLICE_SKIPPED", tuple(unsliced.shape), "pad_rows_returned",
        pad_rows_returned, "sentinel_derived_elements", pad_rows_returned * HIDDEN)
    if tuple(unsliced.shape) == (ONE_TOKEN, HIDDEN):
        raise VacuousControlError(
            "removing the slice-back changed nothing the caller can see; the "
            "control cannot fail and this item would be meaningless"
        )
    if pad_rows_returned != ORACLE_TOKENS - ONE_TOKEN:
        raise VacuousControlError(
            f"the control returned {pad_rows_returned} extra rows, not "
            f"{ORACLE_TOKENS - ONE_TOKEN}; it is not the pad that survived"
        )


# --------------------------------------------------------------------------- #
# ITEM 1 of 3 -- ``Glm5NextDenseMLP.forward``.                                 #
# --------------------------------------------------------------------------- #
def test_the_dense_mlp_runs_a_one_token_step_and_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One token through the dense MLP: padded at entry, sliced at the return."""
    operands = _operands()
    _one_item(monkeypatch, "DENSE", _run_dense, _dense_module(operands),
              operands["row"])


# --------------------------------------------------------------------------- #
# ITEM 2 of 3 -- ``Glm5NextSharedExperts.shared_expert_mm``.                    #
# --------------------------------------------------------------------------- #
def test_the_shared_expert_runs_a_one_token_step_and_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same, through the prebuilt-scale-operand call form."""
    operands = _operands()
    module = _shared_module(operands)

    def run(mod, hidden):
        return _run_shared(mod, operands, hidden)

    _one_item(monkeypatch, "SHARED", run, module, operands["row"])


# --------------------------------------------------------------------------- #
# ITEM 3 of 3 -- the seam's refusal is UNTOUCHED.                               #
# --------------------------------------------------------------------------- #
def test_the_seam_still_refuses_a_short_token_count_by_itself() -> None:
    """The pad is the caller's; the refusal stays unconditional (P13, D6).

    Without this reading the increment could have been realised by weakening
    ``_require_blocked``, which is the defect that guard exists to prevent. The
    refusal is asked for directly, at the seam, with no MLP in the way.
    """
    seam = _seam_module()
    operands = _operands()
    row = operands["row"]
    weight, grid = operands["gate_proj_weight"]
    with pytest.raises(seam.BlockwiseFp8MmError) as caught:
        seam.blockwise_fp8_mm(row, weight, grid)
    message = str(caught.value)
    say("SEAM", "REFUSAL_STILL_FIRES", f"M={int(row.shape[0])}",
        "TILE_SIZE", TILE_SIZE, "mentions_M", f"M={int(row.shape[0])}" in message)
    if f"M={int(row.shape[0])}" not in message:
        raise AssertionError(
            f"the seam refused but not on the token count: {message!r}"
        )
