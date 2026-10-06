# SPDX-License-Identifier: Apache-2.0
"""Short dense steps run unpadded; larger partial tiles keep their padding."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
import torch

#: Asked of the one definition rather than retyped: the seam re-exports both.
from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE, TILE_SIZE

#: ``H`` and ``I``: four whole ``SCALE_BLOCK_SIZE`` columns each. The dense path
#: narrowed the granularity from 256 to 128, so this is no longer the smallest
#: legal geometry -- the extent is held at 512 on purpose, so this padding test
#: keeps the geometry it was written for.
HIDDEN = 4 * SCALE_BLOCK_SIZE
INTERMEDIATE = 4 * SCALE_BLOCK_SIZE

#: Compare the new single-token route with the existing whole-tile route.
ONE_TOKEN = 1
ORACLE_TOKENS = TILE_SIZE

SENTINEL_ATTRIBUTE = "_TOKEN_PAD_SENTINEL"

RTOL = 3e-2
ATOL = 1e-5

SEED_HIDDEN, SEED_GATE, SEED_UP, SEED_DOWN = 2601, 2602, 2603, 2604

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

PROJECTIONS = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


class VacuousControlError(AssertionError):
    """A control that cannot fail, which makes the test it guards meaningless."""


def _impl():
    """Import the modeling module inside a test body, this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _seam_module():
    """The seam module, not the function. """
    return importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")


def _pinned_raw_config() -> dict:
    """The fixture config, so nothing here is hand-fed."""
    return json.loads(FIXTURE_PATH.read_bytes().decode())


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
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``. """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_grid(exponent: int, rows: int, cols: int) -> torch.Tensor:
    """A ``[rows, cols]`` fp32 grid of one exact power of two. """
    return torch.full((rows, cols), float(2.0**exponent), dtype=torch.float32)


def _operands() -> dict:
    """The three weights, the three public grids, and one row of activations."""
    fp8 = torch.float8_e4m3fn
    k_parallel = HIDDEN // SCALE_BLOCK_SIZE
    n_parallel = INTERMEDIATE // SCALE_BLOCK_SIZE
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
    """The grid's attribute name, asked of its one definition rather than retyped."""
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
    """Watch every dispatch: the rows it saw, its sentinel rows, its raw output. """
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


def _record_the_fused_seam(monkeypatch: pytest.MonkeyPatch, sentinel: float) -> dict:
    """The same watch on the fused MLP call short steps now make. """
    seam = _seam_module()
    real = seam.blockwise_fp8_mlp
    log: dict = {"rows": [], "sentinel_rows": [], "outputs": []}

    def watcher(x, *args, **kwargs):
        log["rows"].append(int(x.shape[0]))
        log["sentinel_rows"].append(int((x == sentinel).all(dim=1).sum()))
        out = real(x, *args, **kwargs)
        log["outputs"].append(out)
        return out

    monkeypatch.setattr(seam, "blockwise_fp8_mlp", watcher)
    return log


def _worst_relative_error(got: torch.Tensor, want: torch.Tensor) -> tuple[float, int]:
    """``(worst ratio, elements outside)`` against ``atol + rtol * |want|``. """
    a = got.to(torch.float32)
    b = want.to(torch.float32)
    allowed = ATOL + RTOL * b.abs()
    gap = (a - b).abs()
    ratio = gap / allowed
    return float(ratio.max()), int((gap > allowed).sum())


def _one_item(
    monkeypatch: pytest.MonkeyPatch, label: str, run, module, hidden: torch.Tensor
) -> None:
    """The six readings and the control, for one of the two call sites."""
    model_fp8 = _impl()
    seam = _seam_module()
    sentinel = float(getattr(model_fp8, SENTINEL_ATTRIBUTE))

    # ---- The existing whole-tile path remains the comparison oracle.
    oracle_rows = hidden.repeat(ORACLE_TOKENS, 1)
    seam.reset_dispatch_counters()
    oracle = run(module, oracle_rows)
    if tuple(oracle.shape) != (ORACLE_TOKENS, HIDDEN):
        raise VacuousControlError(
            f"the oracle returned {tuple(oracle.shape)}, not "
            f"({ORACLE_TOKENS}, {HIDDEN}); there would be no reference row"
        )

    # ---- one token through the same path, watched. Short steps now make one
    # fused MLP call (gate, up, SwiGLU, down in one kernel), so the watch is on
    # that call; the three-call seam must not be entered at all.
    log = _record_the_seam(monkeypatch, sentinel)
    fused = _record_the_fused_seam(monkeypatch, sentinel)
    seam.reset_dispatch_counters()
    seam.reset_mlp_dispatch_counters()
    got = run(module, hidden)
    dispatches, fallbacks = seam.mlp_dispatch_counters()

    if log["rows"] != []:
        raise AssertionError(
            f"the three-call seam saw {log['rows']} rows; a short step must take "
            f"the fused call"
        )
    if fused["rows"] != [ONE_TOKEN]:
        raise AssertionError(
            f"the fused seam saw {fused['rows']} rows per dispatch; the short "
            f"MLP must run on {ONE_TOKEN} real row"
        )
    if fused["sentinel_rows"] != [0]:
        raise AssertionError(
            f"the fused call that consumes the caller's activation saw "
            f"{fused['sentinel_rows']} sentinel rows; short steps must not pad"
        )

    # ---- No hidden padded output was computed or sliced away.
    raw = fused["outputs"][-1]
    survivors = int(raw.shape[0]) - int(got.shape[0])
    identical = bool(torch.equal(got, raw[: int(got.shape[0])]))
    sentinel_derived = 0 if (identical and tuple(got.shape) == (ONE_TOKEN, HIDDEN)) else (
        survivors * HIDDEN
    )
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
    if survivors != 0:
        raise AssertionError(f"the short kernel still computed {survivors} padded rows")

    # ---- reading: the criterion.
    _worst, outside = _worst_relative_error(got[0], oracle[0])
    if outside != 0:
        raise AssertionError(
            f"{outside} elements of the unpadded one-token result differ from the "
            f"whole-tile oracle's first row by more than atol + rtol * |want| "
            f"(rtol={RTOL}, atol={ATOL})"
        )

    if (dispatches, fallbacks) != (1, 0) or seam.dispatch_counters() != (0, 0):
        raise AssertionError(
            f"the route predicate reads fused (nki_dispatch={dispatches}, "
            f"torch_fallback={fallbacks}) and three-call "
            f"{seam.dispatch_counters()}; this call declares (1, 0) and (0, 0)"
        )

    # ---- Larger partial prefill tiles still pad. Check their sentinel rows
    # and retain a non-vacuous slice-back control on that route.
    partial_tokens = TILE_SIZE + ONE_TOKEN
    partial_hidden = hidden.repeat(partial_tokens, 1)
    log["rows"].clear()
    log["sentinel_rows"].clear()
    partial = run(module, partial_hidden)
    if log["rows"] != [2 * TILE_SIZE] * 3:
        raise AssertionError(f"the partial prefill did not pad: {log['rows']}")
    if log["sentinel_rows"][:2] != [TILE_SIZE - ONE_TOKEN] * 2:
        raise AssertionError(
            f"the partial prefill lost its sentinel rows: {log['sentinel_rows']}"
        )
    if tuple(partial.shape) != (partial_tokens, HIDDEN):
        raise AssertionError(f"the partial prefill returned padded rows: {partial.shape}")
    torch.testing.assert_close(partial[0], oracle[0], rtol=RTOL, atol=ATOL)
    monkeypatch.setattr(model_fp8, "_unpad_rows", lambda out, tokens: out)
    unsliced = run(module, partial_hidden)
    pad_rows_returned = int(unsliced.shape[0]) - partial_tokens
    if tuple(unsliced.shape) == (partial_tokens, HIDDEN):
        raise VacuousControlError(
            "removing the slice-back changed nothing the caller can see; the "
            "control cannot fail and this test would be meaningless"
        )
    if pad_rows_returned != TILE_SIZE - ONE_TOKEN:
        raise VacuousControlError(
            f"the control returned {pad_rows_returned} extra rows, not "
            f"{TILE_SIZE - ONE_TOKEN}; it is not the pad that survived"
        )


# --------------------------------------------------------------------------- #
# check 1 of 3 -- ``Glm5NextDenseMLP.forward``.                                 #
# --------------------------------------------------------------------------- #
def test_the_dense_mlp_runs_a_one_token_step_and_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense decode runs one real row; partial prefill retains padding."""
    operands = _operands()
    _one_item(monkeypatch, "DENSE", _run_dense, _dense_module(operands),
              operands["row"])


# --------------------------------------------------------------------------- #
# check 2 of 3 -- ``Glm5NextSharedExperts.shared_expert_mm``.                    #
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
# check 3 of 3 -- the seam now accepts a single token directly.                  #
# --------------------------------------------------------------------------- #
def test_the_seam_accepts_a_short_token_count_by_itself() -> None:
    """Direct single-token dispatch matches an independent CPU oracle."""
    seam = _seam_module()
    operands = _operands()
    row = operands["row"]
    weight, grid = operands["gate_proj_weight"]
    seam.reset_dispatch_counters()
    got = seam.blockwise_fp8_mm(row, weight, grid)
    expected = seam.blockwise_fp8_mm_torch_oracle(row, weight, grid)
    assert got.shape == (ONE_TOKEN, INTERMEDIATE)
    assert seam.dispatch_counters() == (1, 0)
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)
