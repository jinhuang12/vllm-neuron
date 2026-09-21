# SPDX-License-Identifier: Apache-2.0
"""Pad to 128 rows and slice back at the dense kernel call site.

The pad never leaves the call, and the kernel still refuses a short token count on
its own.
"""

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

#: The count under test, and the count the oracle runs. ``1`` is the decode step
#: the seam refused; ``TILE_SIZE`` is the smallest count it ever accepted, so the
#: oracle is the shipped path on the shape it already ran.
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

    # ---- the unpadded oracle: the shipped path on the only count it ever ran.
    oracle_rows = hidden.repeat(ORACLE_TOKENS, 1)
    seam.reset_dispatch_counters()
    oracle = run(module, oracle_rows)
    if tuple(oracle.shape) != (ORACLE_TOKENS, HIDDEN):
        raise VacuousControlError(
            f"the oracle returned {tuple(oracle.shape)}, not "
            f"({ORACLE_TOKENS}, {HIDDEN}); there would be no reference row"
        )

    # ---- one token through the same path, watched.
    log = _record_the_seam(monkeypatch, sentinel)
    seam.reset_dispatch_counters()
    got = run(module, hidden)
    dispatches, fallbacks = seam.dispatch_counters()

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

    # ---- reading: no pad row survives into the result, and none is derived from
    # one -- the return is the seam's own output restricted to the caller's rows.
    raw = log["outputs"][-1]
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

    # ---- reading: the criterion.
    _worst, outside = _worst_relative_error(got[0], oracle[0])
    if outside != 0:
        raise AssertionError(
            f"{outside} elements of the padded one-token result differ from the "
            f"unpadded oracle's first row by more than atol + rtol * |want| "
            f"(rtol={RTOL}, atol={ATOL})"
        )

    if (dispatches, fallbacks) != (3, 0):
        raise AssertionError(
            f"the route predicate reads (nki_dispatch={dispatches}, "
            f"torch_fallback={fallbacks}); this call declares (3, 0)"
        )

    # ---- the control: remove the slice-back and nothing else.
    monkeypatch.setattr(model_fp8, "_unpad_rows", lambda out, tokens: out)
    unsliced = run(module, hidden)
    pad_rows_returned = int(unsliced.shape[0]) - ONE_TOKEN
    if tuple(unsliced.shape) == (ONE_TOKEN, HIDDEN):
        raise VacuousControlError(
            "removing the slice-back changed nothing the caller can see; the "
            "control cannot fail and this test would be meaningless"
        )
    if pad_rows_returned != ORACLE_TOKENS - ONE_TOKEN:
        raise VacuousControlError(
            f"the control returned {pad_rows_returned} extra rows, not "
            f"{ORACLE_TOKENS - ONE_TOKEN}; it is not the pad that survived"
        )


# --------------------------------------------------------------------------- #
# check 1 of 3 -- ``Glm5NextDenseMLP.forward``.                                 #
# --------------------------------------------------------------------------- #
def test_the_dense_mlp_runs_a_one_token_step_and_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One token through the dense MLP: padded at entry, sliced at the return."""
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
# check 3 of 3 -- the seam's refusal is untouched.                               #
# --------------------------------------------------------------------------- #
def test_the_seam_still_refuses_a_short_token_count_by_itself() -> None:
    """The pad is the caller's and the kernel's refusal stays unconditional."""
    seam = _seam_module()
    operands = _operands()
    row = operands["row"]
    weight, grid = operands["gate_proj_weight"]
    with pytest.raises(seam.BlockwiseFp8MmError) as caught:
        seam.blockwise_fp8_mm(row, weight, grid)
    message = str(caught.value)
    if f"M={int(row.shape[0])}" not in message:
        raise AssertionError(
            f"the seam refused but not on the token count: {message!r}"
        )
