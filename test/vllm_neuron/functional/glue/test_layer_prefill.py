# SPDX-License-Identifier: Apache-2.0
"""One KDA + MoE layer at the three prefill blocks the device A/B measured, against 0a08ff4.

``test_layer_glue.py`` holds the decode batches (B in {1, 4, 64}). This module holds one
opening request's prefill chunk (``glue_case.kda_prefill_carriers``) at:

* 128 rows, where ``1`` fuses mhc_pre and mhc_post. Every switch value the A/B ran
  (``0``, ``1``, ``all`` and each kernel alone) runs the layer from this tree.
* 1024 rows, the served p1 chunk, where mhc_pre and mhc_post can be fused (the two KDA
  kernels serve at most 128 rows and decline). ``0``, ``1``, ``all`` and mhc_pre run.
* 2048 rows, the chunk of the uncapped prefill line, where ``1`` fuses mhc_pre and
  mhc_post (the two KDA kernels decline). ``1`` runs against the snapshot, mhc_post
  against this tree's ``0`` (below).

The 0a08ff4 snapshot runs the same layer once per row count, with the same weights and
operands. In the simulator:

* a value that fuses at most mhc_post is 0a08ff4 bit for bit (mhc_post's bf16 combine
  is the same fp32 values, rounded once).
* every other value is within ``test_layer_glue.py``'s layer tolerances. It may
  also be bit for bit: kda_projections at 128 rows was with 4, 16 and 24 CPU threads and
  was not with 192 (torch's CPU matmul sums in an order that depends on the thread count).

At 2048 rows this tree's ``0`` is not the snapshot bit for bit: with 4 CPU threads, 2 of
the 33,554,432 output elements differ, by one bf16 step, in one row, also at ab4f37f, with
every glue site on its torch route. So the difference comes from outside the glue, and at
2048 rows mhc_post is held to this tree's ``0`` instead
(:func:`test_mhc_post_alone_is_the_zero_route_at_2048_rows`).

Each run also checks which kernels served it: the value's selection, bounded by each
kernel's own row limit. Those counters, not a nonzero difference, are the proof that a
kernel ran.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_snapshot
from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.functional.glue import test_layer_glue as layer_glue
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
from vllm_neuron.functional.mhc import hyper_connection as combine
from vllm_neuron.model.glm5_next import model_fp8

VALUES = {128: ("0", "1", "all", "mhc_pre", "kda_projections", "kda_output", "mhc_post"),
          1024: ("0", "1", "all", "mhc_pre"),
          glue_case.UNCAPPED_PREFILL_CHUNK: ("1",)}
#: The largest row count each bounded kernel serves (its ``*_admits`` shape rule);
#: mhc_pre serves any row count.
MAX_ROWS = {kda_projections: kda_projections.KDA_PROJECTIONS_MAX_TOKENS,
            kda_output: kda_output.KDA_OUTPUT_MAX_TOKENS}
#: Calls per layer step: both mHC sites, one KDA input projection, one output projection.
CALLS = {mhc_pre: 2, kda_projections: 1, kda_output: 1}
_REFERENCE: dict = {}


def _layer_output(model, rows: int) -> torch.Tensor:
    torch.manual_seed(0)
    case = glue_case.kda_layer(model)
    carriers = glue_case.kda_prefill_carriers(case, rows)
    streams = glue_case.streams_input(case.cfg, rows)
    with torch.no_grad():
        out = glue_case.layer_step(model, case, streams, carriers,
                                   torch.tensor(0, dtype=torch.int64),
                                   glue_case.quant_config(model))
    assert out.dtype == torch.bfloat16 and out.shape == streams.shape
    return out.float()


def _reference(rows: int) -> torch.Tensor:
    if rows not in _REFERENCE:
        _REFERENCE[rows] = _layer_output(load_snapshot().model_fp8, rows)
    return _REFERENCE[rows]


@pytest.mark.parametrize(("rows", "value"),
                         [(rows, value) for rows, values in VALUES.items()
                          for value in values])
def test_prefill_layer_matches_the_snapshot(rows, value, monkeypatch):
    want = _reference(rows)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)
    for module in CALLS:
        module.reset_dispatch_counters()
    got = _layer_output(model_fp8, rows)
    sel = glue.glue_selection(value)
    name = {mhc_pre: "mhc_pre", kda_projections: "kda_projections",
            kda_output: "kda_output"}
    fused = set()
    for module, calls in CALLS.items():
        served = (sel.selects(name[module], rows, "prefill")
                  and rows <= MAX_ROWS.get(module, rows))
        assert module.dispatch_counters() == ((calls, 0) if served else (0, calls)), (
            name[module], value, rows)
        if served:
            fused.add(module)
    # Of the two mHC sites, the feed-forward one also returns the FFN norm.
    assert mhc_pre.normed_dispatches() == (1 if mhc_pre in fused else 0), (value, rows)
    if not fused:
        assert torch.equal(got, want), (
            f"{value} at {rows} rows is not the snapshot bit for bit")
        return
    diff = (got - want).abs()
    peak = float(want.abs().max())
    step = 2.0 ** (torch.floor(torch.log2(torch.tensor(peak))) - 7.0)
    rel_l2 = float(diff.norm() / want.norm())
    flips = float((diff > 0).float().mean())
    reading = (f"{value} at {rows} rows: rel_l2={rel_l2:.2e} max|d|={float(diff.max()):.2e} "
               f"peak={peak:.2f} flips={flips:.4f}")
    assert rel_l2 <= layer_glue.LAYER_REL_L2, reading
    assert float(diff.max()) <= layer_glue.LAYER_MAX_STEPS * float(step), reading
    assert flips <= layer_glue.LAYER_MAX_FLIPS, reading


def test_mhc_post_alone_is_the_zero_route_at_2048_rows(monkeypatch):
    """At the uncapped line's 2048-row chunk, mhc_post alone is this tree's ``0`` bit for bit.

    The reference is this tree's ``0``, not the snapshot (see the module docstring). The
    combine is spied on, so the test also shows that ``0`` widened the operands to fp32
    and mhc_post handed them over as bf16: the two runs took different routes."""
    rows = glue_case.UNCAPPED_PREFILL_CHUNK
    real = combine.hyper_connection_combine
    outputs, dtypes = {}, {}
    for value in ("0", "mhc_post"):
        seen = set()

        def spy(x, residual, post_layer_mix, comb_res_mix, seen=seen):
            seen.add((x.dtype, residual.dtype))
            return real(x, residual, post_layer_mix, comb_res_mix)

        monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)
        monkeypatch.setattr(combine, "hyper_connection_combine", spy)
        for module in CALLS:
            module.reset_dispatch_counters()
        outputs[value] = _layer_output(model_fp8, rows)
        monkeypatch.setattr(combine, "hyper_connection_combine", real)
        for module, calls in CALLS.items():
            assert module.dispatch_counters() == (0, calls), (module.__name__, value)
        dtypes[value] = seen
    assert dtypes == {"0": {(torch.float32, torch.float32)},
                      "mhc_post": {(torch.bfloat16, torch.bfloat16)}}
    assert torch.equal(outputs["mhc_post"], outputs["0"])
