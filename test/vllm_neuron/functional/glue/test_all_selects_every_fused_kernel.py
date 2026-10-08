# SPDX-License-Identifier: Apache-2.0
"""``VLLM_NEURON_GLUE_FUSED=all`` selects every fused kernel at every phase and row count.

``all`` is 821274e's selection, where each site asked one on/off switch and then its
own shape rules. So under ``all`` each site's predicate is its shape rules alone.
Checked on a grid of row counts on both sides of each kernel's row bound, in both
operand dtypes, with the phase known or not: each
predicate admits a call exactly when it has at most the kernel's ``*_MAX_TOKENS`` rows
and operands of a dtype the kernel takes, whatever phase the caller passes. Under ``0``
no predicate admits any call.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.functional.glue import test_kda_output as output_case
from test.vllm_neuron.functional.glue import test_kda_projections as projection_case
from test.vllm_neuron.functional.glue import test_mhc_pre as pre_case
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre

ROWS = (1, 2, 4, 63, 64, 65, 127, 128, 129, 256, 1024)
PHASES = (None, "prefill", "decode")
MAX_ROWS = {"mhc_pre": mhc_pre.MHC_PRE_MAX_TOKENS,
            "kda_projections": kda_projections.KDA_PROJECTIONS_MAX_TOKENS,
            "kda_output": kda_output.KDA_OUTPUT_MAX_TOKENS}
PREDICATES = {"mhc_pre": mhc_pre.mhc_pre_admits,
              "kda_projections": kda_projections.kda_projections_admits,
              "kda_output": kda_output.kda_gated_projection_admits}

sites = pre_case.sites
pair = projection_case.pair


def _questions(sites, pair):
    """``(kernel, args, rows, dtypes the kernel takes)`` for every predicate and row count.

    mhc_pre and kda_projections take bf16 (served) and fp32 activations; kda_output
    takes fp32 rows only, so its bf16 ``core`` must be refused at every row count.
    """
    live_site, _, cfg = sites
    attn, _, attn_cfg = pair
    out = []
    for rows in ROWS:
        streams = glue_case.streams_input(cfg, rows)
        for residual in (streams, streams.float()):
            out.append(("mhc_pre", (residual, live_site.fn, live_site.hc_scale,
                                    live_site.hc_base), rows, True))
        hidden = projection_case._hidden(attn_cfg, rows)
        for h in (hidden, hidden.float()):
            out.append(("kda_projections", (h, attn), rows, True))
        core, gate, _ = output_case._operands(rows)
        out.append(("kda_output", (core, gate, attn), rows, True))
        out.append(("kda_output", (core.to(torch.bfloat16), gate, attn), rows, False))
    return out


def test_the_grid_holds_both_sides_of_every_bound():
    for bound in MAX_ROWS.values():
        assert bound in ROWS and bound + 1 in ROWS


def test_all_selects_every_kernel_at_every_phase_and_row_count():
    sel = glue.glue_selection("all")
    for kernel in glue.KERNELS:
        for rows in ROWS:
            for phase in PHASES:
                assert sel.selects(kernel, rows, phase), (kernel, rows, phase)


@pytest.mark.parametrize("value", ("all", "0"))
def test_each_site_admits_by_its_shape_rules_alone(sites, pair, value, monkeypatch):
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)
    admitted = set()
    for kernel, args, rows, dtype_taken in _questions(sites, pair):
        want = value == "all" and dtype_taken and rows <= MAX_ROWS[kernel]
        for phase in PHASES:
            got = PREDICATES[kernel](*args, phase=phase)
            assert got == want, (
                f"{value}: {kernel} at {rows} rows ({args[0].dtype}, phase {phase}) "
                f"answers {got}, its shape rules say {want}")
            admitted.add((kernel, got))
    if value == "all":
        assert admitted == {(k, answer) for k in PREDICATES for answer in (True, False)}
