# SPDX-License-Identifier: Apache-2.0
"""The bf16 all-reduce wire on the tiny worlds, against bounds derived from bf16 arithmetic.

At world size 1 the tiny worlds have no coordinator, so ``collective_policy`` never
reduces there and the wire dtype changes nothing. These tests inject
``allreduce_emulation.EmulatedTPGroup`` as a 64-rank coordinator (recursive
halving/doubling over synthetic partials with a spread of ``1/sqrt(64)`` of the row RMS)
and compare the ``bf16`` wire with the as-built ``fp32`` wire, teacher-forced on the same
tokens. Every bound is derived, not measured:

* per site, the reduction's error lies in ``bf16_rdh_rel_rms_bracket``, the rounding
  model of a bf16 RDH sum;
* at the logits, a move is counted in bf16 ulps of the row's largest logit: the as-built
  64-rank order already moves a logit by up to one ulp (an fp32 sum that lands on the
  other side of a bf16 rounding boundary), and the bf16 wire may add one more, and half
  an ulp on average;
* a top-1 or a top-8 routing change may occur only where the margin is within twice
  the move that could cause it.

Each test also has a floor, so a policy that silently stays fp32 fails instead of passing.
"""

from __future__ import annotations

import pytest

from test.vllm_neuron.model.glm5_next.tiny import allreduce_emulation as em
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.collective_policy import RowParallelSite

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The study's operating point (``allreduce_emulation.study`` defaults).
BATCH, STEPS, SEED = 4, 8, 20261007

#: Largest and mean logit move the bf16 wire may add, in bf16 ulps of the row's maximum.
LOGIT_MAX_ULPS = 2.0
LOGIT_MEAN_ULPS = 0.5
#: The wire must move the logits clearly more than the as-built order does by itself.
POWER_OVER_FP32_ORDER = 3.0
#: fp32 has 16 more mantissa bits than bf16; 2**10 leaves 2**6 for the synthetic
#: partials' own fp32 construction error.
FP32_OVER_BF16 = 2.0**-10
#: Design margins of the real-partials check, each relative to its predicted value: 64
#: heads of random weights have a spread near ``1/sqrt(64)``; the as-built path rounds
#: once, so its error is near ``BF16_ROUNDING_REL_RMS``; the synthetic partials at the
#: measured spread reproduce the real ones.
KAPPA_MARGIN = 0.2
ONE_ROUNDING_MARGIN = 0.05
EMULATION_MARGIN = 0.10


def _mean(values: list[float]) -> float:
    assert values, "no call was emulated"
    return sum(values) / len(values)


def test_the_bf16_wire_moves_the_dsa_logits_within_the_derived_bounds(monkeypatch):
    sites: list[RowParallelSite] = []
    reduce = model_fp8.reduce_row_parallel

    def recording(partial, *, site, group):
        sites.append(RowParallelSite(site))
        return reduce(partial, site=site, group=group)

    monkeypatch.setattr(model_fp8, "reduce_row_parallel", recording)
    base = em.dsa_arm(BATCH, STEPS)
    fp32_group = em.EmulatedTPGroup("fp32", seed=SEED)
    fp32 = em.dsa_arm(BATCH, STEPS, group=fp32_group, wire="fp32", fed=base["ids"])
    sites.clear()
    bf16_group = em.EmulatedTPGroup("bf16", seed=SEED)
    bf16 = em.dsa_arm(BATCH, STEPS, group=bf16_group, wire="bf16", fed=base["ids"])

    # The world prefills each request on its own, then decodes the batch STEPS times.
    # Every forward reduces at each DSA layer's MLA projection, then its FFN half.
    forwards = BATCH + STEPS
    per_forward = [RowParallelSite.MLA_O_PROJ, RowParallelSite.FFN] * tiny.STACK_LAYERS
    assert sites == per_forward * forwards
    assert fp32_group.calls == bf16_group.calls == em.sites_per_forward() * forwards
    # Under fp32 only a site that hands bf16 (the CPU routed-expert sum) passes through.
    handed_bf16 = sum(1 for d in fp32_group.dtypes if d is em.torch.bfloat16)
    assert fp32_group.passed_through == handed_bf16
    assert set(bf16_group.dtypes) == {em.torch.bfloat16}

    # Per site: the error of the 64-rank bf16 sum, against the sum of its partials.
    low, high = em.bf16_rdh_rel_rms_bracket(em.RANKS, em.KAPPA_INDEPENDENT)
    per_site = _mean(bf16_group.injected_rel_rms)
    assert low <= per_site <= high, (low, per_site, high)

    dev = em.logit_deviation(bf16, fp32)
    assert dev["max_row_ulps"] <= LOGIT_MAX_ULPS, dev
    assert dev["mean_row_ulps"] <= LOGIT_MEAN_ULPS, dev
    assert dev["top1_flips"] <= dev["top1_rows_at_risk"], dev
    assert dev["route_tokens_flipped"] <= dev["route_tokens_at_risk"], dev

    floor = em.logit_deviation(fp32, base)["mean_row_ulps"]
    moved = em.logit_deviation(bf16, base)["mean_row_ulps"]
    assert moved >= POWER_OVER_FP32_ORDER * floor, (moved, floor)


def test_the_bf16_wire_moves_the_kda_output_by_the_rounding_model():
    """The KDA world's carrier is fp32 and its two modules read the same rows, so each
    output row carries one site's error as is, measured against the exact sum."""
    base = em.kda_arm(BATCH, STEPS)
    fp32 = em.kda_arm(
        BATCH, STEPS, group=em.EmulatedTPGroup("fp32", seed=SEED), wire="fp32"
    )
    bf16_group = em.EmulatedTPGroup("bf16", seed=SEED)
    bf16 = em.kda_arm(BATCH, STEPS, group=bf16_group, wire="bf16")

    assert set(bf16_group.dtypes) == {em.torch.bfloat16}
    low, high = em.bf16_rdh_rel_rms_bracket(em.RANKS, em.KAPPA_INDEPENDENT)
    moved = em.output_deviation(bf16, base)["rel_rms"]
    assert low <= moved <= high, (low, moved, high)
    assert em.output_deviation(fp32, base)["rel_rms"] <= FP32_OVER_BF16 * low


def test_the_rounding_model_holds_on_real_tp64_partials():
    """One KDA ``o_proj`` at full geometry, cut into its 64 real per-rank partials."""
    real = em.kda_real_partials()
    # 64 heads of random weights are 64 independent contributions of similar size.
    kappa = real["kappa_measured"]
    assert abs(kappa - em.KAPPA_INDEPENDENT) <= KAPPA_MARGIN * em.KAPPA_INDEPENDENT
    # The as-built path rounds the fp32 sum to bf16 once.
    rounding = real["as_built_rounding_rel_rms"]
    predicted = em.BF16_ROUNDING_REL_RMS
    assert abs(rounding - predicted) <= ONE_ROUNDING_MARGIN * predicted
    low, high = em.bf16_rdh_rel_rms_bracket(em.RANKS, kappa)
    moved = real["bf16_wire_real_partials_rel_rms_vs_exact"]
    assert low <= moved <= high, (low, moved, high)
    # The synthetic partials at the measured spread reproduce the real ones.
    measured = real["bf16_wire_real_partials_ulps"]["rel_rms"]
    emulated = real["bf16_wire_emulated_ulps"]["rel_rms"]
    assert abs(emulated - measured) <= EMULATION_MARGIN * measured, (measured, emulated)
