# SPDX-License-Identifier: Apache-2.0
"""One decoder layer, this tree against 0a08ff4, in the simulator at B in {1, 4, 64}.

The benchmark's composition (``glue_case``: a KDA + MoE layer and a DSA + MoE layer at
one rank's TP=64 / EP=16 shapes, both mHC sites bound) runs from each tree with the same
weights and operands, and the bf16 output streams are compared. The kernel tests bound
each fused kernel against the expression it replaces; this test bounds what the layer
does with their small differences.

Tolerance. The layer is sensitive to its inputs: in 0a08ff4 itself, moving one input
element by one bf16 step moves the DSA layer's output by rel_l2 1.2e-3 (9.8% of the
elements change) and the KDA layer's by 8.9e-4 (5.4%), because a one-step flip in a
normed or routed value carries through the attention and the MoE. The fused kernels
change some values by one step, so the layer output is held to that scale:

* ``rel_l2 <= LAYER_REL_L2`` (observed at most 1.29e-3),
* ``max|d| <= LAYER_MAX_STEPS`` bf16 steps at ``max|ref|`` (observed one step),
* at most ``LAYER_MAX_FLIPS`` of the elements changed (observed at most 9.5%).

A wrong kernel (a missing term, a swapped weight) moves the output by rel_l2 ~1. Each
case also asserts that the fused kernels served the layer, with no decline.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
from vllm_neuron.functional.mhc import hyper_connection
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 64)
LAYER_REL_L2 = 3e-3
LAYER_MAX_STEPS = 2
LAYER_MAX_FLIPS = 0.15

#: ``(nki_dispatch, declined or fallback)`` per counter family, one layer step.
_SERVED = {
    "kda": {mhc_pre: (2, 0), kda_projections: (1, 0), kda_output: (1, 0),
            hyper_connection: (2, 0)},
    "dsa": {mhc_pre: (2, 0), hyper_connection: (2, 0)},
}


def _layer_output(model, family: str, batch: int) -> torch.Tensor:
    torch.manual_seed(0)
    case = (glue_case.kda_layer if family == "kda" else glue_case.dsa_layer)(model)
    carriers = (glue_case.kda_carriers if family == "kda" else glue_case.dsa_carriers)(
        case, batch)
    streams = glue_case.streams_input(case.cfg, batch)
    with torch.no_grad():
        out = glue_case.layer_step(model, case, streams, carriers,
                                   torch.tensor(0, dtype=torch.int64),
                                   glue_case.quant_config(model))
    assert out.dtype == torch.bfloat16 and out.shape == streams.shape
    return out.float()


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("family", ("kda", "dsa"))
def test_layer_matches_0a08ff4(family, batch):
    want = _layer_output(load_0a08ff4().model_fp8, family, batch)
    for module in _SERVED[family]:
        module.reset_dispatch_counters()
    got = _layer_output(model_fp8, family, batch)
    served = {module.__name__.rsplit(".", 1)[1]: module.dispatch_counters()
              for module in _SERVED[family]}
    assert served == {module.__name__.rsplit(".", 1)[1]: count
                      for module, count in _SERVED[family].items()}
    diff = (got - want).abs()
    peak = float(want.abs().max())
    step = 2.0 ** (torch.floor(torch.log2(torch.tensor(peak))) - 7.0)
    rel_l2 = float(diff.norm() / want.norm())
    flips = float((diff > 0).float().mean())
    reading = f"rel_l2={rel_l2:.2e} max|d|={float(diff.max()):.2e} peak={peak:.2f} flips={flips:.4f}"
    assert rel_l2 <= LAYER_REL_L2, reading
    assert float(diff.max()) <= LAYER_MAX_STEPS * float(step), reading
    assert flips <= LAYER_MAX_FLIPS, reading
