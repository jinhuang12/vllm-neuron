# SPDX-License-Identifier: Apache-2.0
"""One MoE block, this tree against 8aa22fa, in the simulator at T in {1, 64, 1024}.

The block is ``moe_prefill_case.block_step``: attention-site combine, feed-forward site
collapse, the experts' norm, the router, EP rank 0's routed experts at TP 4, the shared
expert, the feed-forward combine; checkpoint layer 3's mHC leaves, gains, router and
correction bias. Both trees get the same weights and operands, and every intermediate
the block collects is compared, then the ``[T, S, H]`` bf16 output streams.

Routes. At 1 and 64 rows both trees take the decode router (``router_decode``), so the
whole block must be bitwise equal. At 1024 rows this tree takes the prefill router
(``router_prefill``: the router's RMSNorm in XLA, then the router GEMM and noaux_tc in
one launch) where 8aa22fa takes the fused RMSNorm + router kernel.

Declared tolerance at 1024 rows. The router's tolerance is ``test_router_prefill``'s:
at most 1% of rows may differ at all ("moved rows"), every other row's logits, index
and weights are bitwise equal, index sets are equal on every non-tie row. The experts'
norm, the experts and the combine are unchanged code, so the block output must be
bitwise equal on every row the router left bitwise equal. On a moved row the gate
weights may move by up to the router's ``5e-3``, so that row's output streams are held
to ``rel_l2 <= MOVED_ROW_REL_L2`` (2e-3; a weight error of 5e-3 on a 2.5-scaled gate is
0.2%), measured on each moved row by itself.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.moe_prefill_8aa22fa import load as load_8aa22fa
from vllm_neuron.functional.moe import router_prefill
from vllm_neuron.functional.moe.router import (
    noaux_tc_dispatch_counters,
    reset_noaux_tc_counters,
)
from vllm_neuron.model.glm5_next import model_fp8

from . import moe_prefill_case as case_lib
from .decode_fixtures import SimulatorCounter, index_sets_equal, tie_rows

TOKENS = (1, 64, 1024)
PREFILL_KERNEL = "noaux_tc_router_prefill_kernel"
FUSED_KERNEL = "_noaux_tc_rmsnorm_router_topk_nki"
DECODE_KERNEL = "noaux_router_decode_kernel"
ROUTER_KERNELS = (PREFILL_KERNEL, FUSED_KERNEL, DECODE_KERNEL)
MOVED_ROW_FRACTION = 0.01
TIE_MARGIN = 1e-4
MOVED_ROW_REL_L2 = 2e-3

#: Index of each intermediate in the block's collector (``moe_prefill_case.block_step``).
NORMED, LOGITS, INDEX, GATHERED, AFFINITIES = range(5)


def _run(model, tokens: int):
    """``(output streams, collector, router kernels simulated, case)`` for one step."""
    case = case_lib.moe_layer(model)
    inputs = case_lib.block_inputs(case.cfg, tokens)
    collector: list[torch.Tensor] = []
    with torch.no_grad(), SimulatorCounter() as sim:
        out = case_lib.block_step(model, case, **inputs,
                                  expert_rank=torch.tensor(0, dtype=torch.int64),
                                  quant=case_lib.quant_config(model), collector=collector)
    assert out.dtype == torch.bfloat16 and out.shape == inputs["streams"].shape
    return out, collector, [k for k in sim.kernels if k in ROUTER_KERNELS], case


@pytest.fixture(scope="module")
def snapshot():
    return load_8aa22fa().model_fp8


@pytest.mark.parametrize("tokens", TOKENS)
def test_moe_block_matches_8aa22fa(snapshot, tokens):
    reset_noaux_tc_counters()
    want, want_col, want_router, want_case = _run(snapshot, tokens)
    want_counts = noaux_tc_dispatch_counters()
    reset_noaux_tc_counters()
    got, got_col, got_router, _case = _run(model_fp8, tokens)
    # One NKI launch per block in both trees, decode or prefill, no fallback: the
    # noaux_tc seam's counters (which the decode router counts into) are unchanged.
    assert noaux_tc_dispatch_counters() == want_counts == (1, 0)
    assert len(got_col) == len(want_col)

    if tokens <= router_prefill.DECODE_ROUTE_MAX_TOKENS:
        assert want_router == got_router == [DECODE_KERNEL]
        for index, (a, b) in enumerate(zip(got_col, want_col)):
            assert torch.equal(a, b), f"collected tensor {index} differs"
        assert torch.equal(got, want)
        return

    assert want_router == [FUSED_KERNEL]
    assert got_router == [PREFILL_KERNEL]
    # The experts' norm reads the same pre-norm rows through unchanged code.
    assert torch.equal(got_col[NORMED], want_col[NORMED])

    moved = ((got_col[LOGITS] != want_col[LOGITS]).any(-1)
             | (got_col[AFFINITIES] != want_col[AFFINITIES]).any(-1)
             | (got_col[INDEX] != want_col[INDEX]).any(-1))
    kept = ~moved
    assert int(moved.sum()) <= max(1, int(tokens * MOVED_ROW_FRACTION)), (
        f"{int(moved.sum())} of {tokens} rows routed differently")
    bias = want_case.layer.mlp.experts.router_bias.detach()
    ties = tie_rows(want_col[LOGITS], bias, TIE_MARGIN)
    same = index_sets_equal(got_col[INDEX], want_col[INDEX])
    assert bool(same[~ties].all()), "index sets differ on a non-tie row"

    # Unchanged code downstream of the router: bitwise on the rows it left alone.
    assert torch.equal(got[kept], want[kept]), "a row the router left alone moved"
    reading = {"moved_rows": moved.nonzero().flatten().tolist(), "tie_rows": int(ties.sum())}
    for row in moved.nonzero().flatten().tolist():
        a, b = got[row].float(), want[row].float()
        rel_l2 = float((a - b).norm() / b.norm())
        reading[f"row{row}_rel_l2"] = rel_l2
        assert rel_l2 <= MOVED_ROW_REL_L2, reading
    print(f"T={tokens}: {reading}")
