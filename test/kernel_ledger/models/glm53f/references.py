# SPDX-License-Identifier: Apache-2.0
"""Reference numbers the 5938748 ledger is reconciled against (written new).

Every number is cited by an exact text snippet of its source file; ``test_references``
checks that each snippet is still in the file. Two references per bucket:

  - ``as_built``: the DECODE_BREAKDOWN.md master-table rows (engine-active ms/step,
    rank 0, 7 traced steps, bs=1, ctx about 1070) that hold the work the ledger's
    measured units cover. Engine-active time excludes the waits inside and between
    kernels: ``waits.md`` holds those.
  - ``in_model_wall``: the same work as wall time in the model (call spans, kernel
    wall), where the breakdown reports one. A microbenchmark median is a wall time, so
    this is the like-for-like reference.

``scope`` says what the ledger's measured units of the bucket are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

BREAKDOWN_DIR = Path("/home/ubuntu/glm53f-decode-breakdown-20261005")
DECODE_BREAKDOWN = BREAKDOWN_DIR / "DECODE_BREAKDOWN.md"
ATTENTION = BREAKDOWN_DIR / "attention.md"
DENSE_BD = BREAKDOWN_DIR / "dense.md"
MOE_HOST = BREAKDOWN_DIR / "moe_host.md"
WAITS = BREAKDOWN_DIR / "waits.md"
MOE_T = Path("/home/ubuntu/glm53f-wt/reports/moe-t.md")
KDA_MD = Path("/home/ubuntu/glm53f-wt/reports/kda.md")


@dataclass(frozen=True)
class Term:
    ms: float
    label: str
    file: Path
    snippet: str


@dataclass(frozen=True)
class BucketReference:
    bucket: str
    scope: str
    as_built: Tuple[Term, ...]
    in_model_wall: Tuple[Term, ...]

    @property
    def as_built_ms(self) -> float:
        return sum(t.ms for t in self.as_built)

    @property
    def in_model_wall_ms(self) -> float:
        return sum(t.ms for t in self.in_model_wall)


#: The 5938748 device step: rank 0, 7 traced steps, quiet host (full machine).
STEP_5938748 = Term(77.36, "device step, DECODE_BREAKDOWN.md", DECODE_BREAKDOWN, "one step (77.36 ms)")

_MHC_TOTAL = Term(16.61, "mHC total: hyper_connection 9.66 + sinkhorn 6.94", DECODE_BREAKDOWN,
                  "| **mHC total** | | **16.61** |")
_KDA_CONV = Term(4.83, "depthwise_conv1d x34", DECODE_BREAKDOWN, "| depthwise_conv1d | KDA x34 | 4.83 |")
_KDA_REST = Term(0.64, "KDA rest (gate_clamp, decode_state) x34", DECODE_BREAKDOWN,
                 "| KDA rest (gate_clamp, decode_state) | KDA x34 | 0.64 |")
_KDA_GLUE = Term(0.975, "KDA-layer glue (ACT, DVE): the benchmark's 'before' holds the region's glue", DECODE_BREAKDOWN,
                 "| KDA-layer other glue (ACT, DVE) | sub-block: KDA | 0.975 |")
_MOE_GLUE = Term(3.982, "MoE-layer glue (pad, map, combine): in the benchmark's 'before' graph (moe-t.md)",
                 DECODE_BREAKDOWN, "| MoE-layer glue (ACT, DVE, POOL) | sub-block: MoE | 3.982 |")
_LM_HEAD = Term(1.826, "lm_head GEMV, replicated", DECODE_BREAKDOWN, "| lm_head GEMV, tail | tail | 1.826 |")

REFERENCES: Dict[str, BucketReference] = {r.bucket: r for r in (
    BucketReference(
        "mHC", "sinkhorn x90 + combine (hyper_connection) x90",
        as_built=(_MHC_TOTAL,),
        in_model_wall=(_MHC_TOTAL,),  # no separate wall figure; the two kernels are serial chains
    ),
    BucketReference(
        "KDA", "kda_step x34: conv + gate clamp + state step + glue, request by request (kda.md)",
        as_built=(_KDA_CONV, _KDA_REST),
        in_model_wall=(
            Term(34 * 0.348, "conv call wall span 348 us x 34 (holds about 7 ms of in-span waits)", ATTENTION,
                 "34 KDA layers: one conv call per layer, median span 348 us"),
            _KDA_REST,
            _KDA_GLUE,
        ),
    ),
    BucketReference(
        "DSA/MLA", "mla_sparse x11: one whole Glm5NextMLAAttention.forward (projections, indexer, attention, o_proj)",
        as_built=(Term(10.71, "DSA sub-block x11 (box 3b): attention.md 8.86 + DSA-layer glue 1.852",
                       DECODE_BREAKDOWN, "3b DSA sub-block x11                10.71"),),
        in_model_wall=(Term(11 * 1.254, "DSA layer block wall span 1254 us x 11", ATTENTION,
                            "11 DSA layers (3, 7, ..., 43): one attention block per layer, median span 1254 us"),),
    ),
    BucketReference(
        "MoE", "moe_router x42 (RMSNorm + router + top-8) + moe_experts x42 (at the expected hits), with the "
               "5938748 XLA glue (moe-t.md)",
        as_built=(
            Term(2.60, "routed experts: moe_fused_fp8_decode", DECODE_BREAKDOWN,
                 "| Routed experts: moe_fused_fp8_decode | MoE x42 | 2.60 |"),
            Term(0.91, "routing: router matmul + top-8 + noaux_tc", DECODE_BREAKDOWN,
                 "| Routing: router matmul + top-8 + noaux_tc | MoE x42 | 0.91 |"),
            Term(0.803, "router RMSNorm (in the norms row, seam 1)", DECODE_BREAKDOWN,
                 "the router RMSNorm (0.82 in `moe_host.md`, 0.803 by `dense.md`'s method)"),
            _MOE_GLUE,
        ),
        in_model_wall=(
            Term(42 * 0.0423, "router call 42.3 us x 42 (norm, transpose, matmul, top-8, noaux)", MOE_HOST,
                 "noaux 10.3 = 42.3 us"),
            _MOE_GLUE,
            Term(3.17, "routed-expert kernel wall (in-kernel waits included)", MOE_HOST, "Kernel wall 3.17 ms/step"),
        ),
    ),
    BucketReference(
        "dense", "shared_expert x42 + dense_mlp x3 (3 blockwise_fp8_mm + torch glue each) + RMSNorm x91",
        as_built=(
            Term(1.705, "blockwise_fp8_mm (shared expert x42, dense MLP x3)", DECODE_BREAKDOWN,
                 "| blockwise_fp8_mm (shared expert x42, dense MLP x3) | MoE shared expert | 1.705 |"),
            Term(0.082, "norms other than the router RMSNorm (0.885 - 0.803)", DECODE_BREAKDOWN,
                 "<=0.082 [XX]   norms 0.885 minus 0.803 on the MoE side"),
            Term(0.129, "dense-MLP layers 0-2 + final glue", DECODE_BREAKDOWN,
                 "| dense-MLP layers 0-2 + final glue | sub-block: dense MLP, tail | 0.129 |"),
        ),
        in_model_wall=(),  # the breakdown gives no wall figure for these calls
    ),
    BucketReference(
        "lm_head", "lm_head x1 (replicated [154880, 4096] bf16 GEMV)",
        as_built=(_LM_HEAD,),
        in_model_wall=(_LM_HEAD,
                       Term(0.335, "lm_head PE tiles tagged as other kernels (seam 5: true lm_head about 2.16)",
                            DECODE_BREAKDOWN, "0.335 ms of lm_head PE tiles carry other kernels' source tags")),
    ),
    BucketReference(
        "collectives", "90 all-reduces, model 17.5 us + bytes / 100 GB/s = the fastest traced op, the floor "
                       "(the logit gather in 'current')",
        as_built=(
            Term(0.83, "AR transfer after the last rank arrives", WAITS,
                 "| Collective: transfer after last rank arrives | 0.83 |"),
            Term(2.07, "rank 0 waits for late ranks", DECODE_BREAKDOWN,
                 "| Collective: rank 0 waits for late ranks | 90 ARs | 2.07 |"),
        ),
        # the constant model is the per-op floor: 90 x 17.5 us = 1.58 ms ("6. Collective: 90 x 17.5 us
        # (fastest op on any of 64 ranks in 7 steps) = 1.58 ms.", waits.md)
        in_model_wall=(Term(5.61, "90 all-reduce op spans (median 44.2 us; 2.71 ms of them hidden under compute)",
                            WAITS, "min 17.5 us, median 44.2, p90 135.3, max 332.0; sum 5.61"),),
    ),
)}

#: The buckets acceptance compares, in report order.
RECONCILED = ("mHC", "KDA", "DSA/MLA", "MoE", "dense", "lm_head", "collectives")
TOLERANCE = 0.15
