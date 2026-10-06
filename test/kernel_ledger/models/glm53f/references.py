# SPDX-License-Identifier: Apache-2.0
"""Reference numbers the 5938748 ledger is reconciled against (written new).

Every number is cited by an exact text snippet of its source file; ``test_references``
checks that each snippet is still in the file.

Per bucket (team-lead ruling on acceptance 2):
  - ``reference``: the same-scope in-model time of the work the ledger's measured units
    cover. It is a wall time (call span, kernel wall) where the breakdown gives one.
    PASS when the ledger's raw benchmark sum is within ``TOLERANCE`` of it.
  - ``engine_active``: the DECODE_BREAKDOWN.md master-table rows for the same work
    (engine-active time; the waits inside and between kernels are a separate bucket
    there). Shown for the scope gap; not a pass criterion.
  - ``expect`` / ``cause``: MoE, dense and collectives are expected to FAIL; ``cause``
    says why (scope), and the ledger prints it on a FAIL.

The scope difference: a benchmark median is the wall time of a call in its own graph,
with its in-span DMA waits and any standalone glue; the master table is engine-active.
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
DENSE_MICRO = Path("/home/ubuntu/glm53f-wt/reports/dense_micro.json")


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
    reference: Tuple[Term, ...]
    reference_kind: str
    engine_active: Tuple[Term, ...]
    expect: str = "PASS"
    cause: str = ""
    note: str = ""
    #: extra cited terms that back ``cause`` / ``note``
    evidence: Tuple[Term, ...] = ()

    @property
    def reference_ms(self) -> float:
        return sum(t.ms for t in self.reference)

    @property
    def engine_active_ms(self) -> float:
        return sum(t.ms for t in self.engine_active)


#: The 5938748 device step: rank 0, 7 traced steps, full host (the whole machine).
STEP_5938748 = Term(77.36, "breakdown, full host", DECODE_BREAKDOWN, "one step (77.36 ms)")
#: Label of the gate baseline step (gate_baseline.json: the server on NUMA node 1 only).
GATE_BASELINE_LABEL = "gate baseline, CPU split"

_MHC_TOTAL = Term(16.61, "mHC total: hyper_connection 9.66 + sinkhorn 6.94", DECODE_BREAKDOWN,
                  "| **mHC total** | | **16.61** |")
_KDA_CONV = Term(4.83, "depthwise_conv1d x34", DECODE_BREAKDOWN, "| depthwise_conv1d | KDA x34 | 4.83 |")
_KDA_REST = Term(0.64, "KDA rest (gate_clamp, decode_state) x34", DECODE_BREAKDOWN,
                 "| KDA rest (gate_clamp, decode_state) | KDA x34 | 0.64 |")
_KDA_GLUE = Term(0.975, "KDA-layer glue (ACT, DVE)", DECODE_BREAKDOWN,
                 "| KDA-layer other glue (ACT, DVE) | sub-block: KDA | 0.975 |")
_MOE_GLUE = Term(3.982, "MoE-layer glue (pad, map, combine)", DECODE_BREAKDOWN,
                 "| MoE-layer glue (ACT, DVE, POOL) | sub-block: MoE | 3.982 |")
_BLOCKWISE = Term(1.705, "blockwise_fp8_mm (shared expert x42, dense MLP x3)", DECODE_BREAKDOWN,
                  "| blockwise_fp8_mm (shared expert x42, dense MLP x3) | MoE shared expert | 1.705 |")
_NORMS = Term(0.082, "norms other than the router RMSNorm (0.885 - 0.803)", DECODE_BREAKDOWN,
              "<=0.082 [XX]   norms 0.885 minus 0.803 on the MoE side")
_DENSE_GLUE = Term(0.129, "dense-MLP layers 0-2 + final glue", DECODE_BREAKDOWN,
                   "| dense-MLP layers 0-2 + final glue | sub-block: dense MLP, tail | 0.129 |")
_LM_HEAD = Term(1.826, "lm_head GEMV, replicated", DECODE_BREAKDOWN, "| lm_head GEMV, tail | tail | 1.826 |")
_AR_TRANSFER = Term(0.83, "AR transfer after the last rank arrives", WAITS,
                    "| Collective: transfer after last rank arrives | 0.83 |")
_AR_LATE = Term(2.07, "rank 0 waits for late ranks", DECODE_BREAKDOWN,
                "| Collective: rank 0 waits for late ranks | 90 ARs | 2.07 |")

REFERENCES: Dict[str, BucketReference] = {r.bucket: r for r in (
    BucketReference(
        "mHC", "sinkhorn x90 + combine (hyper_connection) x90",
        reference=(_MHC_TOTAL,),
        reference_kind="in-model (the breakdown gives no separate wall figure)",
        engine_active=(_MHC_TOTAL,),
    ),
    BucketReference(
        "KDA", "kda_step x34: conv + gate clamp + state step, request by request (kda.md)",
        reference=(
            Term(34 * 0.348, "conv call wall span 348 us x 34 (holds about 7 ms of in-span waits)", ATTENTION,
                 "34 KDA layers: one conv call per layer, median span 348 us"),
            _KDA_REST,
        ),
        reference_kind="in-model wall",
        engine_active=(_KDA_CONV, _KDA_REST),
        note="KDA-layer glue 0.975 ms (ACT, DVE; dense.md #4) is not in the reference: the breakdown does not "
             "split it between this region and the KDA projections. With it, the reference is 13.45 ms.",
        evidence=(_KDA_GLUE,),
    ),
    BucketReference(
        "DSA/MLA", "mla_sparse x11: one whole Glm5NextMLAAttention.forward (projections, indexer, attention, o_proj)",
        reference=(Term(11 * 1.254, "DSA layer block wall span 1254 us x 11", ATTENTION,
                        "11 DSA layers (3, 7, ..., 43): one attention block per layer, median span 1254 us"),),
        reference_kind="in-model wall",
        engine_active=(Term(10.71, "DSA sub-block x11 (box 3b): attention.md 8.86 + DSA-layer glue 1.852",
                            DECODE_BREAKDOWN, "3b DSA sub-block x11                10.71"),),
    ),
    BucketReference(
        "MoE", "moe_router x42 (RMSNorm + router + top-8) + moe_experts x42 (at the expected hits), with the "
               "5938748 XLA glue",
        reference=(
            Term(42 * 0.0423, "router call 42.3 us x 42 (norm, transpose, matmul, top-8, noaux)", MOE_HOST,
                 "noaux 10.3 = 42.3 us"),
            _MOE_GLUE,
            Term(3.17, "routed-expert kernel wall (in-kernel waits included)", MOE_HOST, "Kernel wall 3.17 ms/step"),
        ),
        reference_kind="in-model wall",
        engine_active=(
            Term(2.60, "routed experts: moe_fused_fp8_decode", DECODE_BREAKDOWN,
                 "| Routed experts: moe_fused_fp8_decode | MoE x42 | 2.60 |"),
            Term(0.91, "routing: router matmul + top-8 + noaux_tc", DECODE_BREAKDOWN,
                 "| Routing: router matmul + top-8 + noaux_tc | MoE x42 | 0.91 |"),
            Term(0.803, "router RMSNorm (in the norms row, seam 1)", DECODE_BREAKDOWN,
                 "the router RMSNorm (0.82 in `moe_host.md`, 0.803 by `dense.md`'s method)"),
            _MOE_GLUE,
        ),
        expect="FAIL",
        cause="scope: the benchmark's 'before' graph runs the 5938748 router + expert path standalone, with its XLA "
              "glue (mapping, pad, token-gather combine) and per-call DMA waits; in the served model the scheduler "
              "overlaps part of that glue with other work (moe-t.md)",
        evidence=(Term(0.0, "moe-t.md on the before graph", MOE_T,
                       'The "before" graph includes the 5938748 XLA glue (mapping, pad,'),),
    ),
    BucketReference(
        "dense", "shared_expert x42 + dense_mlp x3 (3 blockwise_fp8_mm + torch glue each) + RMSNorm x91",
        reference=(_BLOCKWISE, _NORMS, _DENSE_GLUE),
        reference_kind="engine-active (the breakdown gives no wall figure for these calls)",
        engine_active=(_BLOCKWISE, _NORMS, _DENSE_GLUE),
        expect="FAIL",
        cause="scope: the benchmark runs each MLP as 3 blockwise_fp8_mm calls + torch clamp/silu/mul/cast in its own "
              "graph, 113 us per shared-expert site; in the model one blockwise call is 37.3 us per MoE layer "
              "engine-active (breakdown dense.md) and the torch glue is in the MoE-layer glue; a 1-row norm is "
              "5.3 us standalone, while all non-router norms together are <= 0.082 ms in the model",
        evidence=(Term(0.0, "breakdown dense.md: blockwise per MoE layer", DENSE_BD, "37.3 µs per MoE layer"),
                  Term(0.0, "dense_micro.json: before route", DENSE_MICRO,
                       "5938748: 3 x blockwise_fp8_mm small-M kernel + torch clamp/silu/mul/bf16 cast")),
    ),
    BucketReference(
        "lm_head", "lm_head x1 (replicated [154880, 4096] bf16 GEMV)",
        reference=(_LM_HEAD,
                   Term(0.335, "lm_head PE tiles tagged as other kernels (seam 5: true lm_head about 2.16)",
                        DECODE_BREAKDOWN, "0.335 ms of lm_head PE tiles carry other kernels' source tags")),
        reference_kind="in-model (seam 5 corrected)",
        engine_active=(_LM_HEAD,),
    ),
    BucketReference(
        "collectives", "90 all-reduces, model 17.5 us + bytes / 100 GB/s (the logit gather in 'current')",
        reference=(_AR_TRANSFER, _AR_LATE),
        reference_kind="in-model (all-reduce time on rank 0's critical path)",
        engine_active=(_AR_TRANSFER, _AR_LATE),
        expect="FAIL",
        cause="scope: the model charges 17.5 us per all-reduce, the fastest of 630 traced (90 x 17.5 us = 1.58 ms, "
              "the floor); rank 0 also waits 2.07 ms for late ranks, partly made by the device profiler (waits.md)",
        evidence=(Term(1.58, "90 x 17.5 us floor", WAITS,
                       "6. Collective: 90 x 17.5 us (fastest op on any of 64 ranks in 7 steps) = 1.58 ms."),),
    ),
)}

#: The buckets acceptance compares, in report order.
RECONCILED = ("mHC", "KDA", "DSA/MLA", "MoE", "dense", "lm_head", "collectives")
TOLERANCE = 0.15
