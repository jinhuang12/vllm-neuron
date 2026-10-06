# SPDX-License-Identifier: Apache-2.0
"""Map a gate run's profile buckets onto the ledger buckets (written new).

``gate/attribute_decode.py`` splits the rank-0 device step into engine-active ms per
kernel source file plus wait, DMA-issue, idle and unnamed compiler-op buckets
(``gate_*.json`` ``device_step_ms.buckets_ms``). This is the measured in-model time of
each ledger bucket at the gated tree. The rules match by source path prefix, so a
wave-1 kernel that keeps its directory (``mhc/``, ``kda/``, ``dsa/``, ``moe/``) maps
with no change. A kernel source that no rule knows is listed in ``unmapped`` and kept
in the residual; add a rule for it.

Mapping notes:
  - ``nkilib/core/subkernels/{rmsnorm_tkg,norm_tkg_utils}.py`` is the router's RMSNorm
    stage (5938748 routes with "nkilib RMSNorm + router + noaux_tc", ``moe-t.md``;
    DECODE_BREAKDOWN seam 1), so it maps to MoE. The ledger's router benchmark holds it.
  - The lm_head GEMV, the fn GEMVs and the other glue are "compiler ops (unnamed XLA)":
    no ledger bucket gets them.
  - ``wait: collective`` is the all-reduce time on rank 0's critical path (transfer
    after the last rank arrives + the wait for late ranks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .decode import COLL, DENSE, DSA, KDA, MHC, MOE

#: (source path prefix, ledger bucket), first match wins.
KERNEL_RULES = (
    ("mhc/", MHC),
    ("kda/", KDA),
    ("nkilib/experimental/conv/depthwise_conv1d.py", KDA),
    ("attention/mla_", DSA),
    ("dsa/", DSA),
    ("vendored_kernels/rotational_topk/", DSA),
    ("moe/", MOE),
    ("nkilib/core/router_topk/", MOE),
    ("nkilib/core/subkernels/rmsnorm_tkg.py", MOE),
    ("nkilib/core/subkernels/norm_tkg_utils.py", MOE),
    ("blockwise_fp8_mm.py", DENSE),
    ("wait: collective", COLL),
)

#: Buckets that are residual by definition (glue, waits, issue cost, idle, copies).
RESIDUAL_RULES = (
    "compiler ops (unnamed XLA)",
    "wait: ",
    "dma issue",
    "idle (nothing traced)",
    "nkilib/core/utils/interleave_copy.py",
    "neuronxcc/private_nkl/transpose.py",
)


@dataclass
class ProfileBuckets:
    by_bucket: Dict[str, float] = field(default_factory=dict)
    files: Dict[str, List[str]] = field(default_factory=dict)
    residual_parts: Dict[str, float] = field(default_factory=dict)
    unmapped: List[str] = field(default_factory=list)

    @property
    def residual_ms(self) -> float:
        return sum(self.residual_parts.values())


def profile_buckets(buckets_ms: Dict[str, float]) -> ProfileBuckets:
    out = ProfileBuckets()
    for name, ms in buckets_ms.items():
        bucket = next((b for prefix, b in KERNEL_RULES if name.startswith(prefix)), None)
        if bucket is not None:
            out.by_bucket[bucket] = out.by_bucket.get(bucket, 0.0) + ms
            out.files.setdefault(bucket, []).append(name)
            continue
        out.residual_parts[name] = ms
        if not any(name.startswith(prefix) for prefix in RESIDUAL_RULES):
            out.unmapped.append(name)
    return out
