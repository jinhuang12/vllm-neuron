# SPDX-License-Identifier: Apache-2.0
"""Kernel-to-model ledger of one GLM-5.3-Flash decode step (written new for the ledger).

For every node of the decode graph: roofline (per call), measured time (per call, from
the microbenchmark JSONs, at this node's shapes and kernel set), and how many times a
step runs it. Rows by kind:

  - ``measured``: a benchmark exists at this shape; its median is the node's time.
  - ``missing``: the node has a kernel but no benchmark at this shape (e.g. bs=64 DSA).
  - ``fused``: a DSA member timed inside ``mla_sparse`` (the layer benchmark); roofline only.
  - ``unmeasured``: a compiler-lowered op with no benchmark; roofline only. Its time is
    in the residual ("compiler glue").
  - ``collective``: the constant collective model (17.5 us + bytes / 100 GB/s).
  - ``inactive``: not in this kernel set's graph (the logit gather before wt/dense).

"sum of measured kernels + collectives" = device-located ``measured`` rows + collectives.
A host-located measurement (the 5938748 host sampler) is reported as host time: it is
outside the device step. Residual = measured device step - that sum: compiler glue,
waits between kernels, launch skew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ...engine.collectives import ConstantCollectiveModel
from ...engine.decode_nodes import DSAIndexerNode, ExpertDecodeNode, SparseMLANode
from ...engine.node import CollectiveNode, KernelNode
from ...readers.emf import KernelResult
from ...readers.micro import config_name
from .arch import GLM53F, GLM53F_SHARDING, Glm53fArch, Glm53fSharding
from .configs import FAMILY_OF, NOT_WIRED, DecodePoint, KernelSet
from .decode import BUCKETS, COLL, build_glm53f_decode_graph

#: How far (in hits) the expected distinct-expert count may sit from a single measured
#: scenario and still use it (the benchmark fixes an integer hit count).
NEAREST_HIT_TOLERANCE = 0.5


def interpolate_points(points: Sequence[Tuple[float, float]], x: float,
                       tol: float = NEAREST_HIT_TOLERANCE) -> Optional[float]:
    """Linear interpolation through sorted ``(x, y)`` points; outside them, the end
    point if ``x`` is within ``tol`` of it, else None."""
    pts = sorted(points)
    if not pts:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    nearest = min(pts, key=lambda p: abs(p[0] - x))
    return nearest[1] if abs(nearest[0] - x) <= tol else None


@dataclass
class LedgerRow:
    node: str
    bucket: str
    kind: str
    kernel: Optional[str]
    layer_count: int
    flops: int
    bytes: int
    roofline_us: float
    measured_us: Optional[float] = None
    variant: Optional[str] = None
    location: str = "device"
    source: str = ""
    note: str = ""

    @property
    def roofline_ms(self) -> float:
        return 0.0 if self.kind == "inactive" else self.roofline_us * self.layer_count / 1e3

    @property
    def measured_ms(self) -> Optional[float]:
        if self.measured_us is None:
            return None
        return self.measured_us * self.layer_count / 1e3

    @property
    def counts_on_device(self) -> bool:
        return self.kind in ("measured", "collective") and self.location == "device"


@dataclass
class BucketTotal:
    bucket: str
    measured_ms: float = 0.0
    roofline_ms: float = 0.0
    unmeasured_roofline_ms: float = 0.0
    host_ms: float = 0.0
    measured_units: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class Ledger:
    point: DecodePoint
    kernel_set: KernelSet
    rows: List[LedgerRow]

    def row(self, name: str) -> LedgerRow:
        for r in self.rows:
            if r.node == name:
                return r
        raise KeyError(name)

    def buckets(self) -> Dict[str, BucketTotal]:
        out = {b: BucketTotal(b) for b in BUCKETS}
        for r in self.rows:
            t = out.setdefault(r.bucket, BucketTotal(r.bucket))
            t.roofline_ms += r.roofline_ms
            if r.kind in ("unmeasured", "fused"):
                t.unmeasured_roofline_ms += r.roofline_ms
            elif r.kind == "missing":
                t.missing.append(r.node)
            elif r.counts_on_device:
                t.measured_ms += r.measured_ms
                t.measured_units.append(r.node)
            elif r.kind == "measured":
                t.host_ms += r.measured_ms
        return out

    @property
    def kernels_ms(self) -> float:
        return sum(r.measured_ms for r in self.rows if r.counts_on_device and r.kind == "measured")

    @property
    def collectives_ms(self) -> float:
        return sum(r.measured_ms for r in self.rows if r.kind == "collective")

    @property
    def measured_ms(self) -> float:
        """Sum of measured kernels + collectives (device)."""
        return self.kernels_ms + self.collectives_ms

    @property
    def host_ms(self) -> float:
        return sum(r.measured_ms for r in self.rows if r.kind == "measured" and r.location != "device")

    @property
    def roofline_ms(self) -> float:
        return sum(r.roofline_ms for r in self.rows)

    @property
    def unmeasured_roofline_ms(self) -> float:
        return sum(r.roofline_ms for r in self.rows if r.kind in ("unmeasured", "fused"))

    @property
    def missing(self) -> List[str]:
        return [r.node for r in self.rows if r.kind == "missing"]

    @property
    def complete(self) -> bool:
        return not self.missing

    def residual_ms(self, step_ms: float) -> float:
        return step_ms - self.measured_ms


def _measure(node: KernelNode, ks: KernelSet, results: Dict[Tuple[str, str], KernelResult]):
    """(us per call, KernelResult, variant, note) or (None, None, variant, note)."""
    kernel = node.kernel_name.value
    family = FAMILY_OF[kernel]
    variant = ks.variant(kernel)
    notes = []
    if kernel in NOT_WIRED and ks.is_after(family):
        notes.append(NOT_WIRED[kernel])
    record = node.shape_record(ks.record_key(family))
    hit = results.get((config_name(kernel, record), variant))
    if isinstance(node, ExpertDecodeNode):
        e = node.expected_distinct_local_experts
        notes.insert(0, f"E[distinct local]={e:.2f} of {node.local_experts}")
        if hit is not None:
            us = interpolate_points(hit.points or ((0, hit.latency_us),), e)
            return us, (hit if us is not None else None), variant, "; ".join(notes)
    if isinstance(node, SparseMLANode):
        notes.insert(0, f"window {record['window_rows']} rows")
    if hit is None:
        return None, None, variant, "; ".join(notes)
    return hit.latency_us, hit, variant, "; ".join(notes)


def build_ledger(point: DecodePoint, kernel_set: KernelSet, results: Dict[Tuple[str, str], KernelResult],
                 collective_model: ConstantCollectiveModel = ConstantCollectiveModel(),
                 shard: Glm53fSharding = GLM53F_SHARDING, arch: Glm53fArch = GLM53F) -> Ledger:
    g = build_glm53f_decode_graph(point, shard, arch)
    rows: List[LedgerRow] = []
    for node in g.resolve_order():
        if isinstance(node, CollectiveNode):
            needs = getattr(node, "requires_after", None)
            active = needs is None or kernel_set.is_after(needs)
            nbytes = node.message_size_bytes
            us = collective_model.lookup_us(node.collective_type.value, f"g{node.group_size}", nbytes)
            rows.append(LedgerRow(
                node.name, COLL, "collective" if active else "inactive", node.collective_type.value,
                node.layer_count, 0, nbytes, us, us if active else None,
                source=f"model {collective_model.latency_us} us + bytes / {collective_model.bandwidth_bytes_per_s / 1e9:g} GB/s",
                note="" if active else f"only with wt/{needs}",
            ))
            continue
        if not isinstance(node, KernelNode):
            continue  # reshape / merge / cast: zero cost
        base = dict(node=node.name, bucket=node.layer_type, layer_count=node.layer_count,
                    flops=node.compute_flops(), bytes=node._total_memory_bytes(),
                    roofline_us=node.compute_roofline())
        if node.fused_in is not None:
            note = f"{node.regime}, {node.selected_tokens} tokens attended" if isinstance(node, DSAIndexerNode) else ""
            rows.append(LedgerRow(kind="fused", kernel=None, source=f"in {node.fused_in.name}", note=note, **base))
        elif node.kernel_name is None:
            rows.append(LedgerRow(kind="unmeasured", kernel=None, source="compiler op, no benchmark", **base))
        else:
            us, hit, variant, note = _measure(node, kernel_set, results)
            if hit is None:
                rows.append(LedgerRow(kind="missing", kernel=node.kernel_name.value, variant=variant,
                                      source="no benchmark at this shape", note=note, **base))
            else:
                rows.append(LedgerRow(kind="measured", kernel=node.kernel_name.value, measured_us=us,
                                      variant=variant, location=hit.location, source=" + ".join(hit.sources),
                                      note=note, **base))
    return Ledger(point, kernel_set, rows)
