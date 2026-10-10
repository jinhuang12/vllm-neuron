# SPDX-License-Identifier: Apache-2.0
"""Calibrated column: a PREDICTOR, not a test (written new).

    k_bucket = in-model ms of the bucket at 5938748 / ledger raw sum at 5938748 ("before")
    calibrated_bucket = k_bucket x ledger raw sum of any kernel set

The in-model number is the gate profile of the 5938748 gate run (``gate_baseline.json``
``device_step_ms.buckets_ms``, mapped by ``profile.py``). A bucket the profile does not
name (lm_head: an unnamed compiler op) falls back to the breakdown reference
(``references.py``). Buckets with no in-model number (sampler, embed/tail) keep k = 1.

k carries the scope gap (standalone wall vs in-model engine-active) of the 5938748
kernel over to its replacement. That is an assumption: a wave-1 kernel can have a
different gap. ``--tip`` shows the measured gate profile next to the calibrated value
for every gated tree, so the assumption is checked per merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ...readers.gate import GateStep
from .ledger import Ledger
from .profile import profile_buckets
from .references import RECONCILED, REFERENCES


@dataclass
class Calibration:
    gate: Optional[GateStep]
    k: Dict[str, float] = field(default_factory=dict)
    in_model_ms: Dict[str, float] = field(default_factory=dict)
    micro_before_ms: Dict[str, float] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)


def calibrate(base: Ledger, gate: Optional[GateStep]) -> Calibration:
    """k per bucket from the 5938748 ledger ``base`` and the 5938748 gate run ``gate``."""
    if base.kernel_set.after:
        raise ValueError(f"calibration needs the 5938748 ledger, got kernel set {base.kernel_set.name}")
    prof = profile_buckets(gate.buckets_ms) if gate is not None and gate.buckets_ms else None
    cal = Calibration(gate)
    buckets = base.buckets()
    for b in RECONCILED:
        micro = buckets[b].measured_ms
        if prof is not None and prof.by_bucket.get(b):
            in_model = prof.by_bucket[b]
            src = f"{Path(gate.file).name} buckets_ms ({', '.join(prof.files[b])})"
        else:
            in_model = REFERENCES[b].reference_ms
            src = f"fallback: breakdown {REFERENCES[b].reference_kind} reference"
        if micro <= 0:
            continue
        cal.k[b], cal.in_model_ms[b], cal.micro_before_ms[b], cal.source[b] = in_model / micro, in_model, micro, src
    return cal


def calibrated_buckets(led: Ledger, cal: Calibration) -> Dict[str, float]:
    return {b: t.measured_ms * cal.k.get(b, 1.0) for b, t in led.buckets().items()}


def calibrated_ms(led: Ledger, cal: Calibration) -> float:
    return sum(calibrated_buckets(led, cal).values())
