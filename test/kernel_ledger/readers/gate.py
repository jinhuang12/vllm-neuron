# SPDX-License-Identifier: Apache-2.0
"""Reader of the TP=64 gate's ``reports/gate_*.json`` (written new for the ledger).

Two schemas exist:
  - a launch record (``gate_baseline.json``, ``gate_tip-*.json``): ``device_step_ms``
    at the top level, the tree's ``environment.head``;
  - a candidate verdict (``gate_kv.json``, ``gate_host.json``): ``name``, ``verdict``,
    ``gate_sha`` (the rebased candidate that was merged), and the candidate's launch
    under ``after``.
``device_step_ms`` is the rank-0 decode step from the device profile, mean of 7 steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .micro import REPORTS_DIR


@dataclass(frozen=True)
class GateStep:
    file: str
    label: str
    name: Optional[str]
    verdict: Optional[str]
    head: Optional[str]
    gate_sha: Optional[str]
    base_sha: Optional[str]
    device_step_ms: float
    host_gap_ms: Optional[float]
    itl_ms: Optional[float]
    mtime: float


def _mean(block) -> Optional[float]:
    if block is None:
        return None
    return block["mean"] if isinstance(block, dict) else float(block)


def _median(block) -> Optional[float]:
    if block is None:
        return None
    return block["median"] if isinstance(block, dict) else float(block)


def read_gate_step(path: Path) -> GateStep:
    path = Path(path)
    d = json.loads(path.read_text())
    launch = d.get("after") if "after" in d else d
    if "device_step_ms" not in launch:
        raise ValueError(f"{path}: no device_step_ms")
    env = launch.get("environment") or {}
    return GateStep(
        file=str(path),
        label=launch.get("label") or d.get("label") or path.stem,
        name=d.get("name"),
        verdict=d.get("verdict"),
        head=env.get("head") or d.get("tip_sha"),
        gate_sha=d.get("gate_sha"),
        base_sha=d.get("base_sha"),
        device_step_ms=_mean(launch["device_step_ms"]),
        host_gap_ms=launch.get("host_gap_ms"),
        itl_ms=_median(launch.get("itl_ms")),
        mtime=path.stat().st_mtime,
    )


def read_gates(reports_dir: Path = REPORTS_DIR) -> List[GateStep]:
    """Every readable ``gate_*.json``, oldest first."""
    out = []
    for path in sorted(Path(reports_dir).glob("gate_*.json")):
        try:
            out.append(read_gate_step(path))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda g: g.mtime)


def latest_gate(gates: List[GateStep]) -> Optional[GateStep]:
    """The most recently written gate record (the integration tip's latest measurement)."""
    return gates[-1] if gates else None
