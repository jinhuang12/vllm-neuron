# SPDX-License-Identifier: Apache-2.0
"""Resolve ``--tip`` to a kernel set and the gate run that measured it (written new).

  - ``current``: every wave-1 kernel ("after"); compared with the latest gate run.
  - a commit with no gate record (``5938748``, a merge sha, ...): a family runs "after"
    when a gate run of its branch (``name``) has verdict MERGE and its gated candidate
    (``gate_sha``) is an ancestor of the commit (team-lead ruling). The gated candidate is
    the rebased head that the merge brings in, so this is the same test as "merge sha is
    an ancestor" for every commit at or after the merge. The rest run 5938748.
  - a commit with a gate record (a gate run whose measured ``head`` is the commit), any
    verdict: the kernels of that tree, i.e. every gated candidate that is the commit or
    an ancestor of it, whatever its verdict (team-lead ruling, round 2). The latest gate
    run is often such a tree (``gate_mhc.json``, BLOCKED, tree f083375). A REJECTed or
    BLOCKED candidate never counts for a commit without a gate record.
  - The gate run for the commit is the newest gate whose measured tree head is the
    commit, else the newest MERGE run in the tree (its "after" run measured the
    candidate that the merge brought in).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ...readers.gate import GateStep, latest_gate
from .configs import CURRENT, FAMILIES, KernelSet

#: The repository the tips live in (this worktree; objects are shared with the main tree).
REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Tip:
    name: str
    sha: Optional[str]
    kernel_set: KernelSet
    #: gated branches whose candidate is in the tree, oldest gate run first
    branches: Tuple[str, ...]
    gate: Optional[GateStep]
    gate_tip: Optional["Tip"] = None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def _rev(repo: Path, rev: str) -> Optional[str]:
    out = _git(repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
    return out.stdout.strip() if out.returncode == 0 else None


def _is_ancestor(repo: Path, ancestor: str, sha: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", ancestor, sha).returncode == 0


def resolve_tip(tip: str, gates: List[GateStep], repo: Path = REPO) -> Tip:
    if tip == "current":
        latest = latest_gate(gates)
        gate_tip = resolve_tip(latest.head, gates, repo) if latest is not None and latest.head else None
        return Tip("current", None, CURRENT, (), latest, gate_tip)
    sha = _rev(repo, tip)
    if sha is None:
        raise ValueError(f"unknown tip {tip!r}: not a commit of {repo}")
    exact = [g for g in gates if g.head == sha]
    candidates = [g for g in gates if g.name and g.gate_sha]
    if exact:  # the tree a gate measured: every gated candidate in it, whatever the verdict
        in_tree = [g for g in candidates if g.head == sha or _is_ancestor(repo, g.gate_sha, sha)]
    else:      # no gate record: merged candidates only
        in_tree = [g for g in candidates if g.verdict == "MERGE" and _is_ancestor(repo, g.gate_sha, sha)]
    after = frozenset(g.name for g in in_tree if g.name in FAMILIES)
    merged = [g for g in in_tree if g.verdict == "MERGE"]
    gate = exact[-1] if exact else (merged[-1] if merged else None)
    branches = tuple(dict.fromkeys(g.name for g in in_tree))
    return Tip(tip, sha, KernelSet(sha[:7], after), branches, gate)
