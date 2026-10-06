# SPDX-License-Identifier: Apache-2.0
"""Operating points and kernel sets of the GLM-5.3-Flash decode ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

#: Shape-record keys of the two kernel generations. "5938748" = the wave-1 baseline (the
#: microbenchmarks' "before"); "current" = every wave-1 worker's "after".
KERNEL_SETS = ("5938748", "current")

#: Wave-1 kernel families, named as their branches (``wt/<family>``, the gate's ``name``).
FAMILIES = ("mhc", "kda", "dsa", "moe-t", "dense", "host")

#: Measured kernel (``KernelName`` value) -> the family whose branch replaces it.
FAMILY_OF: Dict[str, str] = {
    "mhc_sinkhorn_tkg": "mhc",
    "mhc_combine_tkg": "mhc",
    "kda_decode_tkg": "kda",
    "dsa_mla_layer_tkg": "dsa",
    "rmsnorm_router_topk_tkg": "moe-t",
    "moe_experts_tkg": "moe-t",
    "shared_expert_tkg": "dense",
    "dense_mlp_tkg": "dense",
    "lm_head": "dense",
    "rmsnorm_tkg": "dense",
    "sampling": "host",
}

#: Kernels a family measured but does not wire: its tree keeps the 5938748 code.
NOT_WIRED: Dict[str, str] = {
    "rmsnorm_tkg": "the NKI RMSNorm is measured, not wired (dense.md part c): the 5938748 lowering stays",
}


@dataclass(frozen=True)
class KernelSet:
    """Which family runs its wave-1 kernel ("after"); the rest run 5938748 ("before")."""

    name: str
    after: FrozenSet[str]

    def is_after(self, family: str) -> bool:
        return family in self.after

    def variant(self, kernel: str) -> str:
        if kernel in NOT_WIRED or not self.is_after(FAMILY_OF[kernel]):
            return "before"
        return "after"

    def record_key(self, family: str) -> str:
        """``KERNEL_SETS`` key of a node whose shapes depend on the family's generation."""
        return "current" if self.is_after(family) else "5938748"


BASELINE = KernelSet("5938748", frozenset())
CURRENT = KernelSet("current", frozenset(FAMILIES))


@dataclass(frozen=True)
class DecodePoint:
    """One decode operating point: ``bs`` requests, ``ctx`` tokens of context each.

    ``max_model_len`` is the served limit (the gate serves 4096 at bs=1; the bs=64
    point serves 8192, ``WAVE2_QUEUE.md``).
    """

    bs: int
    ctx: int
    max_model_len: int

    @property
    def label(self) -> str:
        return f"bs={self.bs} ctx={self.ctx}"

    @property
    def decode_window_rows(self) -> int:
        """Rows of the DSA decode window with power-of-two decode context buckets.

        The smallest power of two above ``ctx`` (the context grows while decoding),
        capped at ``max_model_len``. At ctx 1024 this is the 2048 bucket the DSA
        bypass needs (``dsa.md`` gate precondition 1).
        """
        rows = 1
        while rows <= self.ctx:
            rows *= 2
        return min(rows, self.max_model_len)


BS1_CTX1K = DecodePoint(bs=1, ctx=1024, max_model_len=4096)
BS64_CTX8K = DecodePoint(bs=64, ctx=8192, max_model_len=8192)


def point_for(bs: int, ctx: int) -> DecodePoint:
    """The served ``max_model_len`` for a requested point: 4096 up to ctx 4096, else ctx."""
    return DecodePoint(bs=bs, ctx=ctx, max_model_len=max(4096, ctx))
