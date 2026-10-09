# SPDX-License-Identifier: Apache-2.0
"""Hand-built `lower_sync` dumps for the depcheck tests.

A dump holds the fields ``depcheck.load`` reads, in the shapes walrus writes them: one
function with SBUF/PSUM/DRAM memory locations and blocks of instructions, each with its
engine, operands (``physical_ap``, or ``register_ap`` for a runtime address), semaphore
waits and updates, and, for a dynamic loop, the per-engine ``CompareAndBranch`` whose
``on_true`` names its own block (the back-edge). DMA blocks started by a ``DMATrigger``
(descriptor-generated, SW-DGE) live in the dump's ``queues``.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Bytes per element of the dtypes these dumps use.
DSZ = {"float32": 4, "bfloat16": 2, "float8e4": 1, "int32": 4}

#: Elements of a DRAM tensor a ``whole`` operand covers.
DRAM_SPAN = 512


class Dump:
    """A dump under construction: memory locations, blocks, instructions, DMA queues."""

    def __init__(self):
        self.allocations: list[dict] = []
        self.blocks: list[dict] = []
        self.memlocs: dict[str, dict] = {}
        self.queues: dict[str, list[dict]] = {}

    def tile(self, name: str, *, kind: str = "SB", addr: int = 0, partitions: int = 128,
             columns: int = 512, dtype: str = "float32", bank: int = 0) -> str:
        """A [partitions, columns] tile at byte ``addr`` of each partition (``SB`` or ``PSUM``)."""
        loc = dict(name=name, type=kind, addr=addr, dims=[partitions, columns * DSZ[dtype]],
                   bank=bank, base=0)
        self.allocations.append(dict(dtype=dtype, kind="Internal", debug=dict(tensor_name=name),
                                     memorylocations=[loc]))
        self.memlocs[name] = dict(partitions=partitions, columns=columns, dtype=dtype)
        return name

    def dram(self, name: str, elements: int = 1 << 20, dtype: str = "float32") -> str:
        """An external DRAM tensor of ``elements`` elements."""
        loc = dict(name=name, type="DRAM", addr=0, dims=[elements])
        self.allocations.append(dict(dtype=dtype, kind="ExternalInput",
                                     debug=dict(tensor_name=name), memorylocations=[loc]))
        self.memlocs[name] = dict(partitions=None, columns=elements, dtype=dtype)
        return name

    def whole(self, name: str) -> dict:
        """A ``physical_ap`` operand over the whole tile (the first ``DRAM_SPAN`` DRAM elements)."""
        m = self.memlocs[name]
        if m["partitions"] is None:
            return dict(kind="physical_ap", memref=name, dtype=m["dtype"],
                        ap=[[1, DRAM_SPAN]], offset=0)
        return dict(kind="physical_ap", memref=name, dtype=m["dtype"],
                    ap=[[m["columns"], m["partitions"]], [1, m["columns"]]], offset=0)

    def runtime(self, name: str) -> dict:
        """A ``register_ap`` operand on the tile: its address is a register the program sets."""
        m = self.memlocs[name]
        return dict(kind="register_ap", memsetref=f"{name}_set", dtype=m["dtype"],
                    ap=[[m["columns"], m["partitions"]], [1, m["columns"]]])

    def block(self, name: str) -> list[dict]:
        """Append an empty block; returns its instruction list."""
        instructions: list[dict] = []
        self.blocks.append(dict(name=name, instructions=instructions))
        return instructions

    def dma_block(self, queue: str, name: str, *, reads=(), writes=(), updates=()) -> str:
        """A descriptor-generated DMA block (SW-DGE) on ``queue``, started by a DMATrigger."""
        desc = dict(name=f"{name}-desc", engine="DMA", opcode="DMADescriptorCopy",
                    ins=list(reads), outs=list(writes))
        sync = {"on_update": [{"id": s, "update_value": v} for s, v in updates]} if updates else {}
        self.queues.setdefault(queue, []).append(dict(
            name=name, engine="DMA", opcode="DMABlock", ins=[], outs=[],
            blocks=[dict(name=f"{name}_Block0", instructions=[desc])], sync_info=sync))
        return name

    @staticmethod
    def ins(block: list[dict], name: str, engine: str, opcode: str, *, reads=(), writes=(),
            waits=(), updates=(), **extra) -> dict:
        """Append one instruction. ``waits``: (from, sem, value); ``updates``: (sem, value)."""
        sync = {}
        if waits:
            sync["on_wait"] = [{"from": f, "id": s, "wait_value": v} for f, s, v in waits]
        if updates:
            sync["on_update"] = [{"id": s, "update_value": v} for s, v in updates]
        i = dict(name=name, engine=engine, opcode=opcode, ins=list(reads), outs=list(writes),
                 sync_info=sync, debug=dict(lineno=len(block) + 1), **extra)
        block.append(i)
        return i

    @classmethod
    def loop_end(cls, block: list[dict], body: str, exit_: str, engines, *, reset=(),
                 waits=()) -> None:
        """walrus's end of a block: Pool waits, the barriers around the semaphore reset, and,
        when ``engines`` is not empty, each engine's branch back to ``body`` or out to
        ``exit_`` (a loop body names itself as ``body``)."""
        for k, (f, s, v) in enumerate(waits):
            cls.ins(block, f"{body}-end-cb-wait-{k}", "Pool", "EventSemaphore", waits=[(f, s, v)])
        cls.ins(block, f"{body}-barrier0", "ALL", "AllEngineBarrier")
        if reset:
            cls.ins(block, f"{body}-sema-reset", "ALL", "GroupResetSemaphores",
                    sema_group=list(reset))
        cls.ins(block, f"{body}-barrier1", "ALL", "AllEngineBarrier")
        for e in engines:
            cls.ins(block, f"scf.for-CompBr_inst-{e}", e, "CompareAndBranch",
                    on_true=body, on_false=exit_)

    def write(self, path: Path) -> Path:
        """Write the dump as JSON to ``path``."""
        queues = [dict(name=q, blocks=[dict(instructions=bs)]) for q, bs in self.queues.items()]
        d = dict(functions=[dict(name="sg0000", allocations=self.allocations,
                                 blocks=self.blocks)], queues=queues)
        path.write_text(json.dumps(d))
        return path
