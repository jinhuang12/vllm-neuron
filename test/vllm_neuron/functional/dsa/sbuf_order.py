# SPDX-License-Identifier: Apache-2.0
"""Ordering check over a compiled NKI kernel: every pair of instructions that touch the
same bytes runs in a fixed order on the device.

Input: the backend's instruction dump after its ``lower_sync`` pass, the last pass that
changes addresses or synchronization (``neuronx-cc --internal-print-after=lower_sync``
writes ``<core>/sg00/bir_debug.*after-lower_sync*.json``). The dump lists, per core,
every instruction with its engine, its memory operands (exact access patterns on
physical addresses) and its semaphore waits.

The check builds a happens-before graph:

* instructions of one engine run in program order;
* an instruction that waits on a semaphore runs after the update it waits for;
* a DMA has an issue node (on its engine or queue) and a completion node, which only a
  semaphore wait leaves;
* an all-engine barrier follows every earlier instruction and precedes every later one.

and reports:

* ``unsync``: two accesses to overlapping bytes, at least one a write, neither of which
  happens before the other. On the device their order is whatever the engines' timing
  makes it. Footprints are exact: SBUF and PSUM by start partition (the memloc's
  ``base`` quadrant), partition set and byte intervals; DRAM by byte intervals.
* ``library_alias``: a GpSimd library instruction (``nc_n_gather``,
  ``nonzero_with_count``, ...) whose output overlaps one of its own inputs. The device
  runs these in pieces that re-read their inputs, so an in-place placement corrupts them.
* ``stepped_dma``: a DMA over a non-contiguous partition set (``tile[0:128:16]``). The
  compiler once left such a DMA's reads unordered against a later write, so one fails
  the check even where every pair it takes part in is ordered.
* ``early_waits``: a semaphore wait whose value the producer's update has not yet
  reached (or that no update reaches), so the edge the wait stands for is not real.

Pairs ordered only because two DMAs of one queue complete in issue order are counted
apart (``queue_order``): the compiler relies on that order everywhere.

The dump format is the backend's internal one; :data:`DUMP_SCHEMA_KEYS` names the fields
read, and :func:`check_dump` raises if one is missing, so an SDK change fails loudly.
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field

#: Bytes per element of the dtypes a dump names.
_DTYPE_BYTES = {"float32": 4, "int32": 4, "uint32": 4, "bfloat16": 2, "float16": 2,
                "int16": 2, "uint16": 2, "int8": 1, "uint8": 1, "float8e4": 1,
                "float8e5": 1, "bool": 1}
#: Library instructions the device runs in pieces that re-read their inputs.
LIBRARY_OPCODES = frozenset({"Gather", "NonzeroWithCount", "Nonzero", "LocalGather",
                             "Max8", "TopK", "MatchReplace8", "RangeSelect", "MaxIndex8"})
#: Fields of the dump the check reads.
DUMP_SCHEMA_KEYS = ("functions", "queues")
#: Largest interval list an access pattern expands to before the bounding box stands in.
_EXPAND_CAP = 100000
_ENGINE_SUFFIX = re.compile(r"-(Activation|PE|DVE|SP|Pool)\d+$")


@dataclass
class Findings:
    """What :func:`check_dump` found in one core's dump."""
    instructions: int = 0
    ordered: int = 0
    queue_order: int = 0
    unsync: list = field(default_factory=list)
    library_alias: list = field(default_factory=list)
    stepped_dma: list = field(default_factory=list)
    early_waits: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.unsync or self.library_alias or self.stepped_dma or self.early_waits)

    def summary(self) -> str:
        return (f"{self.instructions} instructions: {len(self.unsync)} unsynchronized pairs, "
                f"{len(self.library_alias)} library dst/src overlaps, "
                f"{len(self.stepped_dma)} stepped-partition DMAs, {len(self.early_waits)} "
                f"early waits ({self.ordered} pairs ordered, {self.queue_order} by same-queue "
                f"DMA order)")


def _intervals(pattern, start):
    """Exact ``[lo, hi)`` element intervals of an access pattern from element ``start``;
    ``None`` past :data:`_EXPAND_CAP` (the caller then compares bounding boxes)."""
    starts = [start]
    for step, count in pattern[:-1]:
        starts = [s + i * step for s in starts for i in range(count)]
        if len(starts) > _EXPAND_CAP:
            return None
    step, count = pattern[-1]
    if step == 1:
        return [(s, s + count) for s in starts]
    out = [(s + i * step, s + i * step + 1) for s in starts for i in range(count)]
    return out if len(out) <= _EXPAND_CAP else None


def _intervals_overlap(a, b):
    if a is None or b is None:
        return True
    a, b = sorted(a), sorted(b)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] <= b[j][0]:
            i += 1
        elif b[j][1] <= a[i][0]:
            j += 1
        else:
            return True
    return False


def _footprint(operand, memlocs):
    """The bytes one operand touches: its space, partitions (SBUF/PSUM) and intervals."""
    if operand.get("kind") != "physical_ap":
        return None
    loc = memlocs[operand["memref"]]
    size = _DTYPE_BYTES[operand["dtype"]]
    pattern, offset = operand["ap"], operand["offset"]
    if loc["type"] in ("SB", "PSUM"):
        pitch = loc["dims"][1] // size
        step0, count0 = pattern[0]
        first_partition = loc.get("base") or 0
        parts = frozenset(first_partition + (offset + i * step0) // pitch
                          for i in range(count0))
        column = offset % pitch
        inner = pattern[1:]
        reach = sum((n - 1) * s for s, n in inner)
        origin = loc["addr"] // size + column
        ivs = _intervals(inner, origin) if inner else [(origin, origin + 1)]
        space = "SB" if loc["type"] == "SB" else f"PSUM{loc['bank']}"
        return dict(space=space, parts=parts,
                    lo=loc["addr"] + column * size,
                    hi=loc["addr"] + (column + reach + 1) * size,
                    ivs=None if ivs is None else [(x * size, y * size) for x, y in ivs],
                    memref=operand["memref"])
    reach = sum((n - 1) * s for s, n in pattern)
    ivs = _intervals(pattern, loc["addr"] // size + offset)
    space = "DRAM:" + ("internal" if loc["kind"] == "Internal" else operand["memref"])
    return dict(space=space, parts=None, lo=loc["addr"] + offset * size,
                hi=loc["addr"] + (offset + reach + 1) * size,
                ivs=None if ivs is None else [(x * size, y * size) for x, y in ivs],
                memref=operand["memref"])


def _overlap(a, b):
    if a["space"] != b["space"] or a["lo"] >= b["hi"] or b["lo"] >= a["hi"]:
        return False
    if a["parts"] is not None and not (a["parts"] & b["parts"]):
        return False
    return _intervals_overlap(a["ivs"], b["ivs"])


def _overlapping_pairs(nodes):
    """``{(x, y, kind): [(a, b), ...]}``: access nodes ``x < y`` with a footprint ``a`` of
    ``x`` and ``b`` of ``y`` on overlapping bytes, at least one of them a write (``kind``
    RAW, WAW or WAR, read from ``x``'s side). A sweep over each space's byte ranges finds
    the candidates, so the cost follows the overlaps, not the square of the accesses."""
    items = sorted(((f["space"], f["lo"], f["hi"], x, write, f)
                    for x, node in enumerate(nodes)
                    for write, side in ((True, node["writes"]), (False, node["reads"]))
                    for f in side), key=lambda item: (item[0], item[1]))
    pairs = collections.defaultdict(list)
    active, space = [], None
    for item in items:
        if item[0] != space:
            active, space = [], item[0]
        active = [other for other in active if other[2] > item[1]]
        for other in active:
            if other[3] == item[3] or not (other[4] or item[4]):
                continue
            first, second = (other, item) if other[3] < item[3] else (item, other)
            if _overlap(first[5], second[5]):
                kind = "WAW" if first[4] and second[4] else "RAW" if first[4] else "WAR"
                pairs[(first[3], second[3], kind)].append((first[5], second[5]))
        active.append(item)
    return pairs


class _Graph:
    """Access nodes and the happens-before edges of one core's dump."""

    def __init__(self, dump):
        func = dump["functions"][0]
        self.memlocs = {}
        for alloc in func["allocations"]:
            for loc in alloc.get("memorylocations", []):
                self.memlocs[loc["name"]] = dict(type=loc["type"], addr=loc["addr"],
                                                 dims=loc["dims"], kind=alloc["kind"],
                                                 bank=loc.get("bank"), base=loc.get("base"))
        self.instructions = [i for b in func["blocks"] for i in b["instructions"]]
        blocks, block_info = {}, {}
        for queue in dump["queues"]:
            for blk in queue["blocks"]:
                for b in blk["instructions"]:
                    blocks[b["name"]] = [x for sub in b.get("blocks", [])
                                         for x in sub["instructions"]]
                    block_info[b["name"]] = b
        self.edges = collections.defaultdict(set)
        self.queue_edges = collections.defaultdict(set)
        self.access = []
        self.updates = []
        last_on_engine, last_on_queue, done_of = {}, {}, {}
        barrier = None
        for inst in self.instructions:
            name, engine, opcode = inst["name"], inst["engine"], inst["opcode"]
            self.updates.append((name, inst))
            if engine == "ALL":
                for e, prev in list(last_on_engine.items()):
                    self.edges[prev].add(name)
                    last_on_engine[e] = name
                if barrier is not None:
                    self.edges[barrier].add(name)
                barrier = name
            elif engine != "Unassigned":
                prev = last_on_engine.get(engine, barrier)
                if prev is not None:
                    self.edges[prev].add(name)
                last_on_engine[engine] = name
            if opcode == "DMACopy":
                done = name + "#done"
                self.edges[name].add(done)
                done_of[name] = done
                self._add(name, name, done, inst, inst.get("ins", []), inst.get("outs", []))
                self._queue(last_on_queue, inst.get("queue"), done)
            elif opcode == "DMATrigger":
                for block in inst.get("dma_blocks", []):
                    self.edges[name].add(block)
                    if block in block_info:
                        self.updates.append((block, block_info[block]))
                    descs = blocks.get(block, [])
                    self._add(block, block, block, inst,
                              [x for d in descs for x in d.get("ins", [])],
                              [x for d in descs for x in d.get("outs", [])])
                    self._queue(last_on_queue, inst.get("queue"), block)
            else:
                self._add(name, name, name, inst, inst.get("ins", []), inst.get("outs", []))
        for inst in self.instructions:
            for wait in (inst.get("sync_info") or {}).get("on_wait", []):
                self.edges[done_of.get(wait["from"], wait["from"])].add(inst["name"])
        self._reach = {}

    def _add(self, name, issue, done, inst, ins, outs):
        reads = [f for f in (_footprint(x, self.memlocs) for x in ins) if f]
        writes = [f for f in (_footprint(x, self.memlocs) for x in outs) if f]
        if reads or writes:
            self.access.append(dict(name=name, issue=issue, done=done, inst=inst,
                                    dma="DMA" in inst["opcode"], reads=reads, writes=writes))

    def _queue(self, last_on_queue, queue, node):
        if queue in last_on_queue:
            self.queue_edges[last_on_queue[queue]].add(node)
        last_on_queue[queue] = node

    def _closure(self, with_queue):
        """Every node's successors in the happens-before order, as a bit set over a node
        index: one pass in reverse topological order (the order is acyclic, or the program
        would deadlock)."""
        succ = collections.defaultdict(set)
        for edges in (self.edges, self.queue_edges) if with_queue else (self.edges,):
            for node, after in edges.items():
                succ[node] |= after
        nodes = set(succ).union(*succ.values())
        index = {node: i for i, node in enumerate(nodes)}
        incoming = dict.fromkeys(nodes, 0)
        for after in succ.values():
            for node in after:
                incoming[node] += 1
        order = [node for node in nodes if not incoming[node]]
        for node in order:
            for later in succ.get(node, ()):
                incoming[later] -= 1
                if not incoming[later]:
                    order.append(later)
        if len(order) != len(nodes):
            raise ValueError("the happens-before graph has a cycle")
        reach = {}
        for node in reversed(order):
            bits = 0
            for later in succ.get(node, ()):
                bits |= reach[later] | (1 << index[later])
            reach[node] = bits
        return index, reach

    def before(self, a, b, with_queue=False):
        """``a`` completes before ``b`` starts."""
        if with_queue not in self._reach:
            self._reach[with_queue] = self._closure(with_queue)
        index, reach = self._reach[with_queue]
        return b["issue"] in index and bool(reach.get(a["done"], 0) >> index[b["issue"]] & 1)

    def early_waits(self):
        """Waits below the producer's cumulative update of that semaphore. Each semaphore
        belongs to one engine or queue, whose updates land in program order; a
        ``GroupResetSemaphores`` (between the blocks of a device loop) zeroes its group."""
        total, after = collections.Counter(), {}
        for name, inst in self.updates:
            if inst["opcode"] == "GroupResetSemaphores":
                for sem in inst.get("sema_group", []):
                    total[sem] = 0
            for update in (inst.get("sync_info") or {}).get("on_update", []):
                total[update["id"]] += update["update_value"]
                after[(name, update["id"])] = total[update["id"]]
        out = []
        for inst in self.instructions:
            for wait in (inst.get("sync_info") or {}).get("on_wait", []):
                reached = after.get((wait["from"], wait["id"]))
                if reached is None or wait["wait_value"] < reached:
                    out.append(f"{inst['name']} waits for {wait['from']} semaphore "
                               f"{wait['id']} >= {wait['wait_value']}; the update reaches "
                               f"{reached}")
        return out


def _label(access):
    inst = access["inst"]
    debug = inst.get("debug") or {}
    return f"{access['name']} {inst['opcode']}@{inst['engine']} line {debug.get('lineno')}"


def check_dump(path: str) -> Findings:
    """The :class:`Findings` of one core's ``lower_sync`` dump."""
    with open(path) as f:
        dump = json.load(f)
    missing = [k for k in DUMP_SCHEMA_KEYS if k not in dump]
    if missing:
        raise ValueError(f"{path}: not a lower_sync instruction dump (no {missing})")
    graph = _Graph(dump)
    found = Findings(instructions=len(graph.instructions))
    nodes = graph.access
    pairs = _overlapping_pairs(nodes)
    for x, y, kind in sorted(pairs):
        first, second = nodes[x], nodes[y]
        if graph.before(first, second) or graph.before(second, first):
            found.ordered += 1
        elif graph.before(first, second, True) or graph.before(second, first, True):
            found.queue_order += 1
        elif (first["inst"]["opcode"] == second["inst"]["opcode"] == "TensorSave"
              and any(fa["memref"] == fb["memref"] for fa, fb in pairs[(x, y, kind)])
              and _ENGINE_SUFFIX.sub("", first["name"]) == _ENGINE_SUFFIX.sub("", second["name"])):
            # One loop register saved by each engine: the same value, one slot.
            found.ordered += 1
        else:
            found.unsync.append(f"{kind} {_label(first)} <-> {_label(second)}")
    for node in nodes:
        if node["inst"]["opcode"] in LIBRARY_OPCODES:
            for r in node["reads"]:
                if any(_overlap(r, w) for w in node["writes"]):
                    found.library_alias.append(_label(node))
                    break
        if node["dma"]:
            for f in node["reads"] + node["writes"]:
                parts = sorted(f["parts"] or ())
                if len(parts) > 1 and parts != list(range(parts[0], parts[-1] + 1)):
                    found.stepped_dma.append(_label(node))
                    break
    found.early_waits = graph.early_waits()
    return found
