"""Check every RAW/WAR/WAW pair in a walrus `--print-after=lower_sync` BIR dump.

usage: python -m test.kernel_depcheck.depcheck DUMP.json [--lines LO-HI] [--all] [--verbose]
       [--pairs]

For every pair of memory accesses with overlapping footprints (SBUF: partition set x
byte interval; DRAM: byte interval in the tensor's space, internal tensors share one
arena) the script asks whether the consumer's start happens after the producer's
completion in the happens-before graph built from (1) same-engine program order and
(2) the semaphore waits (`sync_info.on_wait[*].from`). DMA transfers are modelled as
issue node (the DMACopy instruction or DMATrigger, ordered with its engine) plus a
completion node (the DMACopy name or the DMABlock name; only semaphore waits leave it).
It also checks the arithmetic of every wait: the wait value must be >= the cumulative
update count of the producer's semaphore at the producer's completion.

The one-pass verdict (`Model.one_pass_verdict`). An UNSYNC pair is confirmed unless one of
its footprints is a runtime-address DRAM operand taken as its whole tensor (its real range is
not known), or it is two TensorSave of one register by two engines (names equal but for the
engine suffix) into one memref: the same value into one slot. An undecided operand only hides
the pairs it takes part in and adds or removes no edge, so a pair the graph leaves unordered
stays unordered in a dump that has one. A cycle or a reset that names no group leaves edges
unknown, and then no pair is confirmed. `VERDICT` is FINDINGS when a pair is confirmed or a
wait's arithmetic is not OK (with the edges known); else UNDECIDED when an operand or edge is
undecided or a pair goes through a runtime address; else CLEAN. The counts follow the word.

That graph is one pass over the program: a dynamic loop's body is checked as one
iteration. Blocks are taken in the dump's order, which is program order apart from the
loop back-edges. A loop body is a block holding a branch that names the block itself; walrus
lowers every dynamic loop that way (loop entry, body, exit, resume blocks).

Which instance a wait names (`Model.resolve`). walrus gives an instruction the same name in
every block that holds it: each loop's counter increment is
``scf.for-Inc_inst__I-10-0[-<engine>0]`` in every loop body, and a wait names its producer
by that name only. A wait binds to the instances earlier in its own block, else to those of
the nearest earlier block that holds the name; among them, to the first whose cumulative
semaphore count reaches the wait value, else the last. A wait whose name its own block holds
only after it (inside a loop body: the previous iteration's instance) and no earlier block
holds gets no edge and the status LATER-PRODUCER.

Loop iterations in the wait arithmetic. A wait inside a loop body on a producer of the same
body is checked for the first iteration (the count from the program's last reset of the
semaphore) and for every later one (the count from the body's last reset of it before the
producer, going back over the back-edge): `... (iteration 2+)` when only a later iteration
fails, ACCUMULATES when the body never resets the semaphore, so its count grows with every
iteration and a fixed wait value is reached early.

The loop-carried class (`Model.loop_carried`). A pair is in the class when, in one loop body,
access P in iteration i and access C in iteration i+1 have overlapping SBUF/PSUM footprints,
at least one writes, and C is at or before P in the body (a pair in one iteration is the
one-pass check's). For RAW (P writes, C reads) no write between them covers C's footprint
(after P in iteration i, before C in iteration i+1), so C reads P's value: the tile is carried
from one iteration to the next. Each pair is tagged by what orders P's completion before
C's start, on the body unrolled twice:
  OK                a path that avoids iteration i's back-edge sequence, the body's longest
                    suffix of instructions with no memory operand (walrus's end-of-body
                    waits, core barrier, drain, all-engine barriers, semaphore reset and
                    branches). An engine still runs its own instructions in order across the
                    branch, so an iteration's own semaphore wait on P, followed by C's engine,
                    is such a path;
  BACK-EDGE-ONLY    only paths through the back-edge sequence;
  QUEUE-ORDER-ONLY  a path exists only if DMAs on one queue complete in issue order;
  UNSYNC            no path.
A RAW pair that is not OK is a member: its only order is the loop back-edge, or nothing (the
v8d pattern, trn2-2 2026-10-08T2250Z). A WAR or WAW pair is a finding only when UNSYNC:
buffer reuse ordered by the back-edge is the normal case. `VERDICT (loop-carried)` is FINDINGS
when there is a finding, UNDECIDED when there is none but part of a loop is not modelled,
CLEAN otherwise (a dump with no loop is CLEAN). Each pair says whether the compiler's
`loop_carried_dependencies` on C names P with the pair's kind (Flow, Anti, Output).

Limits of the class:
  - distance 1 only (iterations i and i+1): a buffer rotated over several iterations is not
    paired;
  - a loop body is one block: a branch back to an earlier block (a loop of several blocks, or
    a loop around a loop) makes the class UNDECIDED; so does a body whose one-pass graph has
    a cycle, a LATER-PRODUCER wait in a body, or a body operand it cannot place;
  - DRAM is not paired across iterations;
  - an SBUF/PSUM operand at a runtime address (`register_ap`) is taken as its whole memory
    location, which assumes the address stays inside it (the pair is marked `runtime`); such
    an operand never counts as a covering write. The one-pass check still calls the dump
    UNDECIDED for it;
  - a covering write is one write that holds every byte of C's footprint; writes that cover
    it only together are not combined, so such a RAW pair is still reported;
  - covering is decided in program order (a write's issue position), not by completion;
  - no pair has C after P in the body. RAW: iteration i+1 runs P again before such a C, and
    P is taken to write the same bytes in every iteration, which a runtime-address P need
    not do. WAR/WAW: the one-pass check orders P before C in iteration i, and C's engine
    issues C of iteration i before C of iteration i+1;
  - a trailing memory-free instruction of the kernel's own (an engine's wait at the end of
    the body) is part of the back-edge sequence by the definition above;
  - `Drain` (`is_reset_sema`) is a plain instruction: its effect on semaphores is not
    modelled. `GroupResetSemaphores` zeroes its `sema_group`.
"""

# CHANGELOG
# 2026-10-08 ~05:50Z  footprint(): SBUF/PSUM partitions start at the memloc's `base` (quadrant).
#                     Every count before this is wrong; corrected counts: REPORT.md 9.5.
# 2026-10-08 ~06:45Z  engine ALL (AllEngineBarrier, GroupResetSemaphores) orders every engine;
#                     GroupResetSemaphores zeroes its group in the wait audit (REPORT.md 10, 11).
# 2026-10-08 07:50Z   the repository check is test/vllm_neuron/functional/dsa/sbuf_order.py
#                     (equal on 313 dumps, w47/sbuf_order/validate_port.json).
# 2026-10-08 10:05Z   (trn2-2 08:50Z notes, team-lead 09:25Z) three defects fixed; CHANGELOG.md:
#                     (1) a `register_ap` operand (runtime address) was dropped, so a dump
#                     with one reported fewer pairs, down to 0, as if checked. DRAM: the whole
#                     tensor is its footprint now. SBUF/PSUM: the dump is UNDECIDED, with the
#                     operand count. Any other operand kind with memory: UNDECIDED.
#                     (2) nodes were keyed by instruction name; repeated names (30 in the MoE
#                     lines) merged into one node, which makes cycles and can mark an
#                     unordered pair as ordered. Nodes are positions now (`name@k` when a name
#                     repeats); a wait on a repeated name binds to the first instance whose
#                     cumulative semaphore update reaches the wait value.
#                     (3) a cycle left in the happens-before graph makes the dump UNDECIDED.
#                     Old copy kept: depcheck_2026-10-08T0748Z.py.
# 2026-10-08 12:23Z   a GroupResetSemaphores that names no group makes the dump UNDECIDED.
# 2026-10-09          (trn2-2 v8d note 2026-10-08T2250Z) a wait binds within its own block
#                     first (Model.resolve): the 07:24Z copy bound a name repeated over loop
#                     bodies to the last body's instance, the 10:05Z copy to the first that
#                     reaches the wait value, which can be another body's. A wait on a
#                     later instruction only is LATER-PRODUCER (was: an edge pointing back).
#                     Waits in a loop body are checked for the later iterations too. New:
#                     the loop-carried class and its verdict line. Tests:
#                     test/kernel_depcheck/tests/.
# 2026-10-09          (trn2-2's v8d re-run) the one-pass VERDICT was UNDECIDED whenever an
#                     operand was undecided, above SBUF RAW pairs nothing orders. A confirmed
#                     pair is now FINDINGS (Model.one_pass_verdict); pairs only through a
#                     runtime address are UNDECIDED (were FINDINGS); one register saved by
#                     each engine into one slot is not a finding (was FINDINGS).

import bisect
import collections
import json
import re
import sys

DSZ = {"float32": 4, "int32": 4, "uint32": 4, "bfloat16": 2, "float16": 2, "int16": 2,
       "uint16": 2, "int8": 1, "uint8": 1, "float8e4": 1, "float8e5": 1, "bool": 1}


def load(path):
    d = json.load(open(path))
    f = d["functions"][0]
    memlocs = {}
    for a in f["allocations"]:
        for m in a.get("memorylocations", []):
            memlocs[m["name"]] = dict(
                type=m["type"], addr=m["addr"], dims=m["dims"], dtype=a["dtype"],
                kind=a["kind"], tensor_name=a.get("debug", {}).get("tensor_name"),
                shape=a.get("debug", {}).get("shape"), bank=m.get("bank"),
                base=m.get("base"), table_entry_id=m.get("table_entry_id"),
                tensor_id=m.get("tensor_id"), runtime_reserved=m.get("runtime_reserved"))
    ins = [i for b in f["blocks"] for i in b["instructions"]]
    blocks = {}
    for q in d["queues"]:
        for blk in q["blocks"]:
            for b in blk["instructions"]:
                descs = [x for sub in b.get("blocks", []) for x in sub["instructions"]]
                blocks[b["name"]] = dict(block=b, descs=descs, queue=q["name"],
                                         trigger=b.get("dma_trigger"))
    return d, f, memlocs, ins, blocks


def expand(ap, base, cap=100000):
    """Exact element intervals [lo,hi) covered by pattern `ap` starting at element `base`."""
    starts = [base]
    for s, n in ap[:-1]:
        starts = [b + i * s for b in starts for i in range(n)]
        if len(starts) > cap:
            return None
    s, n = ap[-1]
    if s == 1:
        return [(b, b + n) for b in starts]
    ivs = [(b + i * s, b + i * s + 1) for b in starts for i in range(n)]
    return ivs if len(ivs) <= cap else None


def ivs_overlap(a, b):
    if a is None or b is None:
        return True  # fall back to bounding-box verdict
    a = sorted(a); b = sorted(b)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] <= b[j][0]:
            i += 1
        elif b[j][1] <= a[i][0]:
            j += 1
        else:
            return True
    return False


NO_MEMORY_KINDS = {"imm_value", "register_access"}   # an immediate, a register read
ENGINE_SUFFIX = re.compile(r"-(Activation|PE|DVE|SP|Pool)\d+$")   # an engine's copy of a name


class Undecided(Exception):
    """An operand the check cannot place in memory."""


def runtime_footprint(op, memlocs):
    """A `register_ap` operand: its address is a register the program sets. DRAM: the whole
    tensor (conservative: it overlaps every access to that tensor). SBUF/PSUM: Undecided."""
    ref = op.get("memref") or op.get("memsetref") or ""
    if ref not in memlocs and ref.endswith("_set"):
        ref = ref[:-len("_set")]
    if ref not in memlocs:
        raise Undecided(f"runtime-address operand on {ref!r}: no memory location")
    m = memlocs[ref]
    if m["type"] != "DRAM":
        raise Undecided(f"runtime-address {m['type']} operand on {ref}")
    size = 1
    for n in m["dims"]:
        size *= n
    space = "DRAM:" + ("INTERNAL" if m["kind"] == "Internal" else ref)
    return dict(space=space, parts=None, lo=m["addr"], hi=m["addr"] + size, cross=False,
                ivs=None, memref=ref, tensor=m["tensor_name"], ap=op.get("ap"), off=None,
                runtime=True)


def whole_location(op, memlocs):
    """An SBUF/PSUM `register_ap` operand as its whole memory location (every partition from
    the location's `base`, every byte of its row), or None when it names no such location.
    The loop-carried class uses it; it assumes the runtime address stays inside."""
    if op.get("kind") != "register_ap":
        return None
    ref = op.get("memref") or op.get("memsetref") or ""
    if ref not in memlocs and ref.endswith("_set"):
        ref = ref[:-len("_set")]
    m = memlocs.get(ref)
    if m is None or m["type"] not in ("SB", "PSUM"):
        return None
    pbase = m.get("base") or 0
    lo, hi = m["addr"], m["addr"] + m["dims"][1]
    space = m["type"] if m["type"] == "SB" else f"PSUM.bank{m['bank']}"
    return dict(space=space, parts=frozenset(range(pbase, pbase + m["dims"][0])), lo=lo, hi=hi,
                cross=False, ivs=[(lo, hi)], memref=ref, tensor=m["tensor_name"],
                ap=op.get("ap"), off=None, runtime=True)


def covers(a, b):
    """Footprint `a` holds every byte of footprint `b` (False when `a`'s bytes are not known)."""
    if a["space"] != b["space"] or a["lo"] > b["lo"] or a["hi"] < b["hi"]:
        return False
    if b["parts"] is not None and (a["parts"] is None or not b["parts"] <= a["parts"]):
        return False
    if a.get("ivs") is None:
        return False
    merged = []
    for lo, hi in sorted(a["ivs"]):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    starts = [lo for lo, _ in merged]
    for lo, hi in (b["ivs"] if b.get("ivs") is not None else [(b["lo"], b["hi"])]):
        k = bisect.bisect_right(starts, lo) - 1
        if k < 0 or merged[k][1] < hi:
            return False
    return True


def footprint(op, memlocs):
    kind = op.get("kind")
    if kind in NO_MEMORY_KINDS:
        return None
    if kind == "register_ap":
        return runtime_footprint(op, memlocs)
    if kind != "physical_ap":
        raise Undecided(f"operand of kind {kind!r}")
    m = memlocs[op["memref"]]
    dsz = DSZ[op["dtype"]]
    ap, off = op["ap"], op["offset"]
    if m["type"] in ("SB", "PSUM"):
        pitch = m["dims"][1] // dsz
        step0, n0 = ap[0]
        pbase = m.get("base") or 0   # memloc start partition (quadrant 0/32/64/96); tiles in different quadrants share byte addresses
        parts = frozenset(pbase + (off + i * step0) // pitch for i in range(n0))
        col0 = off % pitch
        inner = ap[1:]
        maxoff = sum((n - 1) * s for s, n in inner)
        cross = col0 + maxoff >= pitch
        lo = m["addr"] + col0 * dsz
        hi = m["addr"] + (col0 + maxoff + 1) * dsz
        space = m["type"] if m["type"] == "SB" else f"PSUM.bank{m['bank']}"
        ivs = expand(inner, m["addr"] // dsz + col0, ) if inner else [(m["addr"] // dsz + col0, m["addr"] // dsz + col0 + 1)]
        ivs = None if ivs is None else [(a * dsz, b * dsz) for a, b in ivs]
        return dict(space=space, parts=parts, lo=lo, hi=hi, cross=cross, ivs=ivs,
                    memref=op["memref"], tensor=m["tensor_name"], ap=ap, off=off)
    maxoff = sum((n - 1) * s for s, n in ap)
    space = "DRAM:" + ("INTERNAL" if m["kind"] == "Internal" else op["memref"])
    lo = m["addr"] + off * dsz
    hi = m["addr"] + (off + maxoff + 1) * dsz
    ivs = expand(ap, m["addr"] // dsz + off)
    ivs = None if ivs is None else [(a * dsz, b * dsz) for a, b in ivs]
    return dict(space=space, parts=None, lo=lo, hi=hi, cross=False, ivs=ivs, memref=op["memref"],
                tensor=m["tensor_name"], ap=ap, off=off)


def overlap(a, b):
    if a["space"] != b["space"]:
        return False
    if a["lo"] >= b["hi"] or b["lo"] >= a["hi"]:
        return False
    if a["parts"] is not None and not (a["parts"] & b["parts"]):
        return False
    return ivs_overlap(a.get("ivs"), b.get("ivs"))


def fp_str(fp):
    if fp is None:
        return "-"
    if fp["parts"] is not None:
        ps = sorted(fp["parts"])
        pstr = f"p{ps[0]}" if len(ps) == 1 else (f"p{ps[0]}..{ps[-1]}" if ps == list(range(ps[0], ps[-1] + 1)) else f"p{ps[0]}..{ps[-1]}/step")
        return f"{fp['space']}[{pstr}][{fp['lo']}:{fp['hi']}) {fp['tensor']}"
    return f"{fp['space']}[{fp['lo']}:{fp['hi']}) {fp['tensor']}"


class Model:
    def __init__(self, path):
        self.d, self.f, self.memlocs, self.ins, self.blocks = load(path)
        self.build()

    def build(self):
        ins, blocks, memlocs = self.ins, self.blocks, self.memlocs
        # access nodes: (node_id, issue_node, done_node, engine, order, reads[fp], writes[fp], meta)
        self.acc = []
        self.edges = collections.defaultdict(set)
        self.pos = {}
        self.info = {}
        self.undecided = []    # operands / edges the check cannot model: the dump is UNDECIDED
        self.loop_undecided = []   # (block or None for the dump, why): the loop-carried class's
        # Nodes are positions: an instruction (or DMA block) name that repeats gets `name@k`.
        icount = collections.Counter(i["name"] for i in ins)
        bcount = collections.Counter(b for i in ins for b in i.get("dma_blocks", []))
        seen = collections.Counter()
        self.ids_of = collections.defaultdict(list)
        self.block_of = {}     # node -> name of the block holding it (a DMA's: its issuer's)
        block_at = [b["name"] for b in self.f["blocks"] for _ in b["instructions"]]

        def node_of(name, count, block):
            node = name if count[name] == 1 else f"{name}@{seen[name]}"
            seen[name] += 1
            self.ids_of[name].append(node)
            self.block_of[node] = block
            return node

        def fps(i, node, ops, block):
            """(footprints, runtime-address SBUF/PSUM operands as their whole locations)"""
            out, whole = [], []
            for x in ops:
                try:
                    f = footprint(x, memlocs)
                except Undecided as why:
                    dbg = i.get("debug") or {}
                    text = f"{node} {i['opcode']}@{i['engine']} L{dbg.get('lineno')}: {why}"
                    self.undecided.append(text)
                    w = whole_location(x, memlocs)
                    if w is None:
                        self.loop_undecided.append((block, text))
                    else:
                        whole.append(w)
                    continue
                if f:
                    out.append(f)
            return out, whole

        self.node = []
        self.updates = []      # (node, instruction or block) in program order
        last_on_engine = {}
        done_of = {}
        self.qedges = collections.defaultdict(set)   # extra: same-queue in-order completion
        last_on_queue = {}
        last_barrier = None   # engine "ALL" instructions (AllEngineBarrier, GroupResetSemaphores): every
        self.barriers = []    # engine's earlier instructions precede it, it precedes every engine's later ones
        for k, i in enumerate(ins):
            eng, op = i["engine"], i["opcode"]
            blk = block_at[k]
            name = node_of(i["name"], icount, blk)
            self.node.append(name)
            self.updates.append((name, i))
            self.info[name] = i
            self.pos[name] = k
            if eng == "ALL":
                for e, prev in list(last_on_engine.items()):
                    self.edges[prev].add(name)
                    last_on_engine[e] = name
                if last_barrier is not None:
                    self.edges[last_barrier].add(name)
                last_barrier = name
                self.barriers.append(name)
            elif eng != "Unassigned":
                prev = last_on_engine.get(eng, last_barrier)
                if prev is not None:
                    self.edges[prev].add(name)
                last_on_engine[eng] = name
            if op == "DMACopy":
                done = name + "#done"
                self.edges[name].add(done)
                done_of[name] = done
                self.pos[done] = k
                self.block_of[done] = blk
                reads, whole_reads = fps(i, name, i.get("ins", []), blk)
                writes, whole_writes = fps(i, name, i.get("outs", []), blk)
                self.acc.append(dict(name=name, issue=name, done=done, engine=f"{eng}/DMA:{i.get('queue')}",
                                     order=k, reads=reads, writes=writes, i=i, queue=i.get("queue"),
                                     whole_reads=whole_reads, whole_writes=whole_writes))
                q = i.get("queue")
                if q in last_on_queue:
                    self.qedges[last_on_queue[q]].add(done)
                last_on_queue[q] = done
            elif op == "DMATrigger":
                for bname in i.get("dma_blocks", []):
                    bn = node_of(bname, bcount, blk)
                    self.edges[name].add(bn)
                    self.pos[bn] = k
                    b = blocks.get(bname)
                    reads, writes, whole_reads, whole_writes = [], [], [], []
                    if b:
                        for dsc in b["descs"]:
                            r, wr = fps(i, bn, dsc.get("ins", []), blk)
                            w, ww = fps(i, bn, dsc.get("outs", []), blk)
                            reads += r
                            writes += w
                            whole_reads += wr
                            whole_writes += ww
                        self.info[bn] = b["block"]
                        self.updates.append((bn, b["block"]))
                    self.acc.append(dict(name=bn, issue=bn, done=bn, engine=f"DMA:{i.get('queue')}(trig {eng})",
                                         order=k, reads=reads, writes=writes, i=i, block=b,
                                         queue=i.get("queue"), whole_reads=whole_reads,
                                         whole_writes=whole_writes))
                    q = i.get("queue")
                    if q in last_on_queue:
                        self.qedges[last_on_queue[q]].add(bn)
                    last_on_queue[q] = bn
            else:
                reads, whole_reads = fps(i, name, i.get("ins", []), blk)
                writes, whole_writes = fps(i, name, i.get("outs", []), blk)
                if reads or writes or whole_reads or whole_writes:
                    self.acc.append(dict(name=name, issue=name, done=name, engine=eng, order=k,
                                         reads=reads, writes=writes, i=i, queue=None,
                                         whole_reads=whole_reads, whole_writes=whole_writes))
        self.loops = self._find_loops(block_at)
        self.reached = self.cumulative()
        # wait edges
        self.waits = []        # (waiter, producer node or None, producer's completion, wait)
        for name, i in zip(self.node, ins):
            for w in (i.get("sync_info") or {}).get("on_wait", []):
                src = self.resolve(name, w)
                if src is None:
                    self.waits.append((name, None, None, w))
                    continue
                src_done = done_of.get(src, src)
                self.edges[src_done].add(name)
                self.waits.append((name, src, src_done, w))
        cyc = self.cycle_nodes()
        if cyc:
            self.undecided.append(f"happens-before graph has a cycle through {cyc} nodes: no pair's order can be read off it")
        # reachability (DFS memo)
        self._reach = {}
        self._loop_pairs = None
        self._body_updates = {}

    def _find_loops(self, block_at):
        """{loop body: its instruction nodes in program order}: the blocks holding a branch
        that names the block itself. A branch to an earlier block (a loop of several blocks)
        is recorded in `loop_undecided` for the dump."""
        index = {b["name"]: n for n, b in enumerate(self.f["blocks"])}
        loops = {}
        for name, blk in zip(self.node, block_at):
            i = self.info[name]
            if "Branch" not in i["opcode"]:
                continue
            for field, target in i.items():
                if not isinstance(target, str) or target not in index:
                    continue
                if target == blk:
                    loops[blk] = []
                elif index[target] < index[blk]:
                    self.loop_undecided.append(
                        (None, f"{name} in {blk} branches ({field}) back to the earlier block "
                               f"{target}: a loop of several blocks is not modelled"))
        for name, blk in zip(self.node, block_at):
            if blk in loops:
                loops[blk].append(name)
        return loops

    def cumulative(self):
        """{(node, sem): the semaphore's cumulative count right after that node's update}.
        Each sem id belongs to one engine or queue, whose updates land in program order; a
        GroupResetSemaphores (block boundary of a device loop) zeroes its group. A reset that
        names no group leaves every count after it unknown, so it makes the dump undecided
        (12:30Z, as the repository check; the 10:05Z copy zeroed every counter)."""
        cum, after = collections.Counter(), {}
        for node, i in self.updates:
            if i.get("opcode") == "GroupResetSemaphores":
                group = i.get("sema_group")
                if group is None:
                    self.undecided.append(f"{node} GroupResetSemaphores@{i.get('engine')}: "
                                          f"no sema_group, so the counts after it are unknown")
                for sem in group or []:
                    cum[sem] = 0
            for u in (i.get("sync_info") or {}).get("on_update", []):
                cum[u["id"]] += u["update_value"]
                after[(node, u["id"])] = cum[u["id"]]
        return after

    def resolve(self, waiter, w):
        """The node `waiter`'s wait `w` waits for (module docstring, "Which instance a wait
        names"): the instances before the waiter in its own block, else those of the nearest
        earlier block that holds the name; of these, the first whose cumulative count reaches
        the wait value, else the last. None when no instance comes before the waiter
        (LATER-PRODUCER); the name itself when no instruction has it (NO-UPDATE-FOUND). A wait
        in a loop body whose own body holds the name only after it also waits for the previous
        iteration's instance, which the loop-carried class does not model."""
        inst = self.ids_of.get(w["from"])
        if not inst:
            return w["from"]
        k, blk = self.pos[waiter], self.block_of[waiter]
        earlier = [n for n in inst if self.pos[n] < k]
        own = [n for n in earlier if self.block_of[n] == blk]
        if not own and blk in self.loops and any(self.block_of[n] == blk for n in inst):
            self.loop_undecided.append(
                (blk, f"{waiter} waits for {w['from']}, which its loop body runs only after "
                      f"it: the previous iteration's instance is not modelled"))
        if not earlier:
            return None
        nearest = self.block_of[earlier[-1]]
        nodes = own or [n for n in earlier if self.block_of[n] == nearest]
        for n in nodes:
            if self.reached.get((n, w["id"]), -1) >= w["wait_value"]:
                return n
        return nodes[-1]

    def cycle_nodes(self):
        """Number of nodes on or behind a cycle of the happens-before graph (0: acyclic)."""
        succ = collections.defaultdict(set)
        for e in (self.edges, self.qedges):
            for n, after in e.items():
                succ[n] |= after
        nodes = set(succ).union(*succ.values()) if succ else set()
        indeg = dict.fromkeys(nodes, 0)
        for after in succ.values():
            for n in after:
                indeg[n] += 1
        order = [n for n in nodes if not indeg[n]]
        for n in order:
            for m in succ.get(n, ()):
                indeg[m] -= 1
                if not indeg[m]:
                    order.append(m)
        return len(nodes) - len(order)

    def succ(self, n, with_queue):
        out = set(self.edges.get(n, ()))
        if with_queue:
            out |= self.qedges.get(n, set())
        return out

    def reach(self, a, with_queue=False):
        key = (a, with_queue)
        if key in self._reach:
            return self._reach[key]
        seen, stack = set(), [a]
        while stack:
            n = stack.pop()
            for m in self.succ(n, with_queue):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        self._reach[key] = seen
        return seen

    def ordered(self, p, c, with_queue=False):
        """producer's completion happens-before consumer's start?"""
        return c["issue"] in self.reach(p["done"], with_queue)

    def path(self, a, b, with_queue=False):
        # BFS path for display
        prev = {a: None}
        q = collections.deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for m in self.succ(n, with_queue):
                if m not in prev:
                    prev[m] = n
                    q.append(m)
        if b not in prev:
            return None
        out = []
        n = b
        while n is not None:
            out.append(n)
            n = prev[n]
        return list(reversed(out))

    def unconfirmed_because(self, P, C, fp, fc):
        """Why the UNSYNC footprint pair (P's `fp`, C's `fc`) is not confirmed: "runtime-address"
        or "same-value TensorSave"; None when it is confirmed (module docstring, "The one-pass
        verdict")."""
        if fp.get("runtime") or fc.get("runtime"):
            return "runtime-address"
        if (P["i"]["opcode"] == C["i"]["opcode"] == "TensorSave" and fp["memref"] == fc["memref"]
                and ENGINE_SUFFIX.sub("", P["i"]["name"]) == ENGINE_SUFFIX.sub("", C["i"]["name"])):
            return "same-value TensorSave"
        return None

    def one_pass_verdict(self, unsync, waits_not_ok):
        """(word, counts) of the one-pass VERDICT line (module docstring): `unsync` the UNSYNC
        footprint pairs (kind, P, C, fp, fc), `waits_not_ok` the number of waits whose
        arithmetic is not OK."""
        why = collections.Counter(self.unconfirmed_because(P, C, fp, fc)
                                  for _, P, C, fp, fc in unsync)
        edges_unknown = self.cycle_nodes() > 0 or any(
            i.get("opcode") == "GroupResetSemaphores" and i.get("sema_group") is None
            for i in self.ins)
        confirmed = 0 if edges_unknown else why[None]
        if confirmed or (waits_not_ok and not edges_unknown):
            word = "FINDINGS"
        elif self.undecided or why["runtime-address"]:
            word = "UNDECIDED"
        else:
            word = "CLEAN"
        counts = (f"{confirmed} confirmed unsynchronized pair{'' if confirmed == 1 else 's'}; "
                  f"not confirmed: {why['same-value TensorSave']} same-value TensorSave, "
                  f"{why['runtime-address']} through a runtime-address operand"
                  + (f", {why[None]} with the edges unknown" if edges_unknown else "")
                  + f"; {len(self.undecided)} "
                  f"{'operand or edge' if len(self.undecided) == 1 else 'operands or edges'} "
                  f"undecided; {waits_not_ok} wait{'' if waits_not_ok == 1 else 's'} not OK")
        return word, counts

    def label(self, a):
        i = a["i"]
        dbg = i.get("debug") or {}
        return f"{a['name']} {i['opcode']}@{a['engine']} L{dbg.get('lineno')} {dbg.get('op_name')}"

    def conflicts(self, lo=None, hi=None):
        out = []
        acc = self.acc
        for x in range(len(acc)):
            P = acc[x]
            for y in range(x + 1, len(acc)):
                C = acc[y]
                if lo is not None:
                    lp = (P["i"].get("debug") or {}).get("lineno") or 0
                    lc = (C["i"].get("debug") or {}).get("lineno") or 0
                    if not (lo <= lp <= hi or lo <= lc <= hi):
                        continue
                for pw in P["writes"]:
                    for cr in C["reads"]:
                        if overlap(pw, cr):
                            out.append(("RAW", P, C, pw, cr))
                    for cw in C["writes"]:
                        if overlap(pw, cw):
                            out.append(("WAW", P, C, pw, cw))
                for pr in P["reads"]:
                    for cw in C["writes"]:
                        if overlap(pr, cw):
                            out.append(("WAR", P, C, pr, cw))
        return out

    def check_sem_arith(self):
        """Every wait against its producer's semaphore count: (status, waiter, producer,
        semaphore, wait value, count). The count is the cumulative one (Model.cumulative), the
        first iteration's inside a loop body. A wait on a producer of its own loop body is
        checked for the later iterations as well (Model.later_count): its status gets the
        suffix " (iteration 2+)" when only those fail, and is ACCUMULATES when the body never
        resets the semaphore."""
        res = []
        for waiter, src, src_done, w in self.waits:
            if src is None:
                res.append(("LATER-PRODUCER", waiter, w["from"], w["id"], w["wait_value"], None))
                continue
            exp = self.reached.get((src, w["id"]))
            status = self._status(w["wait_value"], exp)
            body = self.block_of.get(waiter)
            if status == "OK" and body in self.loops and self.block_of.get(src) == body:
                later = self.later_count(body, src, w["id"])
                if later is None:
                    status = "ACCUMULATES"
                elif self._status(w["wait_value"], later) != "OK":
                    status = f"{self._status(w['wait_value'], later)} (iteration 2+)"
                    exp = later
            res.append((status, waiter, src, w["id"], w["wait_value"], exp))
        return res

    @staticmethod
    def _status(value, count):
        if count is None:
            return "NO-UPDATE-FOUND"
        if value < count:
            return "UNSAFE-EARLY"
        if value > count:
            return "OVER-WAIT"
        return "OK"

    def later_count(self, body, src, sem):
        """The count of semaphore `sem` right after `src`'s update in the second and later
        iterations of loop `body`: its updates in the body since the body's last reset of it
        before `src`, else since the body's last reset of it (over the back-edge). None when
        the body never resets it. A reset that names no group (the dump is UNDECIDED for it)
        is taken to reset every semaphore."""
        if body not in self._body_updates:
            seq = [(n, i) for n, i in self.updates if self.block_of.get(n) == body]
            self._body_updates[body] = (seq, {n: j for j, (n, _) in enumerate(seq)})
        seq, index = self._body_updates[body]
        at = index[src]
        resets = [j for j, (_, i) in enumerate(seq) if i.get("opcode") == "GroupResetSemaphores"
                  and (i.get("sema_group") is None or sem in i["sema_group"])]
        if not resets:
            return None
        before = [j for j in resets if j < at]
        window = seq[before[-1] + 1:at + 1] if before else seq[resets[-1] + 1:] + seq[:at + 1]
        return sum(u["update_value"] for _, i in window
                   for u in (i.get("sync_info") or {}).get("on_update", []) if u["id"] == sem)

    # ---- the loop-carried class (module docstring) ----

    def loop_carried(self):
        """Every pair of the loop-carried class in every loop body, OK pairs included."""
        if self._loop_pairs is None:
            self._loop_pairs = [p for body in self.loops for p in self._carried_pairs(body)]
        return self._loop_pairs

    def loop_carried_findings(self):
        """The class's findings: RAW pairs that are not OK, WAR/WAW pairs that are UNSYNC."""
        return [p for p in self.loop_carried()
                if p.tag == "UNSYNC" or (p.kind == "RAW" and p.tag != "OK")]

    def loop_undecided_reasons(self):
        """What the class could not model in this dump: its loops, or the dump's control flow."""
        self.loop_carried()
        return [why for blk, why in self.loop_undecided if blk is None or blk in self.loops]

    def loop_carried_verdict(self):
        """FINDINGS (a member found; a part not modelled may hold more), UNDECIDED (none
        found, part of a loop not modelled) or CLEAN (none, every loop modelled, or no loop)."""
        if self.loop_carried_findings():
            return "FINDINGS"
        if self.loop_undecided_reasons():
            return "UNDECIDED"
        return "CLEAN"

    def back_edge_sequence(self, body):
        """The body's longest suffix of instructions with no memory operand."""
        accessing = {a["issue"] for a in self.acc}
        tail = []
        for n in reversed(self.loops[body]):
            i = self.info[n]
            if (n in accessing or i["opcode"] in ("DMACopy", "DMATrigger")
                    or any(x.get("kind") not in NO_MEMORY_KINDS
                           for x in i.get("ins", []) + i.get("outs", []))):
                break
            tail.append(n)
        return set(tail)

    def _crossings(self, body, drop=frozenset()):
        """Program-order edges from iteration i into iteration i+1, {node: nodes}: build()'s
        engine-order and all-engine-barrier rules over the body twice, the first time without
        the nodes in `drop`."""
        nodes = self.loops[body]
        out = collections.defaultdict(set)
        last_on, last_bar = {}, None
        for it, seq in ((0, [n for n in nodes if n not in drop]), (1, nodes)):
            for n in seq:
                eng, me = self.info[n]["engine"], (n, it)
                if eng == "ALL":
                    preds = list(last_on.values()) + ([last_bar] if last_bar else [])
                    for e in last_on:
                        last_on[e] = me
                    last_bar = me
                elif eng != "Unassigned":
                    prev = last_on.get(eng, last_bar)
                    preds = [prev] if prev else []
                    last_on[eng] = me
                else:
                    preds = []
                for pn, pit in preds:
                    if pit == 0 and it == 1:
                        out[pn].add(n)
        return out

    def _queue_crossings(self, body, accs):
        """Same-queue completion order from iteration i into iteration i+1: each queue's last
        DMA of the body to its first one."""
        first, last = {}, {}
        for a in accs:
            if a["queue"] is not None:
                first.setdefault(a["queue"], a["done"])
                last[a["queue"]] = a["done"]
        out = collections.defaultdict(set)
        for q, done in last.items():
            out[done].add(first[q])
        return out

    def _carried_pairs(self, body):
        members = {n for n, b in self.block_of.items() if b == body}
        succ = {n: [v for v in self.edges.get(n, ()) if v in members] for n in members}
        qsucc = {n: [v for v in self.qedges.get(n, ()) if v in members] for n in members}
        order = self._topo(members, succ, qsucc)
        if order is None:
            self.loop_undecided.append((body, f"{body}: its one-pass graph has a cycle"))
            return []
        accs = sorted((a for a in self.acc if self.block_of[a["name"]] == body),
                      key=lambda a: a["order"])
        tail = self.back_edge_sequence(body)
        bit = {n: 1 << j for j, n in enumerate(order)}

        def closure(*edge_sets):
            """node -> the nodes it reaches in one iteration, itself included"""
            out = {}
            for n in reversed(order):
                m = bit[n]
                for e in edge_sets:
                    for v in e[n]:
                        m |= out[v]
                out[n] = m
            return out

        def from_prev(edge_sets, crossings, into, cut=frozenset()):
            """node of iteration i -> the nodes of iteration i+1 it reaches, not through `cut`"""
            out = {}
            for n in reversed(order):
                m = 0
                if n not in cut:
                    for e in edge_sets:
                        for v in e[n]:
                            m |= out[v]
                    for c in crossings:
                        for v in c.get(n, ()):
                            m |= into[v]
                out[n] = m
            return out

        program = self._crossings(body)
        within, within_q = closure(succ), closure(succ, qsucc)
        explicit = from_prev([succ], [self._crossings(body, tail)], within, tail)
        full = from_prev([succ], [program], within)
        queued = from_prev([succ, qsucc], [program, self._queue_crossings(body, accs)], within_q)

        def tag(src, dst):
            b = bit[dst]
            if explicit[src] & b:
                return "OK"
            if full[src] & b:
                return "BACK-EDGE-ONLY"
            if queued[src] & b:
                return "QUEUE-ORDER-ONLY"
            return "UNSYNC"

        def on_chip(fps):
            return [f for f in fps if f["space"] == "SB" or f["space"].startswith("PSUM")]

        writes, reads = collections.defaultdict(list), collections.defaultdict(list)
        for a in accs:
            for f in on_chip(a["writes"] + a["whole_writes"]):
                writes[f["space"]].append((a, f))
            for f in on_chip(a["reads"] + a["whole_reads"]):
                reads[f["space"]].append((a, f))
        pairs, seen = [], set()

        def add(kind, P, C, fp, fc):
            key = (kind, P["done"], C["issue"])
            if key in seen:
                return
            seen.add(key)
            exact = fc if fp.get("runtime") else fp
            listed = any(d[0] == P["i"]["name"] and d[1] == LCD_KIND[kind]
                         for d in C["i"].get("loop_carried_dependencies") or [])
            pairs.append(LoopPair(body, kind, tag(P["done"], C["issue"]), P["done"], C["issue"],
                                  exact["tensor"] or exact["memref"], fp, fc,
                                  bool(fp.get("runtime") or fc.get("runtime")), listed, P, C))

        def near(fps, fc):
            return [(a, f) for a, f in fps if f["lo"] < fc["hi"] and fc["lo"] < f["hi"]]

        # RAW: C reads in iteration i+1 what P, at or after C in the body, wrote in iteration i
        for space, rs in reads.items():
            for C, fc in rs:
                cands = near(writes[space], fc)
                exact = [(W, fw) for W, fw in cands if not fw.get("runtime")]
                if any(W["order"] < C["order"] and covers(fw, fc) for W, fw in exact):
                    continue        # iteration i+1 writes it before C reads it
                for P, fp in cands:
                    if P["order"] < C["order"] or not overlap(fp, fc):
                        continue
                    if any(W["order"] > P["order"] and covers(fw, fc) for W, fw in exact):
                        continue    # iteration i writes it again after P
                    add("RAW", P, C, fp, fc)
        # WAR (P reads in iteration i, C writes in i+1) and WAW (both write); C at or before P
        for space, ws in writes.items():
            for C, fc in ws:
                for kind, fps in (("WAR", reads[space]), ("WAW", ws)):
                    for P, fp in near(fps, fc):
                        if P["order"] >= C["order"] and overlap(fp, fc):
                            add(kind, P, C, fp, fc)
        return pairs

    @staticmethod
    def _topo(members, *edge_sets):
        indeg = dict.fromkeys(members, 0)
        for e in edge_sets:
            for n in members:
                for v in e[n]:
                    indeg[v] += 1
        order = [n for n in members if not indeg[n]]
        for n in order:
            for e in edge_sets:
                for v in e[n]:
                    indeg[v] -= 1
                    if not indeg[v]:
                        order.append(v)
        return order if len(order) == len(members) else None


#: Each pair kind's name in walrus's `loop_carried_dependencies`.
LCD_KIND = {"RAW": "Flow", "WAR": "Anti", "WAW": "Output"}

LoopPair = collections.namedtuple(
    "LoopPair", "loop kind tag src dst tile src_fp dst_fp runtime listed src_access dst_access")
LoopPair.__doc__ = """One pair of the loop-carried class in loop body `loop`.

`src` is the node of the access in iteration i whose completion must come first (a DMA's
completion node), `dst` the node of the access in iteration i+1 that must start after it
(RAW: writer, reader; WAR: reader, writer; WAW: both writers). `tag`: OK, BACK-EDGE-ONLY,
QUEUE-ORDER-ONLY or UNSYNC (module docstring). `tile`: the tensor (else memory location) of
the exact footprint. `runtime`: a footprint is a runtime-address operand taken whole.
`listed`: the compiler's `loop_carried_dependencies` on `dst` names `src`'s instruction with
the pair's kind."""


def main():
    path = sys.argv[1]
    lo = hi = None
    verbose = "--verbose" in sys.argv
    show_all = "--all" in sys.argv
    for a in sys.argv[2:]:
        if a.startswith("--lines"):
            rng = a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            lo, hi = map(int, rng.split("-"))
    M = Model(path)
    print(f"# {path}")
    print(f"instructions={len(M.ins)} access-nodes={len(M.acc)} waits={len(M.waits)}")
    crosses = [(a['name'], fp_str(f)) for a in M.acc for f in a['reads'] + a['writes'] if f['cross']]
    if crosses:
        print("WARNING: inner AP crosses partition pitch (footprint approx):", crosses[:5])
    print("\n## semaphore wait arithmetic")
    bad = 0
    for st, waiter, src, sid, v, exp in M.check_sem_arith():
        if st != "OK":
            bad += 1
            print(f"  {st}: {waiter} waits sem{sid}>={v} for {src}; producer's count = {exp}")
    print(f"  {len(M.waits)} waits checked, {bad} not OK")
    print("\n## conflicting pairs (producer -> consumer) lacking happens-before")
    conf = M.conflicts(lo, hi)
    n_ok = n_bad = 0
    seen = set()
    n_q = 0
    unsync = []
    for kind, P, C, fa, fb in conf:
        ok = M.ordered(P, C)
        okq = ok or M.ordered(P, C, True)
        key = (P["name"], C["name"], kind)
        if ok:
            n_ok += 1
        elif okq:
            n_q += 1
        else:
            n_bad += 1
            unsync.append((kind, P, C, fa, fb))
        if (not ok) or show_all:
            if key in seen and not verbose:
                continue
            seen.add(key)
            deps = [d[0] for d in (C["i"].get("dependencies") or [])]
            dep_flag = "dep-listed" if P["name"] in deps or P["name"].replace("#done", "") in deps else "NO-DEP"
            tag = "OK " if ok else ("QUEUE-ORDER-ONLY" if okq else "UNSYNC")
            because = None if okq else M.unconfirmed_because(P, C, fa, fb)
            print(f"  [{tag}] {kind} {M.label(P)}  ->  {M.label(C)}")
            print(f"         prod {fp_str(fa)}")
            print(f"         cons {fp_str(fb)}   ({dep_flag}"
                  f"{f'; not confirmed: {because}' if because else ''})")
            if okq and (verbose or not ok):
                print(f"         via {' -> '.join(M.path(P['done'], C['issue'], not ok) or [])}")
    if "--pairs" in sys.argv:
        print("\n## every overlapping pair in the selected line range, with its ordering path")
        seen2 = set()
        for kind, P, C, fa, fb in conf:
            key = (P["name"], C["name"], kind)
            if key in seen2:
                continue
            seen2.add(key)
            ok = M.ordered(P, C); okq = ok or M.ordered(P, C, True)
            tag = "OK" if ok else ("QUEUE-ORDER-ONLY" if okq else "UNSYNC")
            same_eng = P["engine"] == C["engine"] and "DMA" not in P["engine"]
            path = M.path(P["done"], C["issue"], not ok) if okq else None
            how = "same engine, in-order" if (same_eng and path and len(path) <= 2) else (" -> ".join(path) if path else "NONE")
            print(f"  [{tag}] {kind} {M.label(P)} {fp_str(fa)}")
            print(f"        -> {M.label(C)} {fp_str(fb)}")
            print(f"        via: {how}")
    print(f"\n  {len(conf)} overlapping pairs: {n_ok} ordered by engine order + semaphores, "
          f"{n_q} ordered only if same-queue DMAs complete in order, {n_bad} UNSYNCHRONIZED")
    runtime = sum(1 for a in M.acc for f in a["reads"] + a["writes"] if f.get("runtime"))
    if runtime:
        print(f"  {runtime} DRAM footprints are runtime-address operands, taken as their whole tensor "
              f"(an UNSYNC pair through one needs its real address range argued)")
    if M.undecided:
        print(f"\n## UNDECIDED: {len(M.undecided)} operands or edges the check cannot model")
        for u in M.undecided[:20]:
            print(f"  {u}")
    word, counts = M.one_pass_verdict(unsync, bad)
    print(f"VERDICT: {word} ({counts})")
    print_loop_carried(M, show_all)


def print_loop_carried(M, show_all=False):
    """The loop-carried class of `M` (module docstring): per loop, its pair counts by kind and
    tag; each finding (every pair with `show_all`); the reasons it is undecided; its verdict."""
    print("\n## loop-carried pairs (iteration i -> i+1; module docstring)")
    pairs = M.loop_carried()
    findings = {id(p) for p in M.loop_carried_findings()}
    for body in M.loops:
        mine = [p for p in pairs if p.loop == body]
        tail = M.back_edge_sequence(body)
        counts = collections.Counter((p.kind, p.tag) for p in mine)
        print(f"  {body}: {len(M.loops[body])} instructions, back-edge sequence {len(tail)}; "
              + (", ".join(f"{k} {t} {n}" for (k, t), n in sorted(counts.items())) or "no pairs"))
        for p in mine:
            if id(p) not in findings and not show_all:
                continue
            listed = "listed" if p.listed else "NOT listed"
            where = " (runtime address, whole tile)" if p.runtime else ""
            print(f"  [{p.tag}] {p.kind} {M.label(p.src_access)}  ->  next iteration  "
                  f"{M.label(p.dst_access)}")
            print(f"         prev {fp_str(p.src_fp)}")
            print(f"         next {fp_str(p.dst_fp)}{where}   (loop_carried_dependencies: {listed})")
    reasons = M.loop_undecided_reasons()
    if reasons:
        print(f"  UNDECIDED: {len(reasons)} parts of the loops the class cannot model")
        for why in reasons[:20]:
            print(f"    {why}")
    verdict = M.loop_carried_verdict()
    print(f"VERDICT (loop-carried): {verdict} ({len(findings)} findings in {len(M.loops)} loops)")


if __name__ == "__main__":
    main()
