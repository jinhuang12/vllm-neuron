"""Check every RAW/WAR/WAW pair in a walrus `--print-after=lower_sync` BIR dump.

usage: depcheck.py DUMP.json [--lines LO-HI] [--all] [--verbose]

For every pair of memory accesses with overlapping footprints (SBUF: partition set x
byte interval; DRAM: byte interval in the tensor's space, internal tensors share one
arena) the script asks whether the consumer's start happens after the producer's
completion in the happens-before graph built from (1) same-engine program order and
(2) the semaphore waits (`sync_info.on_wait[*].from`). DMA transfers are modelled as
issue node (the DMACopy instruction or DMATrigger, ordered with its engine) plus a
completion node (the DMACopy name or the DMABlock name; only semaphore waits leave it).
It also checks the arithmetic of every wait: the wait value must be >= the cumulative
update count of the producer's semaphore at the producer's completion.
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

import json, sys, collections

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
        # Nodes are positions: an instruction (or DMA block) name that repeats gets `name@k`.
        icount = collections.Counter(i["name"] for i in ins)
        bcount = collections.Counter(b for i in ins for b in i.get("dma_blocks", []))
        seen = collections.Counter()
        self.ids_of = collections.defaultdict(list)

        def node_of(name, count):
            node = name if count[name] == 1 else f"{name}@{seen[name]}"
            seen[name] += 1
            self.ids_of[name].append(node)
            return node

        def fps(i, node, ops):
            out = []
            for x in ops:
                try:
                    f = footprint(x, memlocs)
                except Undecided as why:
                    dbg = i.get("debug") or {}
                    self.undecided.append(f"{node} {i['opcode']}@{i['engine']} L{dbg.get('lineno')}: {why}")
                    continue
                if f:
                    out.append(f)
            return out

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
            name = node_of(i["name"], icount)
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
                reads = fps(i, name, i.get("ins", []))
                writes = fps(i, name, i.get("outs", []))
                self.acc.append(dict(name=name, issue=name, done=done, engine=f"{eng}/DMA:{i.get('queue')}",
                                     order=k, reads=reads, writes=writes, i=i))
                q = i.get("queue")
                if q in last_on_queue:
                    self.qedges[last_on_queue[q]].add(done)
                last_on_queue[q] = done
            elif op == "DMATrigger":
                for bname in i.get("dma_blocks", []):
                    bn = node_of(bname, bcount)
                    self.edges[name].add(bn)
                    self.pos[bn] = k
                    b = blocks.get(bname)
                    reads, writes = [], []
                    if b:
                        for dsc in b["descs"]:
                            reads += fps(i, bn, dsc.get("ins", []))
                            writes += fps(i, bn, dsc.get("outs", []))
                        self.info[bn] = b["block"]
                        self.updates.append((bn, b["block"]))
                    self.acc.append(dict(name=bn, issue=bn, done=bn, engine=f"DMA:{i.get('queue')}(trig {eng})",
                                         order=k, reads=reads, writes=writes, i=i, block=b))
                    q = i.get("queue")
                    if q in last_on_queue:
                        self.qedges[last_on_queue[q]].add(bn)
                    last_on_queue[q] = bn
            else:
                reads = fps(i, name, i.get("ins", []))
                writes = fps(i, name, i.get("outs", []))
                if reads or writes:
                    self.acc.append(dict(name=name, issue=name, done=name, engine=eng, order=k,
                                         reads=reads, writes=writes, i=i))
        self.reached = self.cumulative()
        # wait edges
        self.waits = []
        for name, i in zip(self.node, ins):
            for w in (i.get("sync_info") or {}).get("on_wait", []):
                src = self.producer(w)
                src_done = done_of.get(src, src)
                self.edges[src_done].add(name)
                self.waits.append((name, src, src_done, w))
        cyc = self.cycle_nodes()
        if cyc:
            self.undecided.append(f"happens-before graph has a cycle through {cyc} nodes: no pair's order can be read off it")
        # reachability (DFS memo)
        self._reach = {}

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

    def producer(self, w):
        """The node a wait waits for: its `from` instruction or, for a repeated name, the first
        instance whose cumulative update reaches the wait value."""
        nodes = self.ids_of.get(w["from"]) or [w["from"]]
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
        """Every wait against its producer's cumulative semaphore count (Model.cumulative)."""
        res = []
        for waiter, src, src_done, w in self.waits:
            exp = self.reached.get((src, w["id"]))
            status = "OK"
            if exp is None:
                status = "NO-UPDATE-FOUND"
            elif w["wait_value"] < exp:
                status = "UNSAFE-EARLY"
            elif w["wait_value"] > exp:
                status = "OVER-WAIT"
            res.append((status, waiter, src, w["id"], w["wait_value"], exp))
        return res


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
            print(f"  {st}: {waiter} waits sem{sid}>={v} for {src}; producer's cumulative value = {exp}")
    print(f"  {len(M.waits)} waits checked, {bad} not OK")
    print("\n## conflicting pairs (producer -> consumer) lacking happens-before")
    conf = M.conflicts(lo, hi)
    n_ok = n_bad = 0
    seen = set()
    n_q = 0
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
        if (not ok) or show_all:
            if key in seen and not verbose:
                continue
            seen.add(key)
            deps = [d[0] for d in (C["i"].get("dependencies") or [])]
            dep_flag = "dep-listed" if P["name"] in deps or P["name"].replace("#done", "") in deps else "NO-DEP"
            tag = "OK " if ok else ("QUEUE-ORDER-ONLY" if okq else "UNSYNC")
            print(f"  [{tag}] {kind} {M.label(P)}  ->  {M.label(C)}")
            print(f"         prod {fp_str(fa)}")
            print(f"         cons {fp_str(fb)}   ({dep_flag})")
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
        print("VERDICT: UNDECIDED (0 pairs above is not evidence)")
    else:
        print(f"VERDICT: {'CLEAN' if n_bad == 0 and bad == 0 else 'FINDINGS'}")


if __name__ == "__main__":
    main()
