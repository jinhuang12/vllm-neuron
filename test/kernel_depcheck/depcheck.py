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
undecided or a pair goes through a runtime address; else CLEAN. The counts follow the word,
always all of them and in this order, so `VERDICT_LINE` parses the line:

  VERDICT: <word> (confirmed <n>: SB-RAW <n> SB-WAW <n> SB-WAR <n> PSUM <n> DRAM <n>;
    not confirmed: same-value-TensorSave <n> runtime-address <n> edges-unknown <n>;
    undecided <n>: runtime-SB-PSUM-operand <n> other-operand <n> cycle <n> ungrouped-reset <n>;
    waits-not-OK <n>)

on one line. The pair counts are UNSYNC footprint pairs (the unit of the `UNSYNCHRONIZED`
count above it). `confirmed` splits by space (SB by kind; PSUM and DRAM all kinds);
`edges-unknown` counts the pairs that would be confirmed if the edges were known.
`undecided` is the number of entries under `## UNDECIDED`, split by cause: an SBUF/PSUM
operand at a runtime address, another operand the check cannot place, a cycle, a
GroupResetSemaphores that names no group. trn2-2's verdicts.json columns read: sbuf_raw_unsync
= SB-RAW, tensorsave_waw_unsync = same-value-TensorSave, other_unsync_classes = the other
pair counts, undecided_class = the four undecided counts.

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

Linked dumps (`Program`). A dump of a linked program holds several functions: the kernels and
`main`. Each of `main`'s `Call` instructions runs the function its `target` names and binds
the function's parameters to locations of `main` (`arguments`: {a set name of the function: a
set name of `main`}, a set name being an allocation's `name`). `main` can call a function more
than once. Each call is checked as a one-function dump is, and its VERDICT lines name the
function and the call: `VERDICT (<function> <call>): ...` (the counts of `VERDICT_LINE`) and
`VERDICT (<function> <call> loop-carried): ...`. `main` is checked too, and so is a function
that `main` does not call, with its parameters unbound. In the check of a call:
  - an operand's location is looked up in the allocations of the function its
    `memref_owner_func` names (`main`, its own function or another one), else of its own
    function, never by its name in another function: the functions of one program often have
    locations of one name at different addresses;
  - a location of the function that the call binds is the location of `main` it binds it to;
    a parameter of another function is the location of `main` that function's one call binds
    it to (UNDECIDED, other-operand, when `main` calls that function more than once);
  - a DMATrigger whose `dma_blocks` has one block per call of its function starts the block
    of this call (block k of call k), and a wait on any of those blocks waits on that one
    (walrus names the first call's block in the wait). A list of another length than 1 starts
    every block, and the call is UNDECIDED (other-operand).
In a linked dump the local DRAM arena, the functions' `Internal` locations and the
`InternalInterface` locations of `main`, is one space by address; another DRAM tensor is a
space of its owner (`DRAM:<owner>:<name>`). The heading of each check counts its operands by
owner: its function's own locations, `main`'s (named by `memref_owner_func` or bound by the
call's arguments) and each other function's. The functions' section also counts each
function's operands in the dump by `memref_owner_func` (`Program.census`; an operand without
one is its function's own): over its instructions and each DMA block its DMATriggers name,
once per name in a `dma_blocks` list.

Cross-function pairs. The accesses of two calls to memory of more than one function are paired
as the accesses of one function are, the earlier call's access first; two calls of one function
are paired too. That memory (`Program.in_scope`) is: `main`'s locations (after the bindings), a
location that a function other than its owner and `main` names by `memref_owner_func` (the
owner's accesses to it too), and the bytes of the local arena those locations hold (any
location there on those bytes). The happens-before graph of the program is the graphs of the
calls, joined in call order by the program-order rules of one function: an engine runs its
instructions in order across calls, an all-engine instruction orders every engine, and the
DMAs of one queue complete in issue order (QUEUE-ORDER-ONLY), except on a queue that `main`
switches (`SwitchQueueInstance`). The check does not build that graph: from call to call it
carries which last nodes of the engines, of the all-engine instructions and of the queues an
access reaches (`JoinedOrder`), so its memory is that of the largest call. `main`'s own
instructions are not in the order; they can only add order. Without a call order (no `main`, a
Call of a function the dump does not have, no Call) every pair is UNCHECKED (each is listed),
and so is every pair when a call's graph has a cycle. `VERDICT (cross-function)` is FINDINGS
when an UNSYNC pair is confirmed (as in the one-pass verdict), UNDECIDED without a call order
(also with no pair: an unbound parameter's memory is not known) or when a pair is UNCHECKED or
goes through a runtime address, else CLEAN. `VERDICT (program)` is FINDINGS when a VERDICT line
of the dump is FINDINGS, else UNDECIDED when one is UNDECIDED, else CLEAN. A linked dump prints
no `VERDICT:` line, so a reader of `VERDICT_LINE` cannot take one call's verdict for the dump's.

Limits of the linked check:
  - memory that one kernel uses alone is not paired across calls: SBUF/PSUM locations and a
    kernel's own DRAM that no other function names, off the arena bytes above, are checked
    inside each call only, also when two calls (or two kernels) reuse them;
  - a `Const` location is a space of its owner (`DRAM:<owner>:<name>`), as in a one-function
    dump, also when it lies in the local arena: another function's location on its bytes is
    not paired with it;
  - a wait binds to a producer of its own call only, and the semaphore counts of each call
    start at zero;
  - a DMA that its call does not wait for stays unordered after the call.
"""

# CHANGELOG
# 2026-10-08 ~05:50Z  footprint(): SBUF/PSUM partitions start at the memloc's `base` (quadrant).
#                     Every count before this is wrong; corrected counts: REPORT.md 9.5.
# 2026-10-08 ~06:45Z  engine ALL (AllEngineBarrier, GroupResetSemaphores) orders every engine;
#                     GroupResetSemaphores zeroes its group in the wait audit (REPORT.md 10, 11).
# 2026-10-08 07:50Z   the repository check is test/vllm_neuron/functional/dsa/sbuf_order.py
#                     (equal on 313 dumps).
# 2026-10-08 10:05Z   three defects fixed; CHANGELOG.md:
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
# 2026-10-13          Model.reach reads a bit-set closure, one per relation, built once in
#                     reverse topological order: the DFS memo kept one set per producer node
#                     and ran out of memory on a large dump. The output is the same on an
#                     acyclic graph. A cycle now stops the pair classification: the pair
#                     section names one cycle, and the dump is UNDECIDED as before.
# 2026-10-13          Model.conflicts finds the overlapping pairs with a sweep: per space, the
#                     footprints in order of address, each against the open ones only. The
#                     loop over every pair of access nodes took hours on a large dump. main()
#                     classifies the pairs as they come, and keeps only the ones it prints
#                     (all of them with --all or --pairs). The output is the same.
# 2026-10-14          A dump of several functions was read as its first function only, and an
#                     operand's location was looked up by name in that function's allocations:
#                     a location of another function (`memref_owner_func`) whose name the
#                     function also has was read at the wrong address, and one whose name it
#                     does not have raised KeyError. Now (Program) every call of main is
#                     checked with its bindings and its own DMA blocks, each operand in its
#                     owner's allocations (main's or another function's), and the pairs of two
#                     calls on memory of more than one function in main's call order
#                     (UNCHECKED without one); each check counts its operands by owner.
#                     Model.classify holds the one-pass classification that main() printed.
#                     A one-function dump gives the same output.

import bisect
import collections
import heapq
import itertools
import json
import re
import sys

DSZ = {"float32": 4, "int32": 4, "uint32": 4, "bfloat16": 2, "float16": 2, "int16": 2,
       "uint16": 2, "int8": 1, "uint8": 1, "float8e4": 1, "float8e5": 1, "bool": 1}


def memory_locations(f):
    """{name: memory location} of the allocations of function `f`. `set`: the name of the
    location's allocation (a memory location set), which a Call's `arguments` name."""
    memlocs = {}
    for a in f["allocations"]:
        for m in a.get("memorylocations", []):
            memlocs[m["name"]] = dict(
                type=m["type"], addr=m["addr"], dims=m["dims"], dtype=a["dtype"],
                kind=a["kind"], tensor_name=a.get("debug", {}).get("tensor_name"),
                shape=a.get("debug", {}).get("shape"), bank=m.get("bank"),
                base=m.get("base"), table_entry_id=m.get("table_entry_id"),
                tensor_id=m.get("tensor_id"), runtime_reserved=m.get("runtime_reserved"),
                set=a.get("name"))
    return memlocs


#: The kinds of DRAM location in the local arena of a linked dump: a function's own, main's.
LOCAL_ARENA_KINDS = ("Internal", "InternalInterface")


class Memlocs(dict):
    """The memory locations of `function` by name. `owners` holds every function's locations
    by function name (module docstring, "Linked dumps"); `linked`: the dump has more than one
    function. `binding`: in a call of the function, {the set name of a location the call
    binds: (main's location or None, its name, main, True)} (Program.binding). `params`: {a
    function main calls: (the binding of its call when main calls it once, else None; the
    number of its calls; the set names its calls bind)} (Program.params)."""

    def __init__(self, f, owners, binding=None, params=None):
        super().__init__(memory_locations(f))
        self.function, self.owners, self.linked = f.get("name"), owners, len(owners) > 1
        self.binding, self.params = binding or {}, params or {}

    def locate(self, op, ref, strip_set=False):
        """(location or None, its name, the function that owns it, whether this call binds it)
        of `ref`, which operand `op` names: a location of the function `memref_owner_func`
        names (main, this function or another one), else of this function. A location of this
        function that the call binds is main's location it binds it to; a parameter of another
        function is main's location that function's one call binds it to, and Undecided when
        main calls that function more than once. With `strip_set`, a `ref` no location has is
        read as a set name (`<location>_set`)."""
        owner = op.get("memref_owner_func") or self.function
        table = self if owner == self.function else self.owners.get(owner, {})
        if strip_set and ref not in table and ref.endswith("_set"):
            ref = ref[:-len("_set")]
        m = table.get(ref)
        if m is None:
            return m, ref, owner, False
        if owner == self.function:
            if m["set"] in self.binding:
                return self.binding[m["set"]]
        elif owner in self.params and m["set"] in self.params[owner][2]:
            binding, calls = self.params[owner][:2]
            if binding is None:
                raise Undecided(f"operand on {self.name(ref, owner)}, a parameter of {owner}, "
                                f"which main calls {calls} times: which call's location it "
                                f"is is not known")
            theirs, name, caller, _ = binding[m["set"]]
            return theirs, name, caller, False
        return m, ref, owner, False

    def name(self, ref, owner):
        """`ref` of `owner` for a message: the bare name for this function's own location."""
        return repr(ref) if owner == self.function else f"{ref!r} of {owner}"

    def dram_space(self, m, ref, owner):
        """The space of DRAM location `m` (named `ref`, owned by `owner`): the internal arena,
        else the tensor; in a linked dump, the local arena or the tensor of its owner."""
        if not self.linked:
            return "DRAM:" + ("INTERNAL" if m["kind"] == "Internal" else ref)
        return "DRAM:INTERNAL" if m["kind"] in LOCAL_ARENA_KINDS else f"DRAM:{owner}:{ref}"


def load(path, function=0, d=None, binding=None, params=None):
    """(the dump, its function number `function`, that function's Memlocs, its instructions in
    program order, every DMA block of the dump by name). `d`: the dump, already read;
    `binding`, `params`: Memlocs's, for a call of the function."""
    if d is None:
        d = json.load(open(path))
    f = d["functions"][function]
    memlocs = Memlocs(f, {g.get("name"): memory_locations(g) for g in d["functions"]}, binding,
                      params)
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
#: The confirmed UNSYNC pairs of the one-pass VERDICT line, by space (module docstring).
CONFIRMED_BY_SPACE = ("SB-RAW", "SB-WAW", "SB-WAR", "PSUM", "DRAM")
#: Parses a one-pass VERDICT line (module docstring): `word` and one group per count.
VERDICT_LINE = re.compile(
    r"^VERDICT: (?P<word>FINDINGS|UNDECIDED|CLEAN) "
    r"\(confirmed (?P<confirmed>\d+): SB-RAW (?P<sb_raw>\d+) SB-WAW (?P<sb_waw>\d+) "
    r"SB-WAR (?P<sb_war>\d+) PSUM (?P<psum>\d+) DRAM (?P<dram>\d+); "
    r"not confirmed: same-value-TensorSave (?P<same_value_tensorsave>\d+) "
    r"runtime-address (?P<runtime_address>\d+) edges-unknown (?P<edges_unknown>\d+); "
    r"undecided (?P<undecided>\d+): runtime-SB-PSUM-operand (?P<runtime_sb_psum_operand>\d+) "
    r"other-operand (?P<other_operand>\d+) cycle (?P<cycle>\d+) "
    r"ungrouped-reset (?P<ungrouped_reset>\d+); waits-not-OK (?P<waits_not_ok>\d+)\)$")


def verdict_line(word, n):
    """The one-pass VERDICT line (module docstring) for `word` and the counts `n`
    (`Model.one_pass_verdict`); `VERDICT_LINE` parses it."""
    return (f"VERDICT: {word} (confirmed {n['confirmed']}: SB-RAW {n['SB-RAW']} "
            f"SB-WAW {n['SB-WAW']} SB-WAR {n['SB-WAR']} PSUM {n['PSUM']} DRAM {n['DRAM']}; "
            f"not confirmed: same-value-TensorSave {n['same-value-TensorSave']} "
            f"runtime-address {n['runtime-address']} edges-unknown {n['edges-unknown']}; "
            f"undecided {n['undecided']}: runtime-SB-PSUM-operand "
            f"{n['runtime-SB-PSUM-operand']} other-operand {n['other-operand']} "
            f"cycle {n['cycle']} ungrouped-reset {n['ungrouped-reset']}; "
            f"waits-not-OK {n['waits-not-OK']})")


class Undecided(Exception):
    """An operand the check cannot place in memory."""


class CycleError(Exception):
    """The happens-before graph has a cycle (the message names one), so it has no closure."""


class _Reach:
    """The nodes one node happens before, for `in` tests (Model.reach): bit x of the node's
    closure row, the node itself left out (a closure row holds its own bit)."""

    __slots__ = ("index", "node", "row")

    def __init__(self, node, row, index):
        self.node, self.row, self.index = node, row, index

    def __contains__(self, x):
        k = self.index.get(x)
        if k is None or x == self.node or k // 8 >= len(self.row):
            return False
        return bool(self.row[k // 8] >> (k % 8) & 1)


def runtime_footprint(op, memlocs):
    """A `register_ap` operand: its address is a register the program sets. DRAM: the whole
    tensor (conservative: it overlaps every access to that tensor). SBUF/PSUM: Undecided."""
    m, ref, owner, bound = memlocs.locate(op, op.get("memref") or op.get("memsetref") or "",
                                          strip_set=True)
    if m is None:
        raise Undecided(f"runtime-address operand on {memlocs.name(ref, owner)}: "
                        f"no memory location")
    if m["type"] != "DRAM":
        raise Undecided(f"runtime-address {m['type']} operand on {ref}")
    size = 1
    for n in m["dims"]:
        size *= n
    return dict(space=memlocs.dram_space(m, ref, owner), parts=None, lo=m["addr"],
                hi=m["addr"] + size, cross=False, ivs=None, memref=ref, tensor=m["tensor_name"],
                ap=op.get("ap"), off=None, runtime=True, owner=owner, bound=bound)


def whole_location(op, memlocs):
    """An SBUF/PSUM `register_ap` operand as its whole memory location (every partition from
    the location's `base`, every byte of its row), or None when it names no such location.
    The loop-carried class uses it; it assumes the runtime address stays inside."""
    if op.get("kind") != "register_ap":
        return None
    try:
        m, ref, owner, bound = memlocs.locate(op, op.get("memref") or op.get("memsetref") or "",
                                              strip_set=True)
    except Undecided:
        return None
    if m is None or m["type"] not in ("SB", "PSUM"):
        return None
    pbase = m.get("base") or 0
    lo, hi = m["addr"], m["addr"] + m["dims"][1]
    space = m["type"] if m["type"] == "SB" else f"PSUM.bank{m['bank']}"
    return dict(space=space, parts=frozenset(range(pbase, pbase + m["dims"][0])), lo=lo, hi=hi,
                cross=False, ivs=[(lo, hi)], memref=ref, tensor=m["tensor_name"],
                ap=op.get("ap"), off=None, runtime=True, owner=owner, bound=bound)


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
    m, ref, owner, bound = memlocs.locate(op, op["memref"])
    if m is None:
        raise Undecided(f"operand on {memlocs.name(ref, owner)}: no memory location")
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
                    memref=ref, tensor=m["tensor_name"], ap=ap, off=off, owner=owner,
                    bound=bound)
    maxoff = sum((n - 1) * s for s, n in ap)
    space = memlocs.dram_space(m, ref, owner)
    lo = m["addr"] + off * dsz
    hi = m["addr"] + (off + maxoff + 1) * dsz
    ivs = expand(ap, m["addr"] // dsz + off)
    ivs = None if ivs is None else [(a * dsz, b * dsz) for a, b in ivs]
    return dict(space=space, parts=None, lo=lo, hi=hi, cross=False, ivs=ivs, memref=ref,
                tensor=m["tensor_name"], ap=ap, off=off, owner=owner, bound=bound)


def overlap(a, b):
    if a["space"] != b["space"]:
        return False
    if a["lo"] >= b["hi"] or b["lo"] >= a["hi"]:
        return False
    if a["parts"] is not None and not (a["parts"] & b["parts"]):
        return False
    return ivs_overlap(a.get("ivs"), b.get("ivs"))


#: The pair of an earlier node's footprint and a later node's, by their roles (read, write):
#: (group, rank, kind). `Model.conflicts` lists one producer/consumer pair's footprint pairs
#: in the order of these keys: the producer's writes first, each against the consumer's reads
#: (RAW) and then its writes (WAW); then the producer's reads against the consumer's writes
#: (WAR). Two reads are not a pair.
PAIR_ROLES = {("w", "r"): (0, 0, "RAW"), ("w", "w"): (0, 1, "WAW"), ("r", "w"): (1, 0, "WAR")}


def fp_str(fp):
    if fp is None:
        return "-"
    if fp["parts"] is not None:
        ps = sorted(fp["parts"])
        pstr = f"p{ps[0]}" if len(ps) == 1 else (f"p{ps[0]}..{ps[-1]}" if ps == list(range(ps[0], ps[-1] + 1)) else f"p{ps[0]}..{ps[-1]}/step")
        return f"{fp['space']}[{pstr}][{fp['lo']}:{fp['hi']}) {fp['tensor']}"
    return f"{fp['space']}[{fp['lo']}:{fp['hi']}) {fp['tensor']}"


class Model:
    """The check of one function of a dump (function number `function`; `d`: the dump, already
    read). `call`: a call of the function (Call; module docstring, "Linked dumps"), with
    `binding` its Memlocs binding; `params`: Memlocs's, the parameters of the functions main
    calls."""

    def __init__(self, path, function=0, d=None, call=None, binding=None, params=None):
        self.d, self.f, self.memlocs, self.ins, self.blocks = load(path, function, d, binding,
                                                                   params)
        self.function = self.f.get("name")
        self.call = call
        self.build()

    @classmethod
    def graph_of(cls, edges, qedges, acc):
        """A Model of a happens-before graph (`edges`, `qedges`) and access nodes `acc` only,
        not of a function: `ordered`, `path`, `overlapping_pairs` and `unconfirmed_because`
        work on it (Program's pairs of two calls)."""
        graph = cls.__new__(cls)
        graph.edges, graph.qedges, graph.acc, graph._reach = edges, qedges, acc, {}
        return graph

    def build(self):
        ins, blocks, memlocs = self.ins, self.blocks, self.memlocs
        # access nodes: (node_id, issue_node, done_node, engine, order, reads[fp], writes[fp], meta)
        self.acc = []
        self.edges = collections.defaultdict(set)
        self.pos = {}
        self.info = {}
        self.undecided = []    # operands / edges the check cannot model: the dump is UNDECIDED
        self.undecided_class = collections.Counter()   # their causes (module docstring)
        self.loop_undecided = []   # (block or None for the dump, why): the loop-carried class's
        # operands with memory by owner: `memref_owner_func` names another function (by
        # owner), or the call binds the location (to main's), or else the function's own
        self.foreign_operands = collections.Counter()   # owner -> operands on its locations
        self.foreign_names = collections.defaultdict(set)   # owner -> the location names
        self.bound_operands = 0          # operands on a location the call binds (main's)
        self.bound_names = set()         # main's names of those locations
        self.own_operands = 0            # the others: on the function's own locations
        self.own_names = set()
        self.alias = {}                  # DMA block a wait names -> the block this call starts
        started = [self._dma_blocks(i) for i in ins]   # the DMA blocks each instruction starts
        # Nodes are positions: an instruction (or DMA block) name that repeats gets `name@k`.
        icount = collections.Counter(i["name"] for i in ins)
        bcount = collections.Counter(b for blocks in started for b in blocks)
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
                owner = x.get("memref_owner_func")
                foreign = bool(owner) and owner != self.function
                if foreign:
                    self.foreign_operands[owner] += 1
                    self.foreign_names[owner].add(x.get("memref") or x.get("memsetref"))
                try:
                    f = footprint(x, memlocs)
                    if f and f["bound"]:
                        self.bound_operands += 1
                        self.bound_names.add(f["memref"])
                    elif f and not foreign:
                        self.own_operands += 1
                        self.own_names.add(f["memref"])
                except Undecided as why:
                    if not foreign:
                        self.own_operands += 1
                        self.own_names.add(x.get("memref") or x.get("memsetref"))
                    dbg = i.get("debug") or {}
                    text = f"{node} {i['opcode']}@{i['engine']} L{dbg.get('lineno')}: {why}"
                    self.undecided.append(text)
                    w = whole_location(x, memlocs)
                    self.undecided_class["other-operand" if w is None
                                         else "runtime-SB-PSUM-operand"] += 1
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
        self.queue_order = []  # (instruction index, queue, completion node) of each DMA, in order
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
                self.queue_order.append((k, q, done))
            elif op == "DMATrigger":
                for bname in started[k]:
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
                    self.queue_order.append((k, q, bn))
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
            self.undecided_class["cycle"] += 1
        # reachability: {with_queue: the relation's bit-set closure}, built on first use (reach)
        self._reach = {}
        self._loop_pairs = None
        self._body_updates = {}

    def _dma_blocks(self, i):
        """The DMA blocks instruction `i` starts: its `dma_blocks`. In call k of the function's
        n calls, a list of n blocks starts its block k, and a wait on any of them waits on
        that one (`alias`); a list of another length than 1 or n starts every block, and the
        check of the call is UNDECIDED (module docstring, "Linked dumps")."""
        blocks = i.get("dma_blocks", [])
        if self.call is None or len(blocks) <= 1:
            return blocks
        if len(blocks) == self.call.calls:
            mine = blocks[self.call.number]
            self.alias.update((b, mine) for b in blocks)
            return [mine]
        dbg = i.get("debug") or {}
        self.undecided.append(
            f"{i['name']} {i['opcode']}@{i['engine']} L{dbg.get('lineno')}: names "
            f"{len(blocks)} DMA blocks for the {self.call.calls} calls of {self.function}: "
            f"which one a call starts is not known")
        self.undecided_class["other-operand"] += 1
        return blocks

    def junction(self, engines, barrier, queues):
        """How the graph of this call joins the graph of the calls before it: (targets, exits),
        by the program-order rules of `build`. `engines`, `queues`: the engines and queues with
        a last node before the call; `barrier`: an all-engine instruction is before it. A slot
        is ("engine", e), ("barrier",) or ("queue", q). `targets`: {slot: the nodes of this
        call that get an edge from the slot's last node before the call} (a queue slot's edges
        are qedges); `exits`: {slot: the slot's last node in this call}."""
        targets, last, bar, last_q = collections.defaultdict(set), {}, None, {}
        queue_at = collections.defaultdict(list)
        for k, q, node in self.queue_order:
            queue_at[k].append((q, node))
        for k, (name, i) in enumerate(zip(self.node, self.ins)):
            eng = i["engine"]
            if eng == "ALL":
                for e in engines | set(last):
                    if e not in last:
                        targets[("engine", e)].add(name)
                    last[e] = name
                if bar is None and barrier:
                    targets[("barrier",)].add(name)
                bar = name
            elif eng != "Unassigned":
                if eng not in last and bar is None:
                    if eng in engines:
                        targets[("engine", eng)].add(name)
                    elif barrier:
                        targets[("barrier",)].add(name)
                last[eng] = name
            for q, node in queue_at[k]:
                if q not in last_q and q in queues:
                    targets[("queue", q)].add(node)
                last_q[q] = node
        exits = {("engine", e): n for e, n in last.items()}
        if bar is not None:
            exits[("barrier",)] = bar
        exits.update((("queue", q), n) for q, n in last_q.items())
        return targets, exits

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
                    self.undecided_class["ungrouped-reset"] += 1
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
        iteration's instance, which the loop-carried class does not model. In a call, a wait
        on a DMA block waits on the block the call starts (`alias`)."""
        inst = self.ids_of.get(self.alias.get(w["from"], w["from"]))
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

    def _graph(self, with_queue):
        """({node: its successors}, every node) of one relation: `edges`, plus `qedges` when
        `with_queue` (same-queue DMAs complete in order)."""
        succ = collections.defaultdict(set)
        for e in (self.edges, self.qedges) if with_queue else (self.edges,):
            for n, after in e.items():
                succ[n] |= after
        return succ, set(succ).union(*succ.values()) if succ else set()

    @staticmethod
    def _topological(succ, nodes):
        """Kahn's order of the graph: every node after its predecessors. The nodes on or
        behind a cycle are not in it."""
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
        return order

    def cycle_nodes(self):
        """Number of nodes on or behind a cycle of the happens-before graph (0: acyclic)."""
        succ, nodes = self._graph(True)
        return len(nodes) - len(self._topological(succ, nodes))

    def one_cycle(self):
        """One cycle of the happens-before graph, its nodes in order ([]: acyclic). Each node
        Kahn's order leaves out has a predecessor it also leaves out, so a walk back through
        them repeats a node; the walk between the two visits is a cycle."""
        succ, nodes = self._graph(True)
        left = nodes - set(self._topological(succ, nodes))
        if not left:
            return []
        pred = {}
        for n in sorted(left):
            for m in sorted(succ.get(n, ())):
                if m in left:
                    pred.setdefault(m, n)
        walk, seen = [min(left)], set()
        while walk[-1] not in seen:
            seen.add(walk[-1])
            walk.append(pred[walk[-1]])
        return list(reversed(walk[walk.index(walk[-1]):]))

    def succ(self, n, with_queue):
        out = set(self.edges.get(n, ()))
        if with_queue:
            out |= self.qedges.get(n, set())
        return out

    def _closure(self, with_queue):
        """(index, rows) of one relation: a node's index is its place in reverse topological
        order, so its descendants have smaller indices; its row is its own bit OR its
        successors' rows, built sinks first, as little-endian bytes (index // 8 + 1 of them).
        Memory: about N^2 / 16 bytes for N nodes. A cycle raises CycleError."""
        succ, nodes = self._graph(with_queue)
        order = self._topological(succ, nodes)
        if len(order) != len(nodes):
            raise CycleError(" -> ".join(self.one_cycle()))
        order.reverse()
        index = {n: k for k, n in enumerate(order)}
        closure = {}
        for n in order:
            row = 1 << index[n]
            for m in succ.get(n, ()):
                row |= closure[m]
            closure[n] = row
        return index, {n: closure.pop(n).to_bytes(index[n] // 8 + 1, "little") for n in order}

    def reach(self, a, with_queue=False):
        """The nodes `a` happens before (not `a` itself), for `in` tests only (`ordered`):
        a view of the relation's closure (`_closure`). Raises CycleError on a cycle."""
        if with_queue not in self._reach:
            self._reach[with_queue] = self._closure(with_queue)
        index, rows = self._reach[with_queue]
        row = rows.get(a)
        return _Reach(a, row, index) if row is not None else frozenset()

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
        or "same-value-TensorSave"; None when it is confirmed (module docstring, "The one-pass
        verdict")."""
        if fp.get("runtime") or fc.get("runtime"):
            return "runtime-address"
        if (P["i"]["opcode"] == C["i"]["opcode"] == "TensorSave" and fp["memref"] == fc["memref"]
                and ENGINE_SUFFIX.sub("", P["i"]["name"]) == ENGINE_SUFFIX.sub("", C["i"]["name"])):
            return "same-value-TensorSave"
        return None

    def one_pass_verdict(self, unsync, waits_not_ok):
        """(word, counts) of the one-pass VERDICT line (module docstring; `verdict_line` prints
        it): `unsync` the UNSYNC footprint pairs (kind, P, C, fp, fc), `waits_not_ok` the
        number of waits whose arithmetic is not OK."""
        edges_unknown = bool(self.undecided_class["cycle"]
                             or self.undecided_class["ungrouped-reset"])
        counts = collections.Counter()
        for kind, P, C, fp, fc in unsync:
            because = self.unconfirmed_because(P, C, fp, fc)
            if because is None and edges_unknown:
                because = "edges-unknown"
            if because is not None:
                counts[because] += 1
            elif fp["space"] == "SB":
                counts[f"SB-{kind}"] += 1
            else:
                counts["PSUM" if fp["space"].startswith("PSUM") else "DRAM"] += 1
        counts["confirmed"] = sum(counts[part] for part in CONFIRMED_BY_SPACE)
        counts.update(self.undecided_class)
        counts["undecided"] = len(self.undecided)
        counts["waits-not-OK"] = waits_not_ok
        if counts["confirmed"] or (waits_not_ok and not edges_unknown):
            word = "FINDINGS"
        elif self.undecided or counts["runtime-address"]:
            word = "UNDECIDED"
        else:
            word = "CLEAN"
        return word, counts

    def classify(self, lo=None, hi=None, keep_all=False):
        """The one-pass check of the function (`Classified`; main prints it). The pairs are
        the overlapping pairs (with `lo`, `hi`: those with P or C on a source line in [lo, hi]),
        classified as the sweep finds them. Only those main prints are kept: the pairs not
        ordered, or every pair with `keep_all`, in the order of `conflicts`. On a cycle no pair
        is classified (`reach` raises CycleError)."""
        waits = self.check_sem_arith()
        cycle = self.one_cycle() if self.undecided_class["cycle"] else []
        n_pairs = n_ok = n_bad = n_q = 0
        kept = []
        for order, kind, P, C, fa, fb in self.overlapping_pairs(lo, hi):
            n_pairs += 1
            if cycle:
                continue
            ok = self.ordered(P, C)
            okq = ok or self.ordered(P, C, True)
            if ok:
                n_ok += 1
            elif okq:
                n_q += 1
            else:
                n_bad += 1
            if keep_all or not ok:
                kept.append((order, kind, P, C, fa, fb, ok, okq))
        kept.sort(key=lambda pair: pair[0])
        unsync = [(kind, P, C, fa, fb) for _, kind, P, C, fa, fb, _, okq in kept if not okq]
        bad = sum(1 for status in waits if status[0] != "OK")
        return Classified(waits, bad, cycle, n_pairs, n_ok, n_q, n_bad,
                          [pair[1:] for pair in kept], unsync, *self.one_pass_verdict(unsync, bad))

    def label(self, a):
        i = a["i"]
        dbg = i.get("debug") or {}
        return f"{a['name']} {i['opcode']}@{a['engine']} L{dbg.get('lineno')} {dbg.get('op_name')}"

    def overlapping_pairs(self, lo=None, hi=None):
        """Every overlapping footprint pair (key, kind, P, C, fp, fc) of two access nodes, P
        before C in program order and `fp` P's footprint, in no fixed order. Sorted by `key`,
        they are the list of `conflicts`. With `lo`, `hi`: only the pairs with P or C on a
        source line in [lo, hi].

        Per space, the footprints are visited in order of `lo`. Each one meets only the open
        footprints of the other role or roles (PAIR_ROLES): those whose `hi` is above its `lo`,
        which is the bounding-box test of `overlap`. `overlap` then decides, with P's footprint
        first. The work is the sort plus one step per bounding-box pair, not one per pair of
        access nodes."""
        acc = self.acc
        in_range = None
        if lo is not None:
            in_range = [lo <= ((a["i"].get("debug") or {}).get("lineno") or 0) <= hi for a in acc]
        by_space = collections.defaultdict(list)
        for x, a in enumerate(acc):
            for role, fps in (("r", a["reads"]), ("w", a["writes"])):
                for index, f in enumerate(fps):
                    by_space[f["space"]].append((f["lo"], x, role, index, f))
        for items in by_space.values():
            items.sort(key=lambda item: (item[0], item[1]))
            open_fps = {"r": {}, "w": {}}
            closing = []   # heap of (hi, sequence number, role) of the open footprints
            for seq, item in enumerate(items):
                f_lo, x, role, index, f = item
                while closing and closing[0][0] <= f_lo:
                    _, done, done_role = heapq.heappop(closing)
                    del open_fps[done_role][done]
                for other in ("w",) if role == "r" else ("r", "w"):
                    for _, ox, o_role, o_index, of in open_fps[other].values():
                        if ox == x:
                            continue
                        if ox < x:
                            px, p_role, p_index, pf, cx, c_role, c_index, cf = (
                                ox, o_role, o_index, of, x, role, index, f)
                        else:
                            px, p_role, p_index, pf, cx, c_role, c_index, cf = (
                                x, role, index, f, ox, o_role, o_index, of)
                        if lo is not None and not (in_range[px] or in_range[cx]):
                            continue
                        if overlap(pf, cf):
                            group, rank, kind = PAIR_ROLES[p_role, c_role]
                            yield ((px, cx, group, p_index, rank, c_index), kind, acc[px], acc[cx],
                                   pf, cf)
                open_fps[role][seq] = item
                heapq.heappush(closing, (f["hi"], seq, role))

    def conflicts(self, lo=None, hi=None):
        """The overlapping footprint pairs (kind, P, C, fp, fc) of `overlapping_pairs`, ordered by
        P, then C, then the footprints (PAIR_ROLES)."""
        pairs = sorted(self.overlapping_pairs(lo, hi), key=lambda pair: pair[0])
        return [pair[1:] for pair in pairs]

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

Classified = collections.namedtuple(
    "Classified", "waits waits_not_ok cycle pairs ordered queue_only unsync_count kept unsync "
                  "word counts")
Classified.__doc__ = """The one-pass check of one function (`Model.classify`).

`waits`: the rows of `check_sem_arith`; `waits_not_ok`: how many are not OK. `cycle`: one cycle
of the graph, [] when it has none. `pairs`: the number of overlapping pairs; `ordered`,
`queue_only`, `unsync_count`: how many are ordered by engine order and semaphores, ordered only
if same-queue DMAs complete in order, not ordered. `kept`: (kind, P, C, fp, fc, ok, okq) of the
pairs main prints. `unsync`: (kind, P, C, fp, fc) of the UNSYNC pairs. `word`, `counts`: the
VERDICT line's (`Model.one_pass_verdict`)."""

Call = collections.namedtuple("Call", "name function number calls arguments position")
Call.__doc__ = """One of main's Calls (module docstring, "Linked dumps"): `name`, the instruction's
name; `function`, its `target`; `number`, its place (from 0) among the `calls` of the function;
`arguments`, {a set name of the function: a set name of main}; `position`, its place in main's
call order."""

Check = collections.namedtuple(
    "Check", "label function call instructions word loop_word pairs ordered queue_only "
             "unsync_count owner_field owner_field_names bound bound_names own own_names "
             "siblings cycle")
Check.__doc__ = """The check of main, of one call (`call`) or of a function main does not call
(`Program.check`). `label`: the name its VERDICT lines carry, `<function> <call>` for a call.
`word`, `loop_word`: its one-pass and loop-carried verdicts; `pairs`, `ordered`, `queue_only`,
`unsync_count`: the counts of its one-pass pairs (Classified). Its operands with memory by
owner, and `*_names` the number of their locations: `owner_field`, on locations of main named
by `memref_owner_func`; `bound`, on locations the call binds to main's; `siblings`, ((another
function, operands, locations), ...) named by `memref_owner_func`; `own`, the others, on the
function's own locations. `cycle`: its graph has a cycle."""

CrossPair = collections.namedtuple(
    "CrossPair", "kind src_label dst_label src_function dst_function tag src dst src_fp dst_fp "
                 "because")
CrossPair.__doc__ = """One cross-function pair (module docstring, "Cross-function pairs").

`src`, an access of the check `src_label` (of `src_function`) with footprint `src_fp`, is
earlier in the program than `dst`, an access of the check `dst_label` with footprint `dst_fp`.
`tag`: OK, QUEUE-ORDER-ONLY, UNSYNC or UNCHECKED. `because`: why an UNSYNC pair is not
confirmed (`Model.unconfirmed_because`), else None."""

#: The tags of a cross-function pair, in the order the counts print.
CROSS_TAGS = ("OK", "QUEUE-ORDER-ONLY", "UNSYNC", "UNCHECKED")


class JoinedOrder:
    """The order across calls (module docstring, "Cross-function pairs"), read without the
    program's graph. A slot (`Model.junction`) holds the last node of an engine, of the
    all-engine instructions or of a queue. For each call and relation (`with_queue` False,
    True), `add` keeps, as bits of the slots: for an access, the slots after its call whose
    last node it reaches (`out`) and the slots before its call whose last node reaches it
    (`into`); for the call, which slots after it each slot before it reaches (`transfer`).
    `switched`: the queues that main switches, which carry no order across calls."""

    def __init__(self, switched=()):
        self.switched = frozenset(switched)
        self.engines, self.barrier, self.queues = set(), False, set()
        self.bit = {}          # slot -> its bit
        self.transfer = []     # per call: {with_queue: {bit of a slot before it: bits after it}}
        self._memo = {}

    def add(self, M, accs):
        """Join the next call, Model `M`; `accs`: its accesses that cross-function pairs take,
        which get the masks and `call`, the call's place here."""
        targets, exits = M.junction(self.engines, self.barrier, self.queues)
        exits = {s: n for s, n in exits.items() if s[0] != "queue" or s[1] not in self.switched}
        before = list(self.bit)
        for slot in exits:
            self.bit.setdefault(slot, 1 << len(self.bit))

        def mask(slots, test):
            out = 0
            for s in slots:
                if test(s):
                    out |= self.bit[s]
            return out

        transfer = {}
        for with_queue in (False, True):
            def reaches(x, y):
                return x == y or y in M.reach(x, with_queue)

            sources = {s: targets.get(s, ()) if with_queue or s[0] != "queue" else ()
                       for s in before}
            transfer[with_queue] = {
                self.bit[s]: (0 if s in exits else self.bit[s]) | mask(
                    exits, lambda t: any(reaches(n, exits[t]) for n in sources[s]))
                for s in before}
            for a in accs:
                a[("out", with_queue)] = mask(exits, lambda t: reaches(a["done"], exits[t]))
                a[("into", with_queue)] = mask(
                    before, lambda s: any(reaches(n, a["issue"]) for n in sources[s]))
        for a in accs:
            a["call"] = len(self.transfer)
        self.transfer.append(transfer)
        self.engines |= {s[1] for s in exits if s[0] == "engine"}
        self.barrier = self.barrier or ("barrier",) in exits
        self.queues |= {s[1] for s in exits if s[0] == "queue"}

    def ordered(self, p, c, with_queue=False):
        """Access `p`'s completion happens-before access `c`'s start, `c` in a later call."""
        key = (p["call"], c["call"], p[("out", with_queue)], with_queue)
        v = self._memo.get(key)
        if v is None:
            v = p[("out", with_queue)]
            for k in range(p["call"] + 1, c["call"]):
                t, after = self.transfer[k][with_queue], 0
                while v:
                    low = v & -v
                    after |= t[low]
                    v ^= low
                v = after
            self._memo[key] = v
        return bool(v & c[("into", with_queue)])


class Program:
    """The check of a dump of several functions (module docstring, "Linked dumps"): main's
    calls; then (`check`) main, each call, each function main does not call, and the
    cross-function pairs. `d`: the dump, already read; `lo`, `hi`: the source lines of
    `--lines`; `keep_all`: keep every pair of a check for printing (`Model.classify`)."""

    #: The function whose calls run the others.
    CALLER = "main"

    def __init__(self, path, d=None, lo=None, hi=None, keep_all=False):
        self.path, self.lo, self.hi, self.keep_all = path, lo, hi, keep_all
        self.d = json.load(open(path)) if d is None else d
        self.names = [f.get("name") for f in self.d["functions"]]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"{path}: function names repeat: {self.names}")
        self.calls, self.order_unknown = self._calls()
        caller = (self.d["functions"][self.names.index(self.CALLER)]
                  if self.CALLER in self.names else dict(allocations=[], blocks=[]))
        self.caller_locations = memory_locations(caller)
        self._caller_sets = collections.defaultdict(list)
        for name, m in self.caller_locations.items():
            self._caller_sets[m["set"]].append(name)
        #: The queues main switches (SwitchQueueInstance): they carry no order across calls.
        self.switched = {i.get("queue") for b in caller["blocks"] for i in b["instructions"]
                         if i.get("opcode") == "SwitchQueueInstance"}
        #: Memlocs's `params`: {a function main calls: (the binding of its call when main calls
        #: it once, else None; the number of its calls; the set names its calls bind)}.
        self.params = {}
        for c in self.calls or ():
            one = self.binding(c) if c.calls == 1 else None
            sets = self.params.get(c.function, (None, 0, frozenset()))[2]
            self.params[c.function] = (one, c.calls, sets | frozenset(c.arguments))
        #: {function: {owner: its operands that name the owner's locations}}, by
        #: `memref_owner_func` (an operand without one is the function's own), over the
        #: function's instructions and every DMA block its DMATriggers name, once per name in a
        #: `dma_blocks` list; and the (owner, location) pairs that a function other than the
        #: owner and main names (module docstring, "Cross-function pairs").
        self.census, self.cross_owned = self._census()
        self._arena = self._arena_bytes()
        self.checks, self.models, self.verdicts = [], {}, {}
        #: Why the cross-function pairs are UNCHECKED (None: they are classified).
        self.unchecked = self.order_unknown
        self._accesses, self._join, self._cross = [], JoinedOrder(self.switched), None

    @property
    def order(self):
        """The functions main calls, in call order (None without a call order)."""
        return None if self.calls is None else [c.function for c in self.calls]

    def _calls(self):
        """(main's Calls in order, None), or (None, why the dump gives no call order)."""
        if self.CALLER not in self.names:
            return None, f"the dump has no function named {self.CALLER}"
        caller = self.d["functions"][self.names.index(self.CALLER)]
        found = []
        for i in (i for b in caller["blocks"] for i in b["instructions"]):
            if i.get("opcode") != "Call":
                continue
            target = i.get("target")
            if target not in self.names or target == self.CALLER:
                return None, (f"{self.CALLER}'s {i.get('name')} calls {target!r}, which is "
                              f"not a function of the dump")
            found.append((i.get("name"), target, i.get("arguments") or {}))
        if not found:
            return None, f"{self.CALLER} has no Call"
        if len({name for name, _, _ in found}) != len(found):
            return None, f"the names of {self.CALLER}'s Calls repeat"
        calls, number = collections.Counter(t for _, t, _ in found), collections.Counter()
        out = []
        for position, (name, target, arguments) in enumerate(found):
            out.append(Call(name, target, number[target], calls[target], arguments, position))
            number[target] += 1
        return out, None

    def binding(self, call):
        """The Memlocs binding of `call`: {each set name of the function in the call's
        arguments: (main's location in the set it binds it to, else None, its name, main,
        True)}."""
        out = {}
        for own, theirs in call.arguments.items():
            names = self._caller_sets.get(theirs, [])
            if len(names) == 1:
                out[own] = (self.caller_locations[names[0]], names[0], self.CALLER, True)
            else:
                ref = theirs[:-len("_set")] if theirs.endswith("_set") else theirs
                out[own] = (None, ref, self.CALLER, True)
        return out

    def _census(self):
        """(`census`, `cross_owned`): see __init__."""
        descs = {}
        for q in self.d.get("queues", []):
            for blk in q["blocks"]:
                for b in blk["instructions"]:
                    descs[b["name"]] = [x for sub in b.get("blocks", [])
                                        for x in sub["instructions"]]
        self._tables = {f.get("name"): memory_locations(f) for f in self.d["functions"]}
        census, named = {}, set()
        for f in self.d["functions"]:
            name, count = f.get("name"), collections.Counter()
            for i in (i for b in f["blocks"] for i in b["instructions"]):
                for x in [i] + [x for bn in i.get("dma_blocks") or () for x in descs.get(bn, ())]:
                    for op in (x.get("ins") or []) + (x.get("outs") or []):
                        ref = op.get("memref") or op.get("memsetref")
                        if not ref:
                            continue
                        owner = op.get("memref_owner_func") or name
                        count[owner] += 1
                        if owner not in (name, self.CALLER):
                            table = self._tables.get(owner, {})
                            if ref not in table and ref.endswith("_set"):
                                ref = ref[:-len("_set")]
                            named.add((owner, ref))
            census[name] = count
        return census, named

    def _arena_bytes(self):
        """(starts, ends) of the byte ranges, merged and in order, of the local arena that
        cross-function pairs are on: main's locations there and the locations there of
        `cross_owned`."""
        spans = []
        for m in itertools.chain(
                self.caller_locations.values(),
                (self._tables.get(owner, {}).get(ref) for owner, ref in self.cross_owned)):
            if m is not None and m["type"] == "DRAM" and m["kind"] in LOCAL_ARENA_KINDS:
                size = 1
                for n in m["dims"]:
                    size *= n
                spans.append((m["addr"], m["addr"] + size))
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [lo for lo, _ in merged], [hi for _, hi in merged]

    def in_scope(self, check, fp):
        """Footprint `fp` of an access of `check` is on memory that cross-function pairs are on
        (module docstring, "Cross-function pairs"): a location of another function (main's,
        bound or named, or a sibling's), a location of its own that another function names,
        or bytes of the local arena (`_arena_bytes`)."""
        if fp["owner"] != check.function or (fp["owner"], fp["memref"]) in self.cross_owned:
            return True
        if fp["space"] != "DRAM:INTERNAL":
            return False
        starts, ends = self._arena
        k = bisect.bisect_right(starts, fp["lo"]) - 1
        if k >= 0 and ends[k] > fp["lo"]:
            return True
        return k + 1 < len(starts) and starts[k + 1] < fp["hi"]

    def scope(self):
        """The memory of `in_scope`, for the output."""
        return ("memory of more than one function (main's, a location another function names, "
                "their local arena bytes)")

    def check(self, on_check=None, keep_models=False):
        """Check main, each call in call order (without a call order: each function, its
        parameters unbound) and each function main does not call; keep the accesses that
        cross-function pairs take. `on_check(check, model, classified)` runs after each check
        (main() prints it there). A check's Model is dropped after it, so the memory is that
        of the largest check; `keep_models` keeps each in `models` by label. Returns self."""
        jobs = [(self.CALLER, None)] if self.CALLER in self.names else []
        if self.calls is not None:
            called = {c.function for c in self.calls}
            jobs += [(c.function, c) for c in self.calls]
            jobs += [(n, None) for n in self.names if n != self.CALLER and n not in called]
        else:
            jobs += [(n, None) for n in self.names if n != self.CALLER]
        for function, call in jobs:
            M = Model(self.path, self.names.index(function), self.d, call,
                      None if call is None else self.binding(call), self.params)
            cl = M.classify(self.lo, self.hi, self.keep_all)
            label = function if call is None else f"{function} {call.name}"
            siblings = tuple((owner, n, len(M.foreign_names[owner]))
                             for owner, n in sorted(M.foreign_operands.items())
                             if owner != self.CALLER)
            check = Check(label, function, call, len(M.ins), cl.word, M.loop_carried_verdict(),
                          cl.pairs, cl.ordered, cl.queue_only, cl.unsync_count,
                          M.foreign_operands[self.CALLER], len(M.foreign_names[self.CALLER]),
                          M.bound_operands, len(M.bound_names), M.own_operands,
                          len(M.own_names), siblings, bool(cl.cycle))
            self.checks.append(check)
            self.verdicts[label] = (check.word, check.loop_word)
            if on_check is not None:
                on_check(check, M, cl)
            if keep_models:
                self.models[label] = M
            if function != self.CALLER and (call is not None or self.calls is None):
                self._keep_accesses(M, check)
        return self

    def _keep_accesses(self, M, check):
        """Keep the accesses of `check` on the memory of `in_scope`, and, in a call order, join
        the call (JoinedOrder). A call whose graph has a cycle leaves every pair UNCHECKED."""
        accs = []
        for a in M.acc:
            reads = [f for f in a["reads"] if self.in_scope(check, f)]
            writes = [f for f in a["writes"] if self.in_scope(check, f)]
            if reads or writes:
                accs.append(dict(name=a["name"], engine=a["engine"], i=a["i"], issue=a["issue"],
                                 done=a["done"], reads=reads, writes=writes, check=check))
        if check.cycle and not self.unchecked:
            self.unchecked = f"the happens-before graph of {check.label} has a cycle"
        if not self.unchecked:
            self._join.add(M, accs)
        self._accesses += accs

    def cross_pairs(self):
        """Every cross-function pair (CrossPair; with `lo`, `hi`: those with an access on a
        source line in [lo, hi]), in program order: the call order's, else, without one, the
        dump's order of the functions."""
        if self._cross is None:
            self._cross = self._cross_pairs()
        return self._cross

    def _cross_pairs(self):
        graph = Model.graph_of({}, {}, self._accesses)
        pairs = []
        for _, kind, P, C, fp, fc in sorted(graph.overlapping_pairs(self.lo, self.hi),
                                            key=lambda pair: pair[0]):
            if P["check"] is C["check"]:
                continue        # a pair of one check: the check's own
            tag, because = "UNCHECKED", None
            if not self.unchecked:
                ok = self._join.ordered(P, C)
                okq = ok or self._join.ordered(P, C, True)
                tag = "OK" if ok else ("QUEUE-ORDER-ONLY" if okq else "UNSYNC")
                if tag == "UNSYNC":
                    because = graph.unconfirmed_because(P, C, fp, fc)
            pairs.append(CrossPair(kind, P["check"].label, C["check"].label, P["check"].function,
                                   C["check"].function, tag, P, C, fp, fc, because))
        return pairs

    def cross_verdict(self):
        """FINDINGS when an UNSYNC cross-function pair is confirmed; else UNDECIDED when the
        pairs are UNCHECKED (also when there is none: an unbound parameter's memory is not
        known) or one goes through a runtime address; else CLEAN."""
        pairs = self.cross_pairs()
        if any(p.tag == "UNSYNC" and p.because is None for p in pairs):
            return "FINDINGS"
        if self.unchecked or any(p.because == "runtime-address" for p in pairs):
            return "UNDECIDED"
        return "CLEAN"

    def verdict(self):
        """FINDINGS when a VERDICT line of the dump is FINDINGS (each check's one-pass and
        loop-carried lines, the cross-function line), else UNDECIDED when one is, else CLEAN."""
        words = [w for pair in self.verdicts.values() for w in pair] + [self.cross_verdict()]
        for word in ("FINDINGS", "UNDECIDED"):
            if word in words:
                return word
        return "CLEAN"


def main():
    path = sys.argv[1]
    lo = hi = None
    verbose = "--verbose" in sys.argv
    show_all = "--all" in sys.argv
    pairs = "--pairs" in sys.argv
    for a in sys.argv[2:]:
        if a.startswith("--lines"):
            rng = a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            lo, hi = map(int, rng.split("-"))
    d = json.load(open(path))
    if len(d["functions"]) > 1:
        program = Program(path, d, lo, hi, show_all or pairs)
        print(f"# {path}")
        print_program(program, verbose, show_all, pairs)
        return
    M = Model(path, 0, d)
    print(f"# {path}")
    report(M, M.classify(lo, hi, show_all or pairs), verbose, show_all, pairs)


def report(M, cl, verbose=False, show_all=False, pairs=False, name=None):
    """Print the one-pass check of `M` (`cl`: its `Model.classify`) and its loop-carried class.
    `pairs`: also every kept pair with its ordering path (`--pairs`). `name`: the function's
    name for its VERDICT lines in a linked dump (module docstring, "Linked dumps")."""
    print(f"instructions={len(M.ins)} access-nodes={len(M.acc)} waits={len(M.waits)}")
    crosses = [(a['name'], fp_str(f)) for a in M.acc for f in a['reads'] + a['writes'] if f['cross']]
    if crosses:
        print("WARNING: inner AP crosses partition pitch (footprint approx):", crosses[:5])
    print("\n## semaphore wait arithmetic")
    for st, waiter, src, sid, v, exp in cl.waits:
        if st != "OK":
            print(f"  {st}: {waiter} waits sem{sid}>={v} for {src}; producer's count = {exp}")
    print(f"  {len(M.waits)} waits checked, {cl.waits_not_ok} not OK")
    print("\n## conflicting pairs (producer -> consumer) lacking happens-before")
    if cl.cycle:
        print(f"  not classified: the happens-before graph has a cycle, {' -> '.join(cl.cycle)}; "
              f"no pair's order can be read off it")
    seen = set()
    for kind, P, C, fa, fb, ok, okq in cl.kept:
        key = (P["name"], C["name"], kind)
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
    if pairs:
        print("\n## every overlapping pair in the selected line range, with its ordering path")
        seen2 = set()
        for kind, P, C, fa, fb, ok, okq in cl.kept:
            key = (P["name"], C["name"], kind)
            if key in seen2:
                continue
            seen2.add(key)
            tag = "OK" if ok else ("QUEUE-ORDER-ONLY" if okq else "UNSYNC")
            same_eng = P["engine"] == C["engine"] and "DMA" not in P["engine"]
            path = M.path(P["done"], C["issue"], not ok) if okq else None
            how = "same engine, in-order" if (same_eng and path and len(path) <= 2) else (" -> ".join(path) if path else "NONE")
            print(f"  [{tag}] {kind} {M.label(P)} {fp_str(fa)}")
            print(f"        -> {M.label(C)} {fp_str(fb)}")
            print(f"        via: {how}")
    if cl.cycle:
        print(f"\n  {cl.pairs} overlapping pairs, not classified (the graph has a cycle)")
    else:
        print(f"\n  {cl.pairs} overlapping pairs: {cl.ordered} ordered by engine order + semaphores, "
              f"{cl.queue_only} ordered only if same-queue DMAs complete in order, "
              f"{cl.unsync_count} UNSYNCHRONIZED")
    runtime = sum(1 for a in M.acc for f in a["reads"] + a["writes"] if f.get("runtime"))
    if runtime:
        print(f"  {runtime} DRAM footprints are runtime-address operands, taken as their whole tensor "
              f"(an UNSYNC pair through one needs its real address range argued)")
    if M.undecided:
        print(f"\n## UNDECIDED: {len(M.undecided)} operands or edges the check cannot model")
        for u in M.undecided[:20]:
            print(f"  {u}")
    line = verdict_line(cl.word, cl.counts)
    print(line if name is None else f"VERDICT ({name}):" + line[len("VERDICT:"):])
    print_loop_carried(M, show_all, name)


def print_program(program, verbose=False, show_all=False, pairs=False):
    """Print the check of a linked dump (module docstring, "Linked dumps"): the functions and
    main's calls; each check (`report`), as it ends; the functions' totals; the
    cross-function pairs; the program's verdict."""
    functions = program.d["functions"]
    print(f"functions={len(functions)}: " + ", ".join(
        f"{f.get('name')} ({sum(len(b['instructions']) for b in f['blocks'])} instructions)"
        for f in functions))
    if program.calls is None:
        print(f"calls: UNKNOWN ({program.order_unknown})")
    else:
        print(f"calls={len(program.calls)}, in {program.CALLER}'s order: "
              + ", ".join(f"{c.name} {c.function}" for c in program.calls))

    def on_check(check, M, cl):
        print(f"\n# {check_header(program, check)}")
        report(M, cl, verbose, show_all, pairs, check.label)

    program.check(on_check)
    print_functions(program)
    print_cross(program, verbose, show_all)
    print(f"VERDICT (program): {program.verdict()} ("
          + ", ".join(f"{label} {word}/{loop_word}"
                      for label, (word, loop_word) in program.verdicts.items())
          + f"; cross-function {program.cross_verdict()})")


def _locations(n):
    return f"{n} location" if n == 1 else f"{n} locations"


def owners_text(program, check):
    """A check's operands with memory by owner (Check): its own locations; main's, named by
    memref_owner_func or bound by the call's arguments; each other function's."""
    parts = [f"own {check.own} ({_locations(check.own_names)})"]
    if check.function != program.CALLER:
        parts.append(f"{program.CALLER} {check.owner_field} named by memref_owner_func "
                     f"({_locations(check.owner_field_names)})"
                     + (f", {check.bound} bound by the call's arguments "
                        f"({_locations(check.bound_names)})" if check.call is not None else ""))
    parts += [f"{owner} {n} named by memref_owner_func ({_locations(k)})"
              for owner, n, k in check.siblings]
    return "operands by owner: " + "; ".join(parts)


def check_header(program, check):
    """The heading of a check: what it checks, and its operands by owner."""
    c = check.call
    if c is not None:
        return (f"call {c.name}: {c.function}, call {c.number + 1} of {c.calls}; "
                f"{owners_text(program, check)}")
    called = "" if program.calls is None or check.function == program.CALLER else ", not called"
    return f"function {check.function}{called}; {owners_text(program, check)}"


def print_functions(program):
    """Each function's totals over its checks (one-pass pairs, verdicts, operands by owner),
    and its operands by `memref_owner_func` in the dump (Program.census)."""
    print("\n## functions: their checks' one-pass pairs, verdicts and operands by owner; the "
          "dump's operands by memref_owner_func (the function's instructions and each DMA "
          "block its DMATriggers name, once per name in a dma_blocks list)")
    for name in program.names:
        mine = [c for c in program.checks if c.function == name]
        if not mine:
            continue
        calls = sum(1 for c in mine if c.call is not None)
        words = collections.Counter(c.word for c in mine)
        loop_words = collections.Counter(c.loop_word for c in mine)
        siblings = collections.Counter()
        for c in mine:
            siblings.update({owner: n for owner, n, _ in c.siblings})
        census = program.census.get(name, collections.Counter())
        print(f"  {name}: {calls} {'call' if calls == 1 else 'calls'}; pairs "
              f"{sum(c.pairs for c in mine)} ({sum(c.ordered for c in mine)} ordered, "
              f"{sum(c.queue_only for c in mine)} queue-order only, "
              f"{sum(c.unsync_count for c in mine)} UNSYNCHRONIZED); one-pass "
              + " ".join(f"{w} {n}" for w, n in sorted(words.items())) + "; loop-carried "
              + " ".join(f"{w} {n}" for w, n in sorted(loop_words.items()))
              + f"; checks' operands: own {sum(c.own for c in mine)}"
              + ("" if name == program.CALLER else
                 f", {program.CALLER} {sum(c.owner_field for c in mine)} named + "
                 f"{sum(c.bound for c in mine)} bound")
              + "".join(f", {owner} {n}" for owner, n in sorted(siblings.items()))
              + f"; dump's operands: own {census[name]}"
              + "".join(f", {owner} {n}" for owner, n in sorted(census.items()) if owner != name))


def print_cross(program, verbose=False, show_all=False):
    """Print the cross-function pairs of `program`: each pair not OK (every pair with
    `show_all`), once per pair of accesses and kind (each one with `verbose`); the counts by
    function pair and tag; the cross-function verdict."""
    pairs = program.cross_pairs()
    graph = Model.graph_of({}, {}, [])
    print(f"\n## cross-function pairs on {program.scope()} (the earlier call's access first)")
    if program.unchecked:
        print(f"  UNCHECKED: {program.unchecked}")
    seen = set()
    for p in pairs:
        key = (p.src_label, p.src["name"], p.dst_label, p.dst["name"], p.kind)
        if (p.tag == "OK" and not show_all) or (key in seen and not verbose):
            continue
        seen.add(key)
        print(f"  [{p.tag}] {p.kind} {p.src_label}:{graph.label(p.src)}  ->  "
              f"{p.dst_label}:{graph.label(p.dst)}")
        print(f"         prod {p.src_fp['owner']}: {fp_str(p.src_fp)}")
        print(f"         cons {p.dst_fp['owner']}: {fp_str(p.dst_fp)}"
              + (f"   (not confirmed: {p.because})" if p.because else ""))
    counts = collections.Counter((p.src_function, p.dst_function, p.tag) for p in pairs)
    for src, dst in sorted({(p.src_function, p.dst_function) for p in pairs}):
        print(f"  {src} -> {dst}: "
              + ", ".join(f"{tag} {counts[src, dst, tag]}" for tag in CROSS_TAGS))
    owners = collections.Counter((p.src_fp["owner"], p.dst_fp["owner"], p.tag) for p in pairs)
    for src, dst in sorted({(p.src_fp["owner"], p.dst_fp["owner"]) for p in pairs}):
        print(f"  on locations of {src} / {dst}: "
              + ", ".join(f"{tag} {owners[src, dst, tag]}" for tag in CROSS_TAGS))
    total = collections.Counter(p.tag for p in pairs)
    confirmed = sum(1 for p in pairs if p.tag == "UNSYNC" and p.because is None)
    print(f"\n  {len(pairs)} cross-function pairs: "
          + ", ".join(f"{tag} {total[tag]}" for tag in CROSS_TAGS)
          + f" (UNSYNC confirmed {confirmed})")
    print(f"VERDICT (cross-function): {program.cross_verdict()} ({len(pairs)} pairs, "
          f"{confirmed} confirmed, {total['UNCHECKED']} UNCHECKED)")


def print_loop_carried(M, show_all=False, name=None):
    """The loop-carried class of `M` (module docstring): per loop, its pair counts by kind and
    tag; each finding (every pair with `show_all`); the reasons it is undecided; its verdict.
    `name`: the function's name for the VERDICT line in a linked dump."""
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
    label = "loop-carried" if name is None else f"{name} loop-carried"
    print(f"VERDICT ({label}): {verdict} ({len(findings)} findings in {len(M.loops)} loops)")


if __name__ == "__main__":
    main()
