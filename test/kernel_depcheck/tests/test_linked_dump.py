# SPDX-License-Identifier: Apache-2.0
"""A linked dump holds several functions: the kernels and ``main``, whose Calls run them.

A Call names its function in ``target`` and binds the function's parameters to ``main``'s
memory locations in ``arguments``; ``main`` can call a function more than once. A DMATrigger of
such a function names one DMA block per call, and the blocks' descriptors name a location of
another function (``main`` or another kernel) with ``memref_owner_func``. The kernels of one
program often share location names (``intermediate0``) at different addresses. The check
reads every call with its bindings and blocks, each operand in its owner's allocations, and
pairs the accesses of different calls to memory of more than one function, in ``main``'s
call order. Without a call order such pairs are UNCHECKED, never CLEAN.
"""

from __future__ import annotations

import collections
import random
import sys

from test.kernel_depcheck import depcheck
from test.kernel_depcheck.tests.synthetic import Dump, write_linked

#: Bytes of a ``Dump.whole`` operand on a float32 DRAM tensor.
SPAN_BYTES = 512 * 4


def calls(main: Dump, *plan, switched=()) -> Dump:
    """``main`` with one block: a SwitchQueueInstance on each queue in ``switched``, then one
    Call ``call_<k>`` for each (function, {parameter: main's location}) of ``plan``."""
    b = main.block("main_Block1")
    for q in switched:
        Dump.ins(b, f"switch_{q}", "SP", "SwitchQueueInstance", queue=q)
    for k, (target, arguments) in enumerate(plan):
        Dump.ins(b, f"call_{k}", "Unassigned", "Call", target=target,
                 arguments={f"{p}_set": f"{a}_set" for p, a in arguments.items()})
    return main


def copy(block: list[dict], name: str, engine: str, *, reads, writes, queue: str, updates=()):
    """A DMACopy on ``engine``'s ``queue``."""
    return Dump.ins(block, name, engine, "DMACopy", reads=reads, writes=writes, queue=queue,
                    updates=updates)


def trigger(dump: Dump, block: list[dict], name: str, blocks, *, queue: str = "qSP",
            engine: str = "SP"):
    """A DMATrigger ``<name>-trig`` on ``engine`` that starts ``blocks`` (made on ``queue``)."""
    return Dump.ins(block, f"{name}-trig", engine, "DMATrigger", dma_blocks=list(blocks),
                    queue=queue)


def program_of(tmp_path, name: str, functions) -> depcheck.Program:
    """The checked Program of a dump of ``functions``, its Models kept."""
    path = write_linked(tmp_path / f"{name}.json", functions)
    return depcheck.Program(str(path)).check(keep_models=True)


def cross(program):
    return [(p.kind, p.src_function, p.dst_function, p.tag) for p in program.cross_pairs()]


def test_a_descriptor_operand_resolves_in_the_allocations_of_its_owner(tmp_path):
    """``sg0000`` and ``main`` both have a location named ``intermediate0``: the parameter of
    ``sg0000`` at byte 0, the location of ``main`` at byte 4096 of the local arena. A
    descriptor that names ``main``'s is read at its address; looked up in the allocations of
    ``sg0000`` it would be read at byte 0."""
    main = Dump("main")
    main.dram("intermediate0", kind="InternalInterface", addr=4096)
    sg = Dump("sg0000")
    sg.dram("intermediate0", kind="Output")
    sg.tile("t")
    sg.dma_block("qSP", "store-block-2", reads=[sg.whole("t")],
                 writes=[Dump.foreign(main, "intermediate0")])
    trigger(sg, sg.block("Block1"), "store", ["store-block-2"])
    path = write_linked(tmp_path / "colliding.json",
                        [sg, calls(main, ("sg0000", {"intermediate0": "intermediate0"}))])

    model = depcheck.Model(str(path))
    (fp,) = model.acc[0]["writes"]
    assert (fp["lo"], fp["hi"]) == (4096, 4096 + SPAN_BYTES)
    assert (fp["owner"], fp["space"]) == ("main", "DRAM:INTERNAL")
    assert model.foreign_operands == {"main": 1}
    assert not model.undecided


def test_a_descriptor_operand_resolves_in_the_allocations_of_a_sibling_owner(tmp_path):
    """``sg0001`` and ``sg0000`` both have an internal location named ``spill``: the one of
    ``sg0001`` at byte 0 of the local arena, the one of ``sg0000`` at byte 4096. A descriptor
    of ``sg0001`` that names the one of ``sg0000`` is read at its address; looked up by its
    name in the allocations of ``sg0001`` it would be read at byte 0."""
    sg0 = Dump("sg0000")
    sg0.dram("spill", kind="Internal", addr=4096)
    sg1 = Dump("sg0001")
    sg1.dram("spill", kind="Internal")
    sg1.tile("t")
    sg1.dma_block("qSP", "reload-block-2", reads=[Dump.foreign(sg0, "spill")],
                  writes=[sg1.whole("t")])
    trigger(sg1, sg1.block("Block1"), "reload", ["reload-block-2"])
    path = write_linked(tmp_path / "sibling.json",
                        [sg1, sg0, calls(Dump("main"), ("sg0000", {}), ("sg0001", {}))])

    model = depcheck.Model(str(path))
    (fp,) = model.acc[0]["reads"]
    assert (fp["lo"], fp["hi"]) == (4096, 4096 + SPAN_BYTES)
    assert (fp["memref"], fp["owner"], fp["space"]) == ("spill", "sg0000", "DRAM:INTERNAL")
    assert model.foreign_operands == {"sg0000": 1}
    assert not model.undecided


def test_a_descriptor_operand_on_a_location_only_main_has_resolves(tmp_path):
    """``input239`` (the weights of a later call) is in ``main``'s allocations only: a lookup
    in the allocations of ``sg0002`` finds nothing."""
    main = Dump("main")
    main.dram("input239")
    sg = Dump("sg0002")
    sg.tile("t")
    sg.dma_block("qSP", "load-block-5", reads=[Dump.foreign(main, "input239")],
                 writes=[sg.whole("t")])
    trigger(sg, sg.block("Block1"), "load", ["load-block-5"])
    path = write_linked(tmp_path / "main_only.json", [sg, calls(main, ("sg0002", {}))])

    model = depcheck.Model(str(path))
    (fp,) = model.acc[0]["reads"]
    assert (fp["memref"], fp["owner"], fp["space"]) == ("input239", "main", "DRAM:main:input239")
    assert not model.undecided


def test_an_operand_whose_owner_has_no_such_location_is_undecided(tmp_path):
    """A wrong owner name is an operand the check cannot place: UNDECIDED, not a KeyError."""
    main = Dump("main")
    main.dram("x")
    sg = Dump("sg0000")
    sg.tile("t")
    sg.dma_block("qSP", "load-block-2", writes=[sg.whole("t")],
                 reads=[dict(Dump.foreign(main, "x"), memref_owner_func="sg0099")])
    trigger(sg, sg.block("Block1"), "load", ["load-block-2"])
    path = write_linked(tmp_path / "no_owner.json", [sg, calls(main, ("sg0000", {}))])

    model = depcheck.Model(str(path))
    assert model.undecided_class["other-operand"] == 1
    assert "sg0099" in model.undecided[0]


def test_a_call_binds_the_parameters_of_its_function_to_locations_of_main(tmp_path):
    """``sg0001`` copies its parameter ``x`` into a tile; ``main`` calls it on
    ``intermediate2`` (byte 8192), then on ``intermediate10`` (byte 16384)."""
    main = Dump("main")
    main.dram("intermediate2", kind="InternalInterface", addr=8192)
    main.dram("intermediate10", kind="InternalInterface", addr=16384)
    sg = Dump("sg0001")
    sg.dram("x", kind="Input")
    sg.tile("t")
    copy(sg.block("Block1"), "load", "SP", reads=[sg.whole("x")], writes=[sg.whole("t")],
         queue="qSP")
    program = program_of(tmp_path, "twice", [sg, calls(
        main, ("sg0001", {"x": "intermediate2"}), ("sg0001", {"x": "intermediate10"}))])

    assert [c.label for c in program.checks] == ["main", "sg0001 call_0", "sg0001 call_1"]
    reads = [program.models[label].acc[0]["reads"][0]
             for label in ("sg0001 call_0", "sg0001 call_1")]
    assert [(f["memref"], f["lo"], f["owner"], f["space"]) for f in reads] == [
        ("intermediate2", 8192, "main", "DRAM:INTERNAL"),
        ("intermediate10", 16384, "main", "DRAM:INTERNAL")]
    assert [(c.bound, c.owner_field) for c in program.checks[1:]] == [(1, 0), (1, 0)]
    assert not cross(program)       # two reads are not a pair
    assert program.verdict() == "CLEAN"


def test_each_call_starts_its_own_dma_block_and_waits_on_it(tmp_path):
    """The trigger of ``sg0001`` names one DMA block per call, and walrus's wait names the
    first call's block. The second call starts the second block, and the wait orders it.
    Read as one function, the dump starts both blocks and the wait orders only the first."""
    main = Dump("main")
    main.dram("w0")
    main.dram("w1")
    sg = Dump("sg0001")
    sg.tile("t")
    for k in (0, 1):
        sg.dma_block("qSP", f"load-block-{k + 3}", reads=[Dump.foreign(main, f"w{k}")],
                     writes=[sg.whole("t")], updates=[(5, 16)])
    b = sg.block("Block1")
    trigger(sg, b, "load", ["load-block-3", "load-block-4"])
    Dump.ins(b, "use", "Activation", "Activation", reads=[sg.whole("t")],
             waits=[("load-block-3", 5, 16)])
    program = program_of(tmp_path, "blocks", [sg, calls(main, ("sg0001", {}), ("sg0001", {}))])

    second = program.models["sg0001 call_1"]
    assert [a["name"] for a in second.acc] == ["load-block-4", "use"]
    assert second.acc[0]["reads"][0]["memref"] == "w1"
    assert program.verdicts == {"main": ("CLEAN", "CLEAN"), "sg0001 call_0": ("CLEAN", "CLEAN"),
                                "sg0001 call_1": ("CLEAN", "CLEAN")}
    assert program.verdict() == "CLEAN"
    assert depcheck.Model(program.path).classify().word == "FINDINGS"


def test_every_call_is_checked(tmp_path):
    """The unordered pair is inside the second call."""
    sg0 = Dump("sg0000")
    sg0.tile("a")
    Dump.ins(sg0.block("Block1"), "fill", "DVE", "Memset", writes=[sg0.whole("a")])
    sg1 = Dump("sg0001")
    sg1.tile("b")
    b1 = sg1.block("Block1")
    Dump.ins(b1, "write_b", "DVE", "TensorCopy", writes=[sg1.whole("b")])
    Dump.ins(b1, "read_b", "Activation", "Activation", reads=[sg1.whole("b")])
    program = program_of(tmp_path, "second", [sg0, sg1, calls(
        Dump("main"), ("sg0000", {}), ("sg0001", {}))])

    assert program.names == ["sg0000", "sg0001", "main"]
    assert {label: words[0] for label, words in program.verdicts.items()} == {
        "main": "CLEAN", "sg0000 call_0": "CLEAN", "sg0001 call_1": "FINDINGS"}
    assert program.verdict() == "FINDINGS"


def store_then_load(tmp_path, *, wait_for_the_store: bool, load_from: str = "intermediate2",
                    load_writes: bool = False):
    """``sg0000`` stores a tile into its parameter ``out`` (queue qA), bound to ``main``'s
    ``intermediate2``. ``sg0001`` then copies its parameter ``x`` (queue qB), bound to
    ``load_from``, into a tile, or with ``load_writes`` a tile into it. ``intermediate10``
    takes the arena bytes of ``intermediate2``. With ``wait_for_the_store``, ``sg0000`` ends
    with a Pool wait on the store and an all-engine barrier."""
    main = Dump("main")
    main.dram("intermediate2", kind="InternalInterface", addr=8192)
    main.dram("intermediate10", kind="InternalInterface", addr=8192)
    sg0 = Dump("sg0000")
    sg0.dram("out", kind="Output")
    sg0.tile("a")
    b0 = sg0.block("Block1")
    copy(b0, "store", "SP", reads=[sg0.whole("a")], writes=[sg0.whole("out")], queue="qA",
         updates=[(3, 1)])
    if wait_for_the_store:
        Dump.ins(b0, "wait_store", "Pool", "EventSemaphore", waits=[("store", 3, 1)])
        Dump.ins(b0, "end_barrier", "ALL", "AllEngineBarrier")
    sg1 = Dump("sg0001")
    sg1.dram("x", kind="Output" if load_writes else "Input")
    sg1.tile("b")
    ops = ([sg1.whole("b")], [sg1.whole("x")]) if load_writes else ([sg1.whole("x")],
                                                                     [sg1.whole("b")])
    copy(sg1.block("Block1"), "load", "SP", reads=ops[0], writes=ops[1], queue="qB")
    return program_of(tmp_path, "store_then_load", [sg0, sg1, calls(
        main, ("sg0000", {"out": "intermediate2"}), ("sg0001", {"x": load_from}))])


def test_a_cross_function_store_nothing_waits_for_is_a_finding(tmp_path):
    program = store_then_load(tmp_path, wait_for_the_store=False)
    assert program.order == ["sg0000", "sg0001"]
    assert cross(program) == [("RAW", "sg0000", "sg0001", "UNSYNC")]
    assert program.cross_verdict() == "FINDINGS"
    assert program.verdict() == "FINDINGS"


def test_a_cross_function_store_waited_for_before_the_end_barrier_is_ordered(tmp_path):
    """The wait orders the store's completion before the barrier; the next call's load issues
    on an engine after it (engine order across the call)."""
    program = store_then_load(tmp_path, wait_for_the_store=True)
    assert cross(program) == [("RAW", "sg0000", "sg0001", "OK")]
    assert program.cross_verdict() == "CLEAN"
    assert program.verdict() == "CLEAN"


def test_two_locations_of_main_on_the_same_arena_bytes_are_paired(tmp_path):
    """``intermediate10`` reuses the bytes of ``intermediate2``: the next call's write into it
    while the store into ``intermediate2`` runs is a WAW pair, though the names differ."""
    program = store_then_load(tmp_path, wait_for_the_store=False, load_from="intermediate10",
                              load_writes=True)
    assert cross(program) == [("WAW", "sg0000", "sg0001", "UNSYNC")]
    assert program.verdict() == "FINDINGS"


def test_an_order_through_a_call_in_between(tmp_path):
    """Only Pool waits for the store of ``sg0000``; the load of ``sg0002`` is on SP. A call of
    ``sg0001`` between them with an all-engine barrier orders the load after the wait;
    without the barrier nothing does."""
    for barrier, tag in ((True, "OK"), (False, "UNSYNC")):
        main = Dump("main")
        main.dram("intermediate2", kind="InternalInterface", addr=8192)
        sg0 = Dump("sg0000")
        sg0.dram("out", kind="Output")
        sg0.tile("a")
        b0 = sg0.block("Block1")
        copy(b0, "store", "SP", reads=[sg0.whole("a")], writes=[sg0.whole("out")], queue="qA",
             updates=[(3, 1)])
        Dump.ins(b0, "wait_store", "Pool", "EventSemaphore", waits=[("store", 3, 1)])
        sg1 = Dump("sg0001")
        sg1.tile("c")
        b1 = sg1.block("Block1")
        Dump.ins(b1, "fill", "DVE", "Memset", writes=[sg1.whole("c")])
        if barrier:
            Dump.ins(b1, "barrier", "ALL", "AllEngineBarrier")
        sg2 = Dump("sg0002")
        sg2.dram("x", kind="Input")
        sg2.tile("b")
        copy(sg2.block("Block1"), "load", "SP", reads=[sg2.whole("x")], writes=[sg2.whole("b")],
             queue="qB")
        program = program_of(tmp_path, f"through_{barrier}", [sg0, sg1, sg2, calls(
            main, ("sg0000", {"out": "intermediate2"}), ("sg0001", {}),
            ("sg0002", {"x": "intermediate2"}))])
        assert cross(program) == [("RAW", "sg0000", "sg0002", tag)], barrier


def descriptors_store_then_load(tmp_path, plan):
    """``sg0000`` stores into ``main``'s ``intermediate2`` and ``sg0001`` loads from it, both
    through DMA blocks whose descriptors name it (no parameter); ``main`` calls ``plan``."""
    main = Dump("main")
    main.dram("intermediate2", kind="InternalInterface", addr=8192)
    sg0 = Dump("sg0000")
    sg0.tile("a")
    sg0.dma_block("qA", "store-block-2", reads=[sg0.whole("a")],
                  writes=[Dump.foreign(main, "intermediate2")])
    trigger(sg0, sg0.block("Block1"), "store", ["store-block-2"], queue="qA")
    sg1 = Dump("sg0001")
    sg1.tile("b")
    sg1.dma_block("qB", "load-block-3", reads=[Dump.foreign(main, "intermediate2")],
                  writes=[sg1.whole("b")])
    trigger(sg1, sg1.block("Block1"), "load", ["load-block-3"], queue="qB")
    return program_of(tmp_path, "descriptors", [sg0, sg1, calls(main, *plan)])


def test_cross_function_pairs_without_a_call_order_are_unchecked(tmp_path):
    """``main`` calls nothing, or calls a function the dump does not have: the dump gives no
    order. Each function is checked as a one-function dump, its parameters unbound."""
    for plan in ((), (("sg0099", {}),)):
        program = descriptors_store_then_load(tmp_path, plan)
        assert program.calls is None and program.order_unknown, plan
        assert program.cross_owned == set(), plan
        assert [c.label for c in program.checks] == ["main", "sg0000", "sg0001"], plan
        assert cross(program) == [("RAW", "sg0000", "sg0001", "UNCHECKED")], plan
        assert program.cross_verdict() == "UNDECIDED", plan
        assert program.verdict() == "UNDECIDED", plan


def test_without_a_call_order_a_pair_on_a_parameter_of_a_sibling_is_listed(tmp_path):
    """``main`` calls nothing: the parameter ``out`` of ``sg0000`` is unbound, a tensor of its
    own, and ``sg0001`` names it. The store of ``sg0000`` into it and the load of ``sg0001``
    are a pair, UNCHECKED and listed."""
    sg0 = Dump("sg0000")
    sg0.dram("out", kind="Output")
    sg0.tile("a")
    copy(sg0.block("Block1"), "store", "SP", reads=[sg0.whole("a")], writes=[sg0.whole("out")],
         queue="qA")
    sg1 = Dump("sg0001")
    sg1.tile("b")
    sg1.dma_block("qB", "load-block-3", reads=[Dump.foreign(sg0, "out")],
                  writes=[sg1.whole("b")])
    trigger(sg1, sg1.block("Block1"), "load", ["load-block-3"], queue="qB")
    program = program_of(tmp_path, "unbound_sibling", [sg0, sg1, calls(Dump("main"))])
    assert program.cross_owned == {("sg0000", "out")}
    assert cross(program) == [("RAW", "sg0000", "sg0001", "UNCHECKED")]
    (p,) = program.cross_pairs()
    assert (p.src_fp["space"], p.dst_fp["space"]) == ("DRAM:sg0000:out", "DRAM:sg0000:out")
    assert program.cross_verdict() == "UNDECIDED"


def test_without_a_call_order_the_cross_verdict_is_undecided_even_with_no_pair(tmp_path):
    """Unbound parameters are memory of main the check cannot place: no pair is not CLEAN."""
    main = Dump("main")
    sg = Dump("sg0000")
    sg.dram("out", kind="Output")
    sg.tile("a")
    copy(sg.block("Block1"), "store", "SP", reads=[sg.whole("a")], writes=[sg.whole("out")],
         queue="qA")
    program = program_of(tmp_path, "unbound", [sg, calls(main)])
    assert cross(program) == []
    assert program.cross_verdict() == "UNDECIDED"


def test_a_dump_without_main_is_undecided(tmp_path):
    """No function calls the kernels, and the descriptors' owner is not in the dump."""
    main = Dump("main")
    main.dram("intermediate2", kind="InternalInterface", addr=8192)
    sg0 = Dump("sg0000")
    sg0.tile("a")
    sg0.dma_block("qA", "store-block-2", reads=[sg0.whole("a")],
                  writes=[Dump.foreign(main, "intermediate2")])
    trigger(sg0, sg0.block("Block1"), "store", ["store-block-2"], queue="qA")
    program = program_of(tmp_path, "no_main", [sg0])
    assert program.calls is None
    assert "main" in program.order_unknown
    assert "main" in program.models["sg0000"].undecided[0]
    assert program.verdict() == "UNDECIDED"


def test_a_trigger_with_another_number_of_blocks_than_calls_is_undecided(tmp_path):
    main = Dump("main")
    sg = Dump("sg0001")
    sg.tile("t")
    for k in range(3):
        sg.dma_block("qSP", f"fill-block-{k}", writes=[sg.whole("t")])
    trigger(sg, sg.block("Block1"), "fill", [f"fill-block-{k}" for k in range(3)])
    program = program_of(tmp_path, "blocks3", [sg, calls(main, ("sg0001", {}), ("sg0001", {}))])
    for label in ("sg0001 call_0", "sg0001 call_1"):
        model = program.models[label]
        assert model.undecided_class["other-operand"] == 1, label
        assert "3 DMA blocks for the 2 calls" in model.undecided[0], label
        assert program.verdicts[label][0] == "UNDECIDED", label


def sibling_store_then_reload(tmp_path, *, wait_for_the_store: bool, scratch_at=None):
    """``sg0000`` stores a tile into its internal location ``spill`` (byte 4096, queue qA);
    ``sg0001`` then reloads it (``memref_owner_func`` ``sg0000``, queue qB). With
    ``wait_for_the_store``, ``sg0000`` ends with a Pool wait on the store and an all-engine
    barrier. With ``scratch_at``, a call of ``sg0002`` between them writes its own internal
    location ``tmp`` at that byte of the arena, nothing waiting for it, and ``sg0001`` then
    writes its own ``spill`` (byte 0)."""
    sg0 = Dump("sg0000")
    sg0.dram("spill", kind="Internal", addr=4096)
    sg0.tile("a")
    b0 = sg0.block("Block1")
    copy(b0, "store", "SP", reads=[sg0.whole("a")], writes=[sg0.whole("spill")], queue="qA",
         updates=[(3, 1)])
    if wait_for_the_store:
        Dump.ins(b0, "wait_store", "Pool", "EventSemaphore", waits=[("store", 3, 1)])
        Dump.ins(b0, "end_barrier", "ALL", "AllEngineBarrier")
    sg1 = Dump("sg0001")
    sg1.dram("spill", kind="Internal")
    sg1.tile("t")
    sg1.dma_block("qB", "reload-block-2", reads=[Dump.foreign(sg0, "spill")],
                  writes=[sg1.whole("t")])
    b1 = sg1.block("Block1")
    trigger(sg1, b1, "reload", ["reload-block-2"], queue="qB")
    functions, plan = [sg0, sg1], [("sg0000", {})]
    if scratch_at is not None:
        copy(b1, "own_store", "Activation", reads=[sg1.whole("t")], writes=[sg1.whole("spill")],
             queue="qD")
        sg2 = Dump("sg0002")
        sg2.dram("tmp", kind="Internal", addr=scratch_at)
        sg2.tile("c")
        copy(sg2.block("Block1"), "scratch", "SP", reads=[sg2.whole("c")],
             writes=[sg2.whole("tmp")], queue="qC")
        functions.append(sg2)
        plan.append(("sg0002", {}))
    plan.append(("sg0001", {}))
    return program_of(tmp_path, f"sibling_{wait_for_the_store}_{scratch_at}",
                      functions + [calls(Dump("main"), *plan)])


def test_a_cross_function_pair_on_a_location_of_a_sibling(tmp_path):
    """The reload of ``sg0001`` and the store of ``sg0000`` are on a location of ``sg0000``
    that another function names: a cross-function pair, the owner's own access included."""
    for wait, tag, word in ((False, "UNSYNC", "FINDINGS"), (True, "OK", "CLEAN")):
        program = sibling_store_then_reload(tmp_path, wait_for_the_store=wait)
        assert program.cross_owned == {("sg0000", "spill")}
        assert cross(program) == [("RAW", "sg0000", "sg0001", tag)], wait
        (p,) = program.cross_pairs()
        assert (p.src_fp["owner"], p.dst_fp["owner"]) == ("sg0000", "sg0000")
        assert program.cross_verdict() == word, wait


def test_a_location_on_the_arena_bytes_of_a_shared_location_is_paired(tmp_path):
    """The scratch of ``sg0002`` from byte 3072 runs into the bytes of the spill of ``sg0000``
    (from byte 4096): it is paired with the store before it and the reload after it, though no
    other function names it. At byte 0, off those bytes, it is checked inside its call only:
    so is its WAW pair with the store of ``sg0001`` into its own ``spill``, which no other
    function names."""
    program = sibling_store_then_reload(tmp_path, wait_for_the_store=True, scratch_at=3072)
    assert sorted(cross(program)) == [("RAW", "sg0000", "sg0001", "OK"),
                                      ("RAW", "sg0002", "sg0001", "UNSYNC"),
                                      ("WAW", "sg0000", "sg0002", "OK")]
    program = sibling_store_then_reload(tmp_path, wait_for_the_store=True, scratch_at=0)
    assert cross(program) == [("RAW", "sg0000", "sg0001", "OK")]


def test_a_parameter_of_a_sibling_is_the_location_its_one_call_binds(tmp_path):
    """A descriptor of ``sg0001`` names the parameter ``out`` of ``sg0000``. ``main`` calls
    ``sg0000`` once, on ``intermediate2``: the operand is on ``intermediate2``. Called twice
    (also on ``intermediate10``), which call's location it names is not known: UNDECIDED."""
    for twice in (False, True):
        main = Dump("main")
        main.dram("intermediate2", kind="InternalInterface", addr=8192)
        main.dram("intermediate10", kind="InternalInterface", addr=16384)
        sg0 = Dump("sg0000")
        sg0.dram("out", kind="Output")
        sg0.tile("a")
        b0 = sg0.block("Block1")
        copy(b0, "store", "SP", reads=[sg0.whole("a")], writes=[sg0.whole("out")], queue="qA",
             updates=[(3, 1)])
        Dump.ins(b0, "wait_store", "Pool", "EventSemaphore", waits=[("store", 3, 1)])
        Dump.ins(b0, "end_barrier", "ALL", "AllEngineBarrier")
        sg1 = Dump("sg0001")
        sg1.tile("t")
        sg1.dma_block("qB", "load-block-2", reads=[Dump.foreign(sg0, "out")],
                      writes=[sg1.whole("t")])
        trigger(sg1, sg1.block("Block1"), "load", ["load-block-2"], queue="qB")
        plan = [("sg0000", {"out": "intermediate2"})]
        plan += [("sg0000", {"out": "intermediate10"})] if twice else []
        program = program_of(tmp_path, f"param_{twice}",
                             [sg0, sg1, calls(main, *plan, ("sg0001", {}))])
        label = f"sg0001 call_{len(plan)}"
        model = program.models[label]
        if twice:
            assert model.undecided_class["other-operand"] == 1
            assert "parameter of sg0000, which main calls 2 times" in model.undecided[0]
            assert program.verdicts[label][0] == "UNDECIDED"
            continue
        (fp,) = model.acc[0]["reads"]
        assert (fp["memref"], fp["lo"], fp["owner"], fp["bound"]) == (
            "intermediate2", 8192, "main", False)
        assert cross(program) == [("RAW", "sg0000", "sg0001", "OK")]
        assert program.verdict() == "CLEAN"


def test_each_check_counts_its_operands_by_owner(tmp_path):
    """The heading's counts (Check) and the dump's (Program.census): ``sg0001``'s reload names
    one location of ``sg0000``, its tile is its own."""
    program = sibling_store_then_reload(tmp_path, wait_for_the_store=True)
    counts = {c.label: (c.own, c.own_names, c.owner_field, c.bound, c.siblings)
              for c in program.checks}
    assert counts == {"main": (0, 0, 0, 0, ()),
                      "sg0000 call_0": (2, 2, 0, 0, ()),
                      "sg0001 call_1": (1, 1, 0, 0, (("sg0000", 1, 1),))}
    assert program.census == {"sg0000": {"sg0000": 2}, "sg0001": {"sg0001": 1, "sg0000": 1},
                              "main": {}}


def test_a_linked_dump_has_no_bare_verdict_line(tmp_path, monkeypatch, capsys):
    """A reader of the one-function VERDICT line finds none, so it cannot take one call's
    verdict for the dump's: the calls' lines name their function and call, and the dump's
    verdict is the `VERDICT (program)` line."""
    program = store_then_load(tmp_path, wait_for_the_store=False)
    monkeypatch.setattr(sys, "argv", ["depcheck", program.path])
    depcheck.main()
    lines = capsys.readouterr().out.splitlines()
    assert not [line for line in lines if depcheck.VERDICT_LINE.match(line)]
    named = [line for line in lines if line.startswith("VERDICT (sg0")
             and "loop-carried" not in line]
    assert [line.split(")", 1)[0] for line in named] == ["VERDICT (sg0000 call_0",
                                                         "VERDICT (sg0001 call_1"]
    assert all(depcheck.VERDICT_LINE.match("VERDICT:" + line.split("):", 1)[1])
               for line in named)
    assert [line for line in lines if line.startswith("VERDICT (program): ")] == [
        "VERDICT (program): FINDINGS (main CLEAN/CLEAN, sg0000 call_0 CLEAN/CLEAN, "
        "sg0001 call_1 CLEAN/CLEAN; cross-function FINDINGS)"]
    assert "  [UNSYNC] RAW sg0000 call_0:store DMACopy@SP/DMA:qA L1 None  ->  " \
           "sg0001 call_1:load DMACopy@SP/DMA:qB L1 None" in lines
    assert "# call call_0: sg0000, call 1 of 1; operands by owner: own 1 (1 location); main 0 " \
           "named by memref_owner_func (0 locations), 1 bound by the call's arguments " \
           "(1 location)" in lines
    assert "  sg0000: 1 call; pairs 0 (0 ordered, 0 queue-order only, 0 UNSYNCHRONIZED); " \
           "one-pass CLEAN 1; loop-carried CLEAN 1; checks' operands: own 1, main 0 named + 1 " \
           "bound; dump's operands: own 2" in lines
    assert "  on locations of main / main: OK 0, QUEUE-ORDER-ONLY 0, UNSYNC 1, UNCHECKED 0" \
           in lines


def random_program(tmp_path, seed: int) -> depcheck.Program:
    """Three kernels of random DMAs (each on its own semaphore), waits on them, all-engine
    barriers and copies; ``main`` calls them in a random order with random bindings of their
    two parameters to four locations of ``main``, two pairs of which share arena bytes. On
    odd seeds ``main`` switches queue qA."""
    rng = random.Random(seed)
    main = Dump("main")
    for k in range(4):
        main.dram(f"m{k}", kind="InternalInterface", addr=4096 * (k % 2))
    kernels = []
    for name in ("sg0000", "sg0001", "sg0002"):
        f = Dump(name)
        f.dram("p", kind="Input")
        f.dram("r", kind="Output")
        f.tile("t")
        b = f.block("Block1")
        dmas = []
        for n in range(rng.randint(4, 12)):
            r = rng.random()
            if r < 0.45:
                param = rng.choice(["p", "r"])
                ops = ([f.whole(param)], [f.whole("t")]) if rng.random() < 0.5 else (
                    [f.whole("t")], [f.whole(param)])
                copy(b, f"d{n}", rng.choice(["SP", "Pool", "Activation"]), reads=ops[0],
                     writes=ops[1], queue=rng.choice(["qA", "qB"]), updates=[(n + 1, 1)])
                dmas.append((f"d{n}", n + 1))
            elif r < 0.65 and dmas:
                src, sem = rng.choice(dmas)
                Dump.ins(b, f"w{n}", rng.choice(["SP", "Pool", "DVE"]), "EventSemaphore",
                         waits=[(src, sem, 1)])
            elif r < 0.75:
                Dump.ins(b, f"bar{n}", "ALL", "AllEngineBarrier")
            else:
                Dump.ins(b, f"c{n}", rng.choice(["DVE", "Activation", "SP"]), "TensorCopy",
                         reads=[f.whole("t")], writes=[f.whole("t")])
        kernels.append(f)
    plan = [(rng.choice(kernels).name, {"p": f"m{rng.randrange(4)}", "r": f"m{rng.randrange(4)}"})
            for _ in range(rng.randint(3, 6))]
    return program_of(tmp_path, f"random_{seed}",
                      kernels + [calls(main, *plan, switched=("qA",) if seed % 2 else ())])


def whole_graph(program: depcheck.Program) -> depcheck.Model:
    """The program's happens-before graph built whole, which Program does not build: each
    call's graph (nodes ``<label>/<node>``), joined in call order by the program-order rules
    of Model.build; the last DMA of a queue main switches is not carried into the next call."""
    edges, qedges = collections.defaultdict(set), collections.defaultdict(set)
    last_on, last_barrier, last_on_queue = {}, None, {}
    for check in program.checks:
        if check.call is None:
            continue
        M, label = program.models[check.label], check.label
        for own, linked in ((M.edges, edges), (M.qedges, qedges)):
            for n, after in own.items():
                linked[f"{label}/{n}"].update(f"{label}/{m}" for m in after)
        queue_at = collections.defaultdict(list)
        for k, q, node in M.queue_order:
            queue_at[k].append((q, f"{label}/{node}"))
        for k, (node, i) in enumerate(zip(M.node, M.ins)):
            me, eng = f"{label}/{node}", i["engine"]
            if eng == "ALL":
                for e, prev in list(last_on.items()):
                    edges[prev].add(me)
                    last_on[e] = me
                if last_barrier is not None:
                    edges[last_barrier].add(me)
                last_barrier = me
            elif eng != "Unassigned":
                prev = last_on.get(eng, last_barrier)
                if prev is not None:
                    edges[prev].add(me)
                last_on[eng] = me
            for q, n in queue_at[k]:
                if q in last_on_queue:
                    qedges[last_on_queue[q]].add(n)
                last_on_queue[q] = n
        for q in program.switched:
            last_on_queue.pop(q, None)
    return depcheck.Model.graph_of(edges, qedges, [])


def test_the_order_across_calls_is_the_order_of_the_whole_graph(tmp_path):
    """Program carries from call to call which last nodes an access reaches instead of building
    the program's graph; on random programs its tags are those of the whole graph."""
    tags = collections.Counter()
    for seed in range(40):
        program = random_program(tmp_path, seed)
        graph = whole_graph(program)
        for p in program.cross_pairs():
            src = dict(done=f"{p.src_label}/{p.src['done']}")
            dst = dict(issue=f"{p.dst_label}/{p.dst['issue']}")
            whole = ("OK" if graph.ordered(src, dst) else
                     "QUEUE-ORDER-ONLY" if graph.ordered(src, dst, True) else "UNSYNC")
            assert p.tag == whole, (seed, p.kind, p.src_label, p.src["name"], p.dst_label,
                                    p.dst["name"])
            tags[p.tag] += 1
    assert tags["OK"] and tags["UNSYNC"] and tags["QUEUE-ORDER-ONLY"], tags
