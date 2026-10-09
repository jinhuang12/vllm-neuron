# SPDX-License-Identifier: Apache-2.0
"""A wait's ``from`` names its producer; walrus repeats that name in every loop body.

Each dynamic loop's counter increment on every engine is named
``scf.for-Inc_inst__I-10-0[-<engine>0]`` in every loop body of a kernel (9 bodies in
trn2-2's v8d dump), and the end-of-body waits name it. The 07:24Z checker kept one node per
name, the last body's, so every such wait was checked against the last body's semaphore
count: 20 UNSAFE-EARLY flags per v8d dump. The 10:05Z copy bound the name to the first
instance whose count reaches the wait value, which can be an earlier body's. A wait now
binds within its own block first, and a wait inside a loop body on a producer of the same
body is checked for the later iterations too.
"""

from __future__ import annotations

from test.kernel_depcheck import depcheck
from test.kernel_depcheck.tests.synthetic import Dump

INC = "scf.for-Inc_inst__I-10-0-SP0"
ENGINES = ("SP", "Pool")


def two_loops(tmp_path, first_updates: int, second_updates: int):
    """Two dynamic loops whose bodies both name their SP counter increment ``INC``.

    Body k updates semaphore 13 ``k_updates`` times (the increment last) and its end waits
    for exactly that count, ``from`` the increment; each body ends by resetting 13.
    """
    d = Dump()
    d.ins(d.block("Block1"), "start", "Pool", "EventSemaphore")
    for k, updates in ((1, first_updates), (2, second_updates)):
        body, exit_ = f"Block1_LoopBody_{k}", f"Block1_LoopExit_{k}"
        b = d.block(body)
        for u in range(updates - 1):
            d.ins(b, f"I-{k}{u}_load-SP0", "SP", "TensorLoad", updates=[(13, 1)])
        d.ins(b, INC, "SP", "RegisterAlu", updates=[(13, 1)])
        Dump.loop_end(b, body, exit_, ENGINES, waits=[(INC, 13, updates)], reset=(13,))
        d.ins(d.block(exit_), f"exit-{k}", "Pool", "EventSemaphore")
    return depcheck.Model(str(d.write(tmp_path / "two_loops.json")))


def test_a_repeated_producer_name_binds_each_wait_to_its_own_loop_body(tmp_path):
    # (2, 1): body 1's instance (count 2) is the first to reach body 2's wait value 1, so
    # binding to the first instance that reaches the value checks body 2's wait against
    # body 1's count. (1, 2): the last instance (body 2, count 2) is above body 1's wait
    # value 1, so binding to the last instance flags body 1's wait UNSAFE-EARLY.
    for first, second in ((2, 1), (1, 2)):
        model = two_loops(tmp_path, first, second)
        assert [s for s in model.check_sem_arith() if s[0] != "OK"] == [], (first, second)
        bound = {model.block_of[waiter]: model.block_of[src]
                 for waiter, src, _, w in model.waits if w["from"] == INC}
        assert bound == {"Block1_LoopBody_1": "Block1_LoopBody_1",
                         "Block1_LoopBody_2": "Block1_LoopBody_2"}, (first, second)
        assert not model.undecided


def test_a_wait_on_a_name_outside_its_block_binds_to_the_nearest_earlier_instance(tmp_path):
    """The exit block's wait names a loop increment that only the bodies hold: the instance
    that ran last before it is the one in the body just left."""
    d = Dump()
    d.ins(d.block("Block1"), "start", "Pool", "EventSemaphore")
    for k in (1, 2):
        body, exit_ = f"Block1_LoopBody_{k}", f"Block1_LoopExit_{k}"
        b = d.block(body)
        d.ins(b, INC, "SP", "RegisterAlu", updates=[(13, 1)])
        Dump.loop_end(b, body, exit_, ENGINES)
        d.ins(d.block(exit_), f"exit-{k}", "Pool", "EventSemaphore", waits=[(INC, 13, 1)])
    model = depcheck.Model(str(d.write(tmp_path / "exit_waits.json")))
    bound = {model.block_of[waiter]: model.block_of[src] for waiter, src, _, _ in model.waits}
    assert bound == {"Block1_LoopExit_1": "Block1_LoopBody_1",
                     "Block1_LoopExit_2": "Block1_LoopBody_2"}


def _one_loop(tmp_path, build_body, *, before=()):
    """Block1 (``before``: SP semaphore-13 updates of its own), one loop body, its exit."""
    d = Dump()
    pre = d.block("Block1")
    d.ins(pre, "start", "Pool", "EventSemaphore")
    for name in before:
        d.ins(pre, name, "SP", "TensorLoad", updates=[(13, 1)])
    body = "Block1_LoopBody_1"
    build_body(d.block(body), body)
    d.ins(d.block("Block1_LoopExit_1"), "exit", "Pool", "EventSemaphore")
    return depcheck.Model(str(d.write(tmp_path / "one_loop.json")))


def test_a_wait_on_a_later_instruction_of_its_own_loop_body_is_later_producer(tmp_path):
    """A wait on an instruction its own body runs only after it waits, from the second
    iteration on, for the previous iteration's instance: no edge in one pass of the body,
    and the loop-carried class does not model it."""
    def body(b, name):
        Dump.ins(b, "early", "Pool", "EventSemaphore", waits=[(INC, 13, 1)])
        Dump.ins(b, INC, "SP", "RegisterAlu", updates=[(13, 1)])
        Dump.loop_end(b, name, "Block1_LoopExit_1", ENGINES, reset=(13,))

    model = _one_loop(tmp_path, body)
    assert [s[:2] for s in model.check_sem_arith()] == [("LATER-PRODUCER", "early")]
    assert [(waiter, src) for waiter, src, _, _ in model.waits] == [("early", None)]
    assert model.loop_carried_verdict() == "UNDECIDED"
    assert any("early" in why for why in model.loop_undecided_reasons())


def test_a_semaphore_the_body_never_resets_accumulates(tmp_path):
    """The end of the body waits for count 1 of a semaphore the body raises by 1 and never
    resets: from the second iteration on the count is already 1 before the increment."""
    def body(b, name):
        Dump.ins(b, INC, "SP", "RegisterAlu", updates=[(13, 1)])
        Dump.loop_end(b, name, "Block1_LoopExit_1", ENGINES, waits=[(INC, 13, 1)])

    model = _one_loop(tmp_path, body)
    assert [s[0] for s in model.check_sem_arith()] == ["ACCUMULATES"]


def test_a_count_only_the_first_iteration_sees_is_checked_for_the_later_ones(tmp_path):
    """Block1 leaves semaphore 13 at 1; the body's increment brings it to 2 in the first
    iteration, and the end waits for 2. The body resets 13 at its end, so from the second
    iteration on the increment reaches 1 only: the wait for 2 waits for an update that
    never comes (OVER-WAIT from iteration 2 on)."""
    def body(b, name):
        Dump.ins(b, INC, "SP", "RegisterAlu", updates=[(13, 1)])
        Dump.loop_end(b, name, "Block1_LoopExit_1", ENGINES, waits=[(INC, 13, 2)], reset=(13,))

    model = _one_loop(tmp_path, body, before=("pre_load",))
    assert [(s[0], s[5]) for s in model.check_sem_arith()] == [("OVER-WAIT (iteration 2+)", 1)]
