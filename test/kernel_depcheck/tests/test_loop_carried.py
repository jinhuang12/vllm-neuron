# SPDX-License-Identifier: Apache-2.0
"""The loop-carried-tile class on hand-built dumps of trn2-2's v8d pattern.

v8d (trn2-2, `moe_fused_fp8` records, 2026-10-08) staged the next item's operands across
the iterations of a dynamic item loop: iteration ``at`` computes from SBUF tiles allocated
outside the loop and refills them for item ``at + programs``: the gate/up weights by an
HWDGE DMA on the Sync queue, the hidden rows by PE transposes out of a tile an SW-DGE
gather filled, evacuated from PSUM by a tensor copy. The next iteration's products read
the refilled tiles, and the only order from the refill to that read is the loop's
back-edge: the end-of-body waits, the all-engine barrier and the branch. The device
computed it wrong; the simulator and the checker's same-iteration pairs did not see it.
"""

from __future__ import annotations

import json
import sys

import pytest

from test.kernel_depcheck import depcheck
from test.kernel_depcheck.tests.synthetic import Dump

BODY, EXIT = "Block1_LoopBody_1", "Block1_LoopExit_1"
ENGINES = ("PE", "DVE", "SP", "Pool")
HWDGE_QUEUE = "qSPDynamicHW"


def v8d_pattern(tmp_path, *, explicit: bool = False, end_wait_on_refill: bool = True,
                refill_queue: str = HWDGE_QUEUE, runtime_read: bool = False):
    """The v8d item loop: a prologue fills the staged tiles, then one body computes from them
    and refills them for the next iteration.

    ``explicit``: the iteration's own PE work waits, by semaphore, for both refills before
    the back-edge (the transposes for the weight DMA, a last PE read for the restaged rows),
    so PE's own instruction order carries them to the next iteration's products.
    ``end_wait_on_refill=False``: the end of the body does not wait for the weight DMA; it
    is then ordered only by a later DMA on its queue, or, on a queue of its own, by nothing.
    ``runtime_read``: the products read ``staged_x`` at a runtime (register) address.
    """
    d = Dump()
    staged_w = d.tile("staged_gate_up", addr=0, dtype="float8e4")
    staged_x = d.tile("staged_x", addr=4096, dtype="bfloat16")
    gathered = d.tile("gathered", addr=8192, dtype="bfloat16")
    products = d.tile("products", addr=12288)
    fence_out = d.tile("fence_out", addr=16384)
    psum = d.tile("psum_products", kind="PSUM", bank=0)
    psum_t = d.tile("psum_transpose", kind="PSUM", bank=1, dtype="bfloat16")
    psum_f = d.tile("psum_fence", kind="PSUM", bank=2)
    weights, hidden, out = d.dram("weights", dtype="float8e4"), d.dram("hidden"), d.dram("out")
    w = d.whole

    pre = d.block("Block1")
    d.ins(pre, "prologue_w", "SP", "DMACopy", reads=[w(weights)], writes=[w(staged_w)],
          queue=HWDGE_QUEUE, updates=[(20, 16)])
    d.dma_block("qPoolDynamic", "prologue_gather_block", reads=[w(hidden)],
                writes=[w(gathered)], updates=[(21, 16)])
    d.ins(pre, "prologue_gather", "Pool", "DMATrigger", dma_blocks=["prologue_gather_block"],
          queue="qPoolDynamic")
    d.ins(pre, "prologue_transpose", "PE", "Matmult", reads=[w(gathered)], writes=[w(psum_t)],
          waits=[("prologue_gather_block", 21, 16)], updates=[(22, 1)])
    d.ins(pre, "prologue_x", "DVE", "TensorCopy", reads=[w(psum_t)], writes=[w(staged_x)],
          waits=[("prologue_transpose", 22, 1)], updates=[(23, 1)])
    # Block1 falls through into the loop: its end sequence, no branch of its own.
    d.loop_end(pre, "Block1", BODY, (), waits=[("prologue_w", 20, 16), ("prologue_x", 23, 1)],
               reset=(20, 21, 22, 23))

    body = d.block(BODY)
    d.dma_block("qPoolDynamic", "gather_block", reads=[w(hidden)], writes=[w(gathered)],
                updates=[(31, 16)])
    d.ins(body, "gather", "Pool", "DMATrigger", dma_blocks=["gather_block"],
          queue="qPoolDynamic")
    x_operand = d.runtime(staged_x) if runtime_read else w(staged_x)
    d.ins(body, "products_mm", "PE", "Matmult", reads=[w(staged_w), x_operand],
          writes=[w(psum)], updates=[(32, 1)])
    d.ins(body, "evacuate", "DVE", "TensorCopy", reads=[w(psum)], writes=[w(products)],
          waits=[("products_mm", 32, 1)], updates=[(33, 1)])
    d.ins(body, "refill_w", "SP", "DMACopy", reads=[w(weights)], writes=[w(staged_w)],
          queue=refill_queue, waits=[("products_mm", 32, 1)], updates=[(34, 16)])
    d.ins(body, "store", "SP", "DMACopy", reads=[w(products)], writes=[w(out)],
          queue=HWDGE_QUEUE, waits=[("evacuate", 33, 1)], updates=[(35, 16)])
    transpose_waits = [("gather_block", 31, 16)] + ([("refill_w", 34, 16)] if explicit else [])
    d.ins(body, "transpose", "PE", "Matmult", reads=[w(gathered)], writes=[w(psum_t)],
          waits=transpose_waits, updates=[(36, 1)])
    d.ins(body, "restage_x", "DVE", "TensorCopy", reads=[w(psum_t)], writes=[w(staged_x)],
          waits=[("transpose", 36, 1)], updates=[(37, 1)])
    end_waits = [("store", 35, 16), ("restage_x", 37, 1)]
    if explicit:
        d.ins(body, "fence", "PE", "Matmult", reads=[w(staged_x)], writes=[w(psum_f)],
              waits=[("restage_x", 37, 1)], updates=[(38, 1)])
        d.ins(body, "fence_evacuate", "DVE", "TensorCopy", reads=[w(psum_f)],
              writes=[w(fence_out)], waits=[("fence", 38, 1)], updates=[(39, 1)])
        end_waits.append(("fence_evacuate", 39, 1))
    if end_wait_on_refill:
        end_waits.insert(0, ("refill_w", 34, 16))
    d.loop_end(body, BODY, EXIT, ENGINES, waits=end_waits, reset=range(31, 40))
    d.ins(d.block(EXIT), "after", "Pool", "EventSemaphore")
    return depcheck.Model(str(d.write(tmp_path / "v8d_pattern.json")))


def _pairs(model, kind="RAW"):
    return {(p.src, p.dst, p.tile): p for p in model.loop_carried() if p.kind == kind}


def test_a_tile_refilled_for_the_next_iteration_is_in_the_loop_carried_class(tmp_path):
    model = v8d_pattern(tmp_path)
    raw = _pairs(model)
    # The HWDGE weight refill and the PE -> PSUM -> copy restage, each read by the next
    # iteration's products, ordered by the back-edge alone.
    assert raw[("refill_w#done", "products_mm", "staged_gate_up")].tag == "BACK-EDGE-ONLY"
    assert raw[("restage_x", "products_mm", "staged_x")].tag == "BACK-EDGE-ONLY"
    assert {p.loop for p in raw.values()} == {BODY}
    # Tiles each iteration writes before it reads them carry nothing across the back-edge.
    assert set(raw) == {("refill_w#done", "products_mm", "staged_gate_up"),
                        ("restage_x", "products_mm", "staged_x")}
    assert {(p.src, p.dst) for p in model.loop_carried_findings()} == {
        ("refill_w#done", "products_mm"), ("restage_x", "products_mm")}
    assert model.loop_carried_verdict() == "FINDINGS"


def test_the_same_dump_with_an_explicit_semaphore_dependency_is_not(tmp_path):
    model = v8d_pattern(tmp_path, explicit=True)
    raw = _pairs(model)
    assert raw[("refill_w#done", "products_mm", "staged_gate_up")].tag == "OK"
    assert raw[("restage_x", "products_mm", "staged_x")].tag == "OK"
    assert model.loop_carried_findings() == []
    assert model.loop_carried_verdict() == "CLEAN"
    assert not model.undecided


def test_a_refill_ordered_only_by_a_later_dma_on_its_queue_is_queue_order_only(tmp_path):
    # Nothing waits for the weight DMA; the end of the body waits for the store, the next
    # DMA on the same queue, so the refill is complete only if that queue completes in order.
    model = v8d_pattern(tmp_path, end_wait_on_refill=False)
    raw = _pairs(model)
    assert raw[("refill_w#done", "products_mm", "staged_gate_up")].tag == "QUEUE-ORDER-ONLY"
    assert raw[("restage_x", "products_mm", "staged_x")].tag == "BACK-EDGE-ONLY"
    assert model.loop_carried_verdict() == "FINDINGS"


def test_a_refill_nothing_orders_is_unsynchronized(tmp_path):
    model = v8d_pattern(tmp_path, end_wait_on_refill=False, refill_queue="qSPDynamicHW1")
    assert _pairs(model)[("refill_w#done", "products_mm", "staged_gate_up")].tag == "UNSYNC"
    assert model.loop_carried_verdict() == "FINDINGS"


def test_the_gather_tile_reuse_is_ordered_by_the_back_edge_and_not_a_finding(tmp_path):
    """The SW-DGE gather writes ``gathered`` at the start of the body and the transposes
    read it later in the same iteration; the next gather overwrites it (a WAR across the
    back-edge). That is ordinary buffer reuse: a WAR or WAW across the back-edge is a
    finding only when not even the back-edge orders it."""
    model = v8d_pattern(tmp_path)
    war = _pairs(model, "WAR")
    assert war[("transpose", "gather_block", "gathered")].tag == "BACK-EDGE-ONLY"
    assert [p for p in model.loop_carried_findings() if p.kind != "RAW"] == []


@pytest.mark.parametrize("explicit", [False, True])
def test_the_same_iteration_check_is_unchanged_by_the_loop(tmp_path, explicit):
    model = v8d_pattern(tmp_path, explicit=explicit)
    unsync = [c for c in model.conflicts() if not model.ordered(c[1], c[2], True)]
    assert unsync == []
    assert [s for s in model.check_sem_arith() if s[0] != "OK"] == []


def test_a_runtime_address_read_is_taken_as_its_whole_tile(tmp_path):
    model = v8d_pattern(tmp_path, runtime_read=True)
    pair = _pairs(model)[("restage_x", "products_mm", "staged_x")]
    assert pair.tag == "BACK-EDGE-ONLY"
    assert pair.runtime
    # The same-iteration check still cannot place the operand.
    assert any("runtime-address SB operand on staged_x" in u for u in model.undecided)
    assert model.loop_carried_verdict() == "FINDINGS"


def test_the_compiler_listed_dependency_is_reported(tmp_path):
    """walrus lists a loop-carried dependency on the later access (``loop_carried_dependencies``:
    [name, Flow|Anti|Output, [distance]]); each pair says whether its kind and source are listed."""
    model = v8d_pattern(tmp_path)
    assert not _pairs(model)[("restage_x", "products_mm", "staged_x")].listed
    path = tmp_path / "v8d_pattern.json"
    d = json.loads(path.read_text())
    body = next(b for b in d["functions"][0]["blocks"] if b["name"] == BODY)
    products = next(i for i in body["instructions"] if i["name"] == "products_mm")
    products["loop_carried_dependencies"] = [["restage_x", "Flow", [1]]]
    path.write_text(json.dumps(d))
    listed = _pairs(depcheck.Model(str(path)))
    assert listed[("restage_x", "products_mm", "staged_x")].listed
    assert not listed[("refill_w#done", "products_mm", "staged_gate_up")].listed


def _single_loop(tmp_path, name, build_body):
    d = Dump()
    d.ins(d.block("Block1"), "start", "Pool", "EventSemaphore")
    build_body(d, d.block(BODY))
    d.ins(d.block(EXIT), "after", "Pool", "EventSemaphore")
    return depcheck.Model(str(d.write(tmp_path / name)))


def test_an_accumulator_on_one_engine_is_ordered_by_that_engine(tmp_path):
    def body(d, b):
        acc, x, src = d.tile("acc"), d.tile("x", addr=4096), d.dram("src")
        d.ins(b, "load_x", "SP", "DMACopy", reads=[d.whole(src)], writes=[d.whole(x)],
              queue=HWDGE_QUEUE, updates=[(40, 16)])
        d.ins(b, "accumulate", "DVE", "TensorTensor", reads=[d.whole(acc), d.whole(x)],
              writes=[d.whole(acc)], waits=[("load_x", 40, 16)], updates=[(41, 1)])
        Dump.loop_end(b, BODY, EXIT, ENGINES, waits=[("accumulate", 41, 1)], reset=(40, 41))

    model = _single_loop(tmp_path, "accumulator.json", body)
    assert _pairs(model)[("accumulate", "accumulate", "acc")].tag == "OK"
    assert model.loop_carried_verdict() == "CLEAN"


def test_a_tile_each_iteration_rewrites_before_reading_is_not_carried(tmp_path):
    def body(d, b):
        t, src = d.tile("t"), d.dram("src")
        psum = d.tile("psum", kind="PSUM")
        d.ins(b, "load_t", "SP", "DMACopy", reads=[d.whole(src)], writes=[d.whole(t)],
              queue=HWDGE_QUEUE, updates=[(40, 16)])
        d.ins(b, "use_t", "PE", "Matmult", reads=[d.whole(t)], writes=[d.whole(psum)],
              waits=[("load_t", 40, 16)], updates=[(41, 1)])
        d.ins(b, "late_t", "DVE", "TensorCopy", reads=[d.whole(psum)], writes=[d.whole(t)],
              waits=[("use_t", 41, 1)], updates=[(42, 1)])
        Dump.loop_end(b, BODY, EXIT, ENGINES, waits=[("late_t", 42, 1)], reset=(40, 41, 42))

    model = _single_loop(tmp_path, "rewritten.json", body)
    assert [k for k in _pairs(model) if k[2] == "t"] == []
    assert _pairs(model, "WAW")[("late_t", "load_t", "t")].tag == "BACK-EDGE-ONLY"
    assert model.loop_carried_verdict() == "CLEAN"


def test_a_loop_of_several_blocks_is_undecided(tmp_path):
    d = Dump()
    head = d.block("Block1_LoopBody_1")
    d.ins(head, "work", "DVE", "Memset", writes=[d.whole(d.tile("t"))])
    tail = d.block("Block1_LoopBody_1_tail")
    d.ins(tail, "back", "Pool", "UnconditionalBranch", target="Block1_LoopBody_1")
    model = depcheck.Model(str(d.write(tmp_path / "two_block_loop.json")))
    assert model.loops == {}
    assert model.loop_carried_verdict() == "UNDECIDED"
    assert any("Block1_LoopBody_1" in why for why in model.loop_undecided_reasons())


def test_a_dump_without_a_loop_is_clean_in_the_class(tmp_path):
    d = Dump()
    d.ins(d.block("Block1"), "work", "DVE", "Memset", writes=[d.whole(d.tile("t"))])
    model = depcheck.Model(str(d.write(tmp_path / "straight.json")))
    assert model.loops == {}
    assert model.loop_carried() == []
    assert model.loop_carried_verdict() == "CLEAN"


def test_the_command_line_prints_the_class_and_its_verdict(tmp_path, monkeypatch, capsys):
    v8d_pattern(tmp_path)
    monkeypatch.setattr(sys, "argv", ["depcheck", str(tmp_path / "v8d_pattern.json")])
    depcheck.main()
    out = capsys.readouterr().out
    assert "VERDICT: CLEAN" in out
    assert "[BACK-EDGE-ONLY] RAW refill_w DMACopy@SP/DMA:qSPDynamicHW" in out
    assert "VERDICT (loop-carried): FINDINGS" in out
