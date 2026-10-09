# SPDX-License-Identifier: Apache-2.0
"""The one-pass VERDICT line: a confirmed unordered pair is a finding, even in a dump where
another operand is undecided.

trn2-2's re-run of the v8d dumps (2026-10-09) printed ``VERDICT: UNDECIDED`` above four to
six SBUF RAW pairs that nothing orders, because runtime-address operands elsewhere in the
dump were undecided. An undecided operand only hides the pairs it takes part in; it adds or
removes no edge, so a pair the graph leaves unordered stays unordered.
"""

from __future__ import annotations

import sys

from test.kernel_depcheck import depcheck
from test.kernel_depcheck.tests.synthetic import Dump


def _verdict(tmp_path, monkeypatch, capsys, build):
    """The ``VERDICT:`` line ``depcheck.main()`` prints for the dump ``build(d, block)`` makes."""
    d = Dump()
    build(d, d.block("Block1"))
    path = d.write(tmp_path / "dump.json")
    monkeypatch.setattr(sys, "argv", ["depcheck", str(path)])
    depcheck.main()
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("VERDICT:")]
    assert len(lines) == 1, lines
    return lines[0]


def _runtime_dram(name):
    """A DRAM operand at a runtime address: the check takes it as its whole tensor."""
    return dict(kind="register_ap", memsetref=f"{name}_set", dtype="float32",
                ap=[[1, 1], [1, 4]])


def test_a_confirmed_pair_is_a_finding_beside_an_undecided_operand(tmp_path, monkeypatch,
                                                                  capsys):
    def build(d, b):
        t, other = d.tile("t"), d.tile("other", addr=4096)
        d.ins(b, "write", "DVE", "Memset", writes=[d.whole(t)])
        d.ins(b, "read", "Pool", "TensorCopy", reads=[d.whole(t)])
        d.ins(b, "at_runtime", "DVE", "TensorCopy", reads=[d.runtime(other)])

    line = _verdict(tmp_path, monkeypatch, capsys, build)
    assert line.startswith("VERDICT: FINDINGS (1 confirmed unsynchronized pair")
    assert "1 operand or edge undecided" in line


def test_pairs_only_through_a_runtime_address_leave_the_dump_undecided(tmp_path, monkeypatch,
                                                                      capsys):
    # The read's real address range is not known, so the pair may not overlap at all.
    def build(d, b):
        out = d.dram("out")
        d.ins(b, "store", "SP", "DMACopy", writes=[d.whole(out)], queue="q")
        d.ins(b, "gather", "Pool", "DMACopy", reads=[_runtime_dram(out)], queue="q2")

    line = _verdict(tmp_path, monkeypatch, capsys, build)
    assert line.startswith("VERDICT: UNDECIDED (0 confirmed unsynchronized pairs")
    assert "1 through a runtime-address operand" in line


def test_each_engine_saving_one_register_to_one_slot_is_not_a_finding(tmp_path, monkeypatch,
                                                                     capsys):
    # walrus saves a loop register from every engine to one SBUF slot: the same value.
    def build(d, b):
        slot = d.tile("reg_3_offset", partitions=1, columns=1)
        for engine in ("DVE", "Pool"):
            d.ins(b, f"I-7_inst__I-10-0-{engine}0", engine, "TensorSave",
                  writes=[d.whole(slot)])

    line = _verdict(tmp_path, monkeypatch, capsys, build)
    assert line.startswith("VERDICT: CLEAN (0 confirmed unsynchronized pairs")
    assert "1 same-value TensorSave" in line


def test_an_unordered_pair_is_not_confirmed_when_the_edges_are_unknown(tmp_path, monkeypatch,
                                                                      capsys):
    # A reset that names no group leaves every count after it unknown, so a wait may bind to
    # the wrong producer and an edge may be missing: no unordered pair is confirmed.
    def build(d, b):
        t = d.tile("t")
        d.ins(b, "reset", "ALL", "GroupResetSemaphores")
        d.ins(b, "write", "DVE", "Memset", writes=[d.whole(t)])
        d.ins(b, "read", "Pool", "TensorCopy", reads=[d.whole(t)])

    line = _verdict(tmp_path, monkeypatch, capsys, build)
    assert line.startswith("VERDICT: UNDECIDED (0 confirmed unsynchronized pairs")
    assert "edges unknown" in line
