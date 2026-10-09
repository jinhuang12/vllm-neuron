# SPDX-License-Identifier: Apache-2.0
"""The one-pass VERDICT line: a confirmed unordered pair is a finding, even in a dump where
another operand is undecided.

trn2-2's re-run of the v8d dumps (2026-10-09) printed ``VERDICT: UNDECIDED`` above four to
six SBUF RAW pairs that nothing orders, because runtime-address operands elsewhere in the
dump were undecided. An undecided operand only hides the pairs it takes part in; it adds or
removes no edge, so a pair the graph leaves unordered stays unordered.

The line's counts come in fixed fields, so trn2-2's verdicts.json columns are filled by
``depcheck.VERDICT_LINE``; every test here reads the line through it.
"""

from __future__ import annotations

import collections
import sys

from test.kernel_depcheck import depcheck
from test.kernel_depcheck.tests.synthetic import Dump


def _verdict(tmp_path, monkeypatch, capsys, build):
    """The fields of the ``VERDICT:`` line ``depcheck.main()`` prints for the dump
    ``build(d, block)`` makes: the word, and every count as an int."""
    d = Dump()
    build(d, d.block("Block1"))
    path = d.write(tmp_path / "dump.json")
    monkeypatch.setattr(sys, "argv", ["depcheck", str(path)])
    depcheck.main()
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("VERDICT:")]
    assert len(lines) == 1, lines
    match = depcheck.VERDICT_LINE.match(lines[0])
    assert match, lines[0]
    fields = {k: v if k == "word" else int(v) for k, v in match.groupdict().items()}
    assert fields["undecided"] == sum(fields[k] for k in (
        "runtime_sb_psum_operand", "other_operand", "cycle", "ungrouped_reset"))
    return fields


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

    v = _verdict(tmp_path, monkeypatch, capsys, build)
    assert (v["word"], v["confirmed"], v["sb_raw"]) == ("FINDINGS", 1, 1)
    assert (v["undecided"], v["runtime_sb_psum_operand"]) == (1, 1)


def test_pairs_only_through_a_runtime_address_leave_the_dump_undecided(tmp_path, monkeypatch,
                                                                      capsys):
    # The read's real address range is not known, so the pair may not overlap at all.
    def build(d, b):
        out = d.dram("out")
        d.ins(b, "store", "SP", "DMACopy", writes=[d.whole(out)], queue="q")
        d.ins(b, "gather", "Pool", "DMACopy", reads=[_runtime_dram(out)], queue="q2")

    v = _verdict(tmp_path, monkeypatch, capsys, build)
    assert (v["word"], v["confirmed"], v["runtime_address"]) == ("UNDECIDED", 0, 1)


def test_each_engine_saving_one_register_to_one_slot_is_not_a_finding(tmp_path, monkeypatch,
                                                                     capsys):
    # walrus saves a loop register from every engine to one SBUF slot: the same value.
    def build(d, b):
        slot = d.tile("reg_3_offset", partitions=1, columns=1)
        for engine in ("DVE", "Pool"):
            d.ins(b, f"I-7_inst__I-10-0-{engine}0", engine, "TensorSave",
                  writes=[d.whole(slot)])

    v = _verdict(tmp_path, monkeypatch, capsys, build)
    assert (v["word"], v["confirmed"], v["same_value_tensorsave"]) == ("CLEAN", 0, 1)


def test_an_unordered_pair_is_not_confirmed_when_the_edges_are_unknown(tmp_path, monkeypatch,
                                                                      capsys):
    # A reset that names no group leaves every count after it unknown, so a wait may bind to
    # the wrong producer and an edge may be missing: no unordered pair is confirmed.
    def build(d, b):
        t = d.tile("t")
        d.ins(b, "reset", "ALL", "GroupResetSemaphores")
        d.ins(b, "write", "DVE", "Memset", writes=[d.whole(t)])
        d.ins(b, "read", "Pool", "TensorCopy", reads=[d.whole(t)])

    v = _verdict(tmp_path, monkeypatch, capsys, build)
    assert (v["word"], v["confirmed"], v["edges_unknown"]) == ("UNDECIDED", 0, 1)
    assert v["ungrouped_reset"] == 1


def test_the_verdict_line_parses_back_into_every_count():
    labels = ("confirmed", *depcheck.CONFIRMED_BY_SPACE, "same-value-TensorSave",
              "runtime-address", "edges-unknown", "undecided", "runtime-SB-PSUM-operand",
              "other-operand", "cycle", "ungrouped-reset", "waits-not-OK")
    counts = collections.Counter({label: k for k, label in enumerate(labels, 1)})
    match = depcheck.VERDICT_LINE.match(depcheck.verdict_line("FINDINGS", counts))
    assert match["word"] == "FINDINGS"
    assert [int(match[label.lower().replace("-", "_")]) for label in labels] == list(
        range(1, len(labels) + 1))
