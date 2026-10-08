# SPDX-License-Identifier: Apache-2.0
"""Per-op device time of the DSA selection graphs, from their ``neuron-explorer`` profiles.

Input: a record of ``indexer_shard_device.py --profile-dir`` (its ``profiles`` entries name
the ingested sessions). Output: for every profiled graph, the device time of each op in each
execution, as JSON::

    python test/vllm_neuron/functional/dsa/indexer_shard_profile.py RECORD.json --output OPS.json

It needs ``duckdb`` and no torch, so it runs under any python that has duckdb; it never
touches the device.

How an execution divides into ops. At LNC2 the two physical cores meet at a core barrier
(``compiler_opcode = PSEUDO_CORE_BARRIER``) at the start and end of the program block and
between consecutive custom calls, so the barriers cut an execution into segments. A barrier
is complete when its last instruction ends. Every instruction that starts in a segment is
charged to it, and the segment is named by the op whose instructions took the most
instruction time in it:

* an NKI kernel, by its ``nki_source_location``: the file name, or ``file.kernel`` when the
  file defines more than one ``@nki.jit`` kernel (``causal_bound.py`` holds the bound and
  the sentinel), or the package name for a vendored kernel (``rotational_topk``);
* a compiler op (traced torch), by its BIR instruction group (``I-55`` for ``I-55-0``): an
  instruction with a BIR name and no NKI source location;
* the barriers themselves, semaphore waits and the unnamed program preamble and epilogue
  instructions are bookkeeping and name no segment.

The program preamble before the first barrier and the epilogue after the last one are two
more segments, named ``preamble`` and ``epilogue`` whatever runs in them (the program start
and the output copies), so the segment times of one execution add up to the execution, and every
op's time is exclusive. An op whose instructions all overlap a longer op's segment (a
kernel the compiler runs beside the next one) gets no time of its own: the segment table
keeps every op found in each segment with its instruction time, so it stays visible there,
and its cost in the chain is the difference of two graphs (``sharded - precut`` for the
row cut).
"""

from __future__ import annotations

import argparse
import ast
import bisect
import glob
import json
from pathlib import Path
import statistics

#: Marks the instructions of a core barrier, which cut an execution into segments.
CORE_BARRIER = "PSEUDO_CORE_BARRIER"
#: The path component under which a vendored kernel package sits.
VENDORED = "/vendored_kernels/"
#: Names of the segments before the first and after the last barrier.
PREAMBLE, EPILOGUE = "preamble", "epilogue"


def session_dir(explorer_data: str, display_name: str) -> str:
    """The ingested session of ``display_name`` that holds executions (logical core 0)."""
    pattern = f"{explorer_data}/profiles/global/{display_name}_*_session_*@latest"
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(f"no ingested session matches {pattern}")
    return found[0]


def _kernels_by_line(path: str) -> list[tuple[int, int, str]] | None:
    """``(first line, last line, name)`` of each ``@nki.jit`` function in ``path``."""
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return None
    kernels = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
                "jit" in ast.unparse(d) for d in node.decorator_list):
            kernels.append((node.lineno, node.end_lineno, node.name))
    return kernels


class KernelNamer:
    """Names an NKI instruction by its source location (see the module docstring)."""

    def __init__(self):
        self._files: dict[str, list[tuple[int, int, str]] | None] = {}

    def __call__(self, location: str) -> str:
        path, _, line = location.rpartition(":")
        if VENDORED in path:
            return path.split(VENDORED, 1)[1].split("/", 1)[0]
        name = Path(path).name
        if path not in self._files:
            self._files[path] = _kernels_by_line(path)
        kernels = self._files[path]
        if not kernels or len(kernels) < 2 or not line.isdigit():
            return name
        for first, last, kernel in kernels:
            if first <= int(line) <= last:
                return f"{name}.{kernel}"
        return name


def _group(bir_name: str) -> str:
    """``I-55-0`` and ``I-124-0-tc-act`` -> ``I-55``, ``I-124``."""
    parts = bir_name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 and parts[0] == "I" else bir_name


def analyse_session(path: str) -> dict:
    """Segments, op times and compiler-op instructions of every execution in ``path``."""
    import duckdb

    con = duckdb.connect()
    table = f"'{path}/Instruction.parquet'"
    executions = con.sql(
        f"SELECT execution_index, execution_start_ts, execution_end_ts "
        f"FROM '{path}/ExecutionInfo.parquet' ORDER BY 2").fetchall()
    namer = KernelNamer()
    out = []
    for index, begin, end in executions:
        mine = con.sql(
            f"SELECT start_ts, end_ts, duration_ns, pcore_idx, engine, opcode, compiler_opcode, "
            f"bir_instruction_name, nki_source_location, hbm_read_bytes, hbm_write_bytes "
            f"FROM {table} WHERE start_ts >= {begin} AND end_ts <= {end} "
            f"ORDER BY start_ts").fetchall()
        barriers: dict[str, int] = {}
        for r in mine:
            if r[6] == CORE_BARRIER:
                barriers[r[7]] = max(barriers.get(r[7], 0), r[1])
        # The program preamble before the first barrier and the epilogue after the last one
        # are segments too, so the segments cover the whole execution. A graph without a
        # custom call (the launch graph) has no barrier and is one segment.
        barrier_ends = sorted(barriers.values())
        cuts = [begin, *barrier_ends, end]
        segments = [{"start_us": (cuts[i] - begin) / 1000.0,
                     "length_us": (cuts[i + 1] - cuts[i]) / 1000.0, "ops": {}}
                    for i in range(len(cuts) - 1)]
        compiler: dict[str, dict] = {}
        for start, _end, duration, _core, engine, opcode, cop, bir, nki, hbm_in, hbm_out in mine:
            if cop == CORE_BARRIER:
                continue
            if nki:
                op = namer(nki)
            elif bir and bir.startswith("I-") and "-wait-" not in bir:
                op = f"compiler {_group(bir)}"
                entry = compiler.setdefault(op, {"opcodes": {}, "instructions": 0,
                                                 "instruction_us": 0.0, "hbm_read_bytes": 0,
                                                 "hbm_write_bytes": 0, "first_us": None})
                entry["opcodes"][f"{engine}:{opcode}"] = entry["opcodes"].get(
                    f"{engine}:{opcode}", 0) + 1
                entry["instructions"] += 1
                entry["instruction_us"] += duration / 1000.0
                entry["hbm_read_bytes"] += hbm_in or 0
                entry["hbm_write_bytes"] += hbm_out or 0
                if entry["first_us"] is None:
                    entry["first_us"] = (start - begin) / 1000.0
            else:
                continue
            at = bisect.bisect_right(cuts, start) - 1
            if 0 <= at < len(segments):
                ops = segments[at]["ops"]
                ops[op] = ops.get(op, 0.0) + duration / 1000.0
        by_op: dict[str, float] = {}
        for seg in segments:
            seg["op"] = max(seg["ops"], key=lambda k: (seg["ops"][k], k)) if seg["ops"] else None
        if len(segments) >= 2:
            segments[0]["op"], segments[-1]["op"] = PREAMBLE, EPILOGUE
        for seg in segments:
            if seg["op"] is not None:
                by_op[seg["op"]] = by_op.get(seg["op"], 0.0) + seg["length_us"]
        for name, entry in compiler.items():
            entry["segment_op"] = next(
                (s["op"] for s in segments if name in s["ops"]), None)
        out.append({"execution": index, "execution_us": (end - begin) / 1000.0,
                    "block_us": ((barrier_ends[-1] - barrier_ends[0]) if barrier_ends
                                 else (end - begin)) / 1000.0,
                    "op_us": by_op, "segments": segments, "compiler_ops": compiler})
    return {"session": path, "executions": out}


def summarise(executions: list[dict]) -> dict:
    """Median over the executions of each op's time and of the execution."""
    ops = sorted({op for e in executions for op in e["op_us"]})
    return {"executions": len(executions),
            "execution_us": statistics.median(e["execution_us"] for e in executions),
            "block_us": statistics.median(e["block_us"] for e in executions),
            "op_us": {op: statistics.median(e["op_us"].get(op, 0.0) for e in executions)
                      for op in ops}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", type=Path, help="a record of indexer_shard_device.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = json.loads(args.record.read_text())
    if "profiles" not in record:
        raise SystemExit(f"{args.record} has no profiles; run the device script with "
                         f"--profile-dir")
    result = {"record": str(args.record), "label": record["label"], "cands": record["cands"],
              "causal_bound": record["causal_bound"], "graphs": {}}
    for graph, entry in record["profiles"].items():
        session = analyse_session(session_dir(entry["explorer_data"], entry["display_name"]))
        result["graphs"][graph] = {"summary": summarise(session["executions"]), **session}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for graph, body in result["graphs"].items():
        summary = body["summary"]
        print(graph, round(summary["execution_us"], 1),
              {op: round(us, 1) for op, us in summary["op_us"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
