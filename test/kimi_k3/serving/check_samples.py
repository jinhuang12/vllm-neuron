# SPDX-License-Identifier: Apache-2.0
"""Check that a candidate lm-eval run sent the reference run's requests, item by item.

Reads the ``samples_<task>_<date>.jsonl`` logs of one benchmark under the candidate
and reference output folders, pairs the entries of each task by ``doc_id`` and
``filter``, and compares ``arguments`` and ``prompt_hash``. With ``--limit N`` only the
reference entries with ``doc_id`` below N are read, and their mean ``exact_match`` on
the ``flexible-extract`` filter is printed in points.

Standard library only. Exit 0 when every pair is equal and no entry is missing; exit
2, the candidate run invalid, on any differing or missing entry.
"""

import argparse
import json
import re
import sys
from pathlib import Path

_SAMPLES = re.compile(r"^samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T[\d.\-]+\.jsonl$")
INVALID = 2


class SampleError(Exception):
    """The sample logs cannot be paired."""


def read_logs(root, limit=None):
    """Return {task: {(doc_id, filter): entry}} for the sample logs under ``root``."""
    logs = {}
    for path in sorted(Path(root).rglob("samples_*.jsonl")):
        m = _SAMPLES.match(path.name)
        if not m:
            continue
        task = m.group("task")
        if task in logs:
            raise SampleError(f"{root}: two sample logs for task {task}")
        entries = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if limit is not None and e["doc_id"] >= limit:
                continue
            key = (e["doc_id"], e["filter"])
            if key in entries:
                raise SampleError(f"{path}: doc_id {key[0]} filter {key[1]} appears twice")
            entries[key] = e
        logs[task] = entries
    if not logs:
        raise SampleError(f"{root}: no sample logs")
    return logs


def compare(candidate, reference):
    """Return (pairs compared, first difference or None)."""
    pairs = 0
    for task in sorted(set(candidate) | set(reference)):
        if task not in candidate:
            return pairs, {"task": task, "doc_id": None, "reason": "task missing in candidate"}
        if task not in reference:
            return pairs, {"task": task, "doc_id": None, "reason": "task missing in reference"}
        cand, ref = candidate[task], reference[task]
        for key in sorted(set(cand) | set(ref), key=lambda k: (k[0], k[1])):
            doc_id, filt = key
            if key not in cand:
                return pairs, {"task": task, "doc_id": doc_id, "filter": filt,
                               "reason": "missing in candidate"}
            if key not in ref:
                return pairs, {"task": task, "doc_id": doc_id, "filter": filt,
                               "reason": "missing in reference"}
            for field in ("arguments", "prompt_hash"):
                if cand[key].get(field) != ref[key].get(field):
                    return pairs, {"task": task, "doc_id": doc_id, "filter": filt,
                                   "reason": f"{field} differs"}
            pairs += 1
    return pairs, None


def flexible_mean(logs):
    """Return the mean ``exact_match`` of the ``flexible-extract`` entries, in points."""
    values = [e["exact_match"] for entries in logs.values()
              for (_, filt), e in entries.items() if filt == "flexible-extract"]
    return 100.0 * sum(values) / len(values) if values else None


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True, help="the candidate's lm-eval output folder")
    ap.add_argument("--reference", required=True, help="the reference's lm-eval output folder")
    ap.add_argument("--limit", type=int, default=None,
                    help="read only the reference entries with doc_id below this")
    args = ap.parse_args(argv)
    report = {"candidate": args.candidate, "reference": args.reference, "limit": args.limit}
    try:
        reference = read_logs(args.reference, args.limit)
        candidate = read_logs(args.candidate)
    except SampleError as e:
        report.update(result="invalid", reason=str(e))
        print(json.dumps(report))
        return INVALID
    if args.limit is not None:
        report["reference_mean_flexible_extract"] = flexible_mean(reference)
    pairs, diff = compare(candidate, reference)
    report["pairs"] = pairs
    report["result"] = "equal" if diff is None else "invalid"
    if diff is not None:
        report["first_difference"] = diff
    print(json.dumps(report))
    return 0 if diff is None else INVALID


if __name__ == "__main__":
    sys.exit(main())
