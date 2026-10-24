# SPDX-License-Identifier: Apache-2.0
"""Reduce the timed ``vllm bench serve`` result files of the 8K, 8-user cell to the steady TPOT.

Reads the ``--save-detailed`` result files of one cell, grouped by their ``process``
and ``run`` metadata: two processes of three timed runs. Per request, the mean of
``itls`` with the first gap dropped; per run, the median over the requests; per
process, the median of its runs; the value of record is the mean of the two process
values, read against the bar, ``--bar-ms`` (default the registered bar). Guard: per
request, the count of gaps after the first above 2x that request's median gap.

Standard library only. Prints one JSON record. Exit 0 pass, 1 fail, 2 invalid run,
3 not adjudicable (a nonzero guard count, or a spread above 10%).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

BAR_MS = 42.12
OUTPUT_LEN = 128
PROCESSES = 2
RUNS = 3
MAX_SPREAD = 0.10
PASS, FAIL, INVALID, NOT_ADJUDICABLE = 0, 1, 2, 3


def percentile(values, q):
    """Return the q-th percentile with linear interpolation between ranks."""
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def validity(result):
    """Return the reasons a result file is not a valid run; empty when valid."""
    n = result.get("num_prompts")
    reasons = []
    if result.get("failed", 0) > 0:
        reasons.append(f"failed {result['failed']}")
    if result.get("completed", 0) < n:
        reasons.append(f"completed {result.get('completed')} below {n}")
    itls, out_lens = result.get("itls", []), result.get("output_lens", [])
    in_lens = result.get("input_lens", [])
    if len(itls) != n or len(out_lens) != n or len(in_lens) != n:
        reasons.append(f"{len(itls)} itls lists, {len(out_lens)} output_lens, "
                       f"{len(in_lens)} input_lens for {n} prompts")
    bad_out = [i for i, v in enumerate(out_lens) if v != OUTPUT_LEN]
    if bad_out:
        reasons.append(f"output_lens other than {OUTPUT_LEN} at requests {bad_out}")
    bad_itl = [i for i, g in enumerate(itls) if len(g) != OUTPUT_LEN - 1]
    if bad_itl:
        reasons.append(f"itls length other than {OUTPUT_LEN - 1} at requests {bad_itl}")
    if len(set(in_lens)) > 1:
        reasons.append(f"input_lens differ: {sorted(set(in_lens))}")
    return reasons


def request_values(gaps_s):
    """Return (steady mean in ms, guard count) of one request's ITL list in seconds."""
    steady = [g * 1000.0 for g in gaps_s[1:]]
    median = statistics.median(steady)
    return statistics.fmean(steady), sum(1 for g in steady if g > 2.0 * median)


def reduce(results, bar_ms=BAR_MS):
    """Return (exit code, record) for the result files of one cell against bar_ms."""
    record = {"bar_ms": round(bar_ms, 2)}
    runs = {}
    invalid = {}
    for name, r in results:
        key = (str(r.get("process")), str(r.get("run")))
        if key in runs:
            invalid[name] = [f"process {key[0]} run {key[1]} appears twice"]
            continue
        runs[key] = (name, r)
        reasons = validity(r)
        if reasons:
            invalid[name] = reasons
    processes = sorted({p for p, _ in runs})
    shape = {p: sorted(k for q, k in runs if q == p) for p in processes}
    if len(processes) != PROCESSES or any(len(v) != RUNS for v in shape.values()):
        invalid["cell"] = [f"expected {PROCESSES} processes of {RUNS} runs, got {shape}"]
    if invalid:
        record.update(result="invalid run", invalid=invalid)
        return INVALID, record

    per_run, guards, diagnostics = {}, {}, {}
    for (p, k), (name, r) in sorted(runs.items()):
        values = [request_values(g) for g in r["itls"]]
        label = f"{p}/{k}"
        per_run[label] = {"requests_ms": [v for v, _ in values],
                          "median_ms": statistics.median(v for v, _ in values)}
        guards[label] = [c for _, c in values]
        all_gaps = [g * 1000.0 for gaps in r["itls"] for g in gaps]
        diagnostics[label] = {
            "file": name,
            "bench_median_tpot_ms": r.get("median_tpot_ms"),
            "ttfts_ms": [t * 1000.0 for t in r.get("ttfts", [])],
            "itl_max_ms": max(all_gaps), "itl_median_ms": statistics.median(all_gaps),
            "itl_p99_ms": percentile(all_gaps, 99),
        }
    process_ms = {p: statistics.median(per_run[f"{p}/{k}"]["median_ms"] for k in shape[p])
                  for p in processes}
    value = statistics.fmean(process_ms.values())
    a, b = (process_ms[p] for p in processes)
    spread = abs(a - b) / value
    guard_total = sum(sum(c) for c in guards.values())
    record.update(value_of_record_ms=value, process_ms=process_ms, spread=spread,
                  guard_counts=guards, guard_total=guard_total, runs=per_run,
                  diagnostics=diagnostics,
                  per_replica_split="read from each engine's log, not from these files")
    if guard_total or spread > MAX_SPREAD:
        record["result"] = "not adjudicable"
        return NOT_ADJUDICABLE, record
    record["result"] = "pass" if value <= bar_ms else "fail"
    return (PASS if value <= bar_ms else FAIL), record


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="result files, or folders of *.json result files")
    ap.add_argument("--bar-ms", type=float, default=BAR_MS,
                    help=f"the bar in ms (default {BAR_MS}, the registered bar)")
    args = ap.parse_args(argv)
    files = []
    for p in map(Path, args.paths):
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    results = [(str(f), json.loads(f.read_text(encoding="utf-8"))) for f in files]
    code, record = reduce(results, args.bar_ms)
    print(json.dumps(record))
    return code


if __name__ == "__main__":
    sys.exit(main())
