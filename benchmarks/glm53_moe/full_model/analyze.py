#!/usr/bin/env python3
"""Compare all declared cohorts. Bootstrap whole cohorts, never tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


METRICS = ("output_throughput", "mean_e2el_ms", "p99_e2el_ms",
           "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms")


def read_arm(root):
    root = Path(root)
    complete = json.loads((root / "complete.json").read_text())
    assert complete["cohorts"] == 5 and complete["measured_requests"] == 50
    cohorts, hashes = [], {}
    for i in range(5):
        path = root / f"cohort-{i:02}.json"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        row = json.loads(path.read_text())
        assert row["completed"] == 10 and row["failed"] == 0
        assert row["total_output_tokens"] == 320 and row["output_lens"] == [32] * 10
        assert not any(row["errors"])
        cohorts.append(row)
    return {
        "workload_sha256": complete["workload_sha256"], "files_sha256": hashes,
        "cohort_values": {key: [c[key] for c in cohorts] for key in METRICS},
        "mean_of_cohorts": {key: statistics.mean(c[key] for c in cohorts) for key in METRICS},
        "pooled_output_tokens_per_second": 1600 / sum(c["duration"] for c in cohorts),
        "input_lengths": cohorts[0]["input_lens"],
        "all_input_lengths_equal": all(c["input_lens"] == cohorts[0]["input_lens"] for c in cohorts),
    }


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def compare(a, b, post):
    assert a["workload_sha256"] == b["workload_sha256"] == post["workload_sha256"]
    assert all(arm["all_input_lengths_equal"] for arm in (a, b, post))
    assert a["input_lengths"] == b["input_lengths"] == post["input_lengths"]
    rng = random.Random(0)
    av, bv = (arm["cohort_values"]["output_throughput"] for arm in (a, b))
    deltas = []
    for _ in range(10000):
        aa = statistics.mean(rng.choices(av, k=5))
        bb = statistics.mean(rng.choices(bv, k=5))
        deltas.append(100 * (bb / aa - 1))
    am, bm, pm = (arm["mean_of_cohorts"] for arm in (a, b, post))
    ci = [percentile(deltas, .025), percentile(deltas, .975)]
    drift = 100 * (pm["output_throughput"] / am["output_throughput"] - 1)
    gates = {
        "otps_gain_ci_positive": ci[0] > 0,
        "mean_e2e_improves": bm["mean_e2el_ms"] < am["mean_e2el_ms"],
        "p99_e2e_improves": bm["p99_e2el_ms"] < am["p99_e2el_ms"],
        "mean_ttft_within_5_percent": bm["mean_ttft_ms"] <= 1.05 * am["mean_ttft_ms"],
        "p99_ttft_within_5_percent": bm["p99_ttft_ms"] <= 1.05 * am["p99_ttft_ms"],
        "baseline_drift_within_5_percent": abs(drift) <= 5,
    }
    return {"baseline": a, "candidate": b, "postbaseline": post,
            "otps_improvement_percent": 100 * (bm["output_throughput"] / am["output_throughput"] - 1),
            "otps_improvement_percent_95_ci": ci,
            "e2e_latency_reduction_percent": 100 * (1 - bm["mean_e2el_ms"] / am["mean_e2el_ms"]),
            "baseline_drift_percent": drift, "performance_gates": gates,
            "performance_pass": all(gates.values()),
            "limits": "C1 warm repeated prompts. Correctness, runtime identity, dispatch, compile exclusion, and profile attribution require separate evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "postbaseline", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = compare(*(read_arm(getattr(args, name)) for name in
                       ("baseline", "candidate", "postbaseline")))
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("baseline", "candidate", "postbaseline")}, indent=2))
    if not result["performance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
