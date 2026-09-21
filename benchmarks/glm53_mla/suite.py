"""Run the fixed native shape, dtype, sentinel and tile-size checks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {
        "decode-c4096": [],
        "decode-c16384": ["--cache-rows", "16384"],
        "prefill-s16": ["--seq", "16"],
        "f16-rope32": ["--seq", "2", "--heads", "8", "--latent", "256", "--topk", "640",
                        "--cache-rows", "1024", "--rope", "32", "--dtype", "float16"],
        "bf16-sentinel": ["--seq", "3", "--heads", "2", "--latent", "128", "--topk", "768",
                           "--cache-rows", "777", "--kind", "sentinel"],
        "fp32-duplicates": ["--seq", "2", "--heads", "16", "--latent", "384", "--topk", "1152",
                            "--cache-rows", "2048", "--kind", "duplicates", "--dtype", "float32"],
        "bf16-heads128": ["--heads", "128", "--latent", "128", "--topk", "640",
                           "--cache-rows", "1024"],
        "fp32-rope7": ["--heads", "3", "--latent", "256", "--topk", "640",
                        "--cache-rows", "1024", "--rope", "7", "--dtype", "float32"],
        "staged": ["--staged"],
        "tile128": ["--block-n", "128"],
        "tile256": ["--block-n", "256"],
        "tile384": ["--block-n", "384"],
    }
    report = {}
    for label, options in cases.items():
        command = [sys.executable, "-m", "benchmarks.glm53_mla.run", "--output",
                   str(args.output / label), "--baseline-source", str(args.baseline_source), *options]
        with (args.output / f"{label}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = args.output / label / "metrics.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        report[label] = {"command": command, "returncode": result.returncode,
                         "pass": result.returncode == 0 and record.get("pass", False),
                         "accuracy": record.get("accuracy"), "timing": record.get("timing"),
                         "canonical": record.get("canonical")}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(label, report[label]["pass"], record.get("timing", {}).get("mean_us"), flush=True)
    return 0 if all(value["pass"] for value in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
