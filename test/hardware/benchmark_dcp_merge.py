# SPDX-License-Identifier: Apache-2.0
"""Time the DCP log-sum-exp merge kernel at its served shapes on Neuron, against its roofline.

It runs on a Neuron device at LNC2 and selects no cores itself. It refuses, by name, to run
with no Neuron device visible, in the simulator or CPU mode, at another LNC, or with no
compile cache root.

Operator note: start it under your core-allocation mechanism, which gives the process its
cores and sets ``NEURON_LOGICAL_NC_CONFIG=2``. On this campaign's hosts that is ``devlease.py
slice <name>``::

    export NEURON_LIBTORCH_CACHE_ROOT=<an empty directory>
    python3 /home/ubuntu/glm53f-wt/devlease.py slice <name> -- python3 \\
        test/hardware/benchmark_dcp_merge.py --out bench.json

The kernel is ``vllm_neuron/functional/attention/dcp_merge.py`` (``dcp_lse_merge``): it merges
the CP ranks' float32 partials ``[CP, H, R, L]`` and lses ``[CP, H, R]`` into ``[R, H, L]``
bf16. It is new, so there is no base revision to time against; the bar is its roofline.

Geometry: L = 512 (the latent rank of GLM-5.3-Flash), R = 2048 rows (one 2048-token prefill
chunk), one head per rank (TP 64), CP 2, 4 and 8 (:data:`CASES`), plus CP 4 at two heads.
The operands are the merge tests' (``test_dcp_merge.make_partials``): partials of the cache's
magnitude, lses in [-5, 15], and a share of empty (rank, head, row) slots (partial 0, lse
``EMPTY_LSE``), so the device run also shows the empty rows stay NaN-free.

Per case: the seam is compiled and run on ``--seeds`` operand sets. Each output must be
finite, and is checked against the float64 llama3 merge within the tests' derived bound
(``merge_bound``) and exactly where the exact value is clear of a bf16 tie (the fp32 part of
the bound). That bound carries the simulator's measured exp error (``EXP_U``); the device's
exp is not measured here, so a miss is reported with its ratio rather than ending the run,
and the exit code is 1. The case is then timed in ``--reps`` repeats of ``--iterations``
calls. A sample is one dispatch with the output copied to CPU; its device time is the
runtime system trace's ``nc_exec_running`` interval with the two physical cores merged.

ROOFLINE (``docs/kernel_ledger.md`` section 3, ``reports/dcp_item4.md`` section 9):
``max(FLOPs / 79e12, bytes / 716e9, 2 us)``, FLOPs ``2 * CP * H * R * L`` (one multiply-add
per partial element) and bytes the float32 partials and lses read once and the bf16 output
written once. The bar is ``median <= 1.5 x roofline``; each case prints its ratio.

LNC2 LOOP PARITY (``reports/dcp_item4.md`` section 14): no site of the merge kernel, or of
the partial kernels that feed it, is exposed to the LNC2 hang class (the two cores of a pair
running different run-time trip counts around a barrier). The merge has no run-time loop or
branch: its loops are Python loops over trace-time ints (heads, the program's row tiles,
latent chunks, ranks), and each program's contiguous share of the row tiles is fixed at
trace time from ``program_id`` (at an odd tile count the first program takes one tile
more). Its compiled programs hold no barrier and no branch instruction on either core at
CP 2, 4 and 8, so neither core waits on the other. The script catches no exception: an
execution error the runtime raises ends the run before the next case.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import glob
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

#: The worktree root, ahead of any installed copy (the launcher sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

LATENT = 512
#: Rates as the inventory cites them (``glm53f-wt3/planner/configs/trn2_constants.json``
#: ``te_peak_flops``, ``hbm_bw_bytes_per_s``), and the ledger's launch floor.
TE_PEAK_FLOPS = 79e12
HBM_BYTES_PER_S = 716e9
FLOOR_US = 2.0
#: The bar of the DCP item 4 packet: at most 1.5 x the roofline.
BAR = 1.5
#: The share of (rank, head, row) slots that are empty in the checked operands.
EMPTY_FRACTION = 0.2


@dataclass(frozen=True)
class Case:
    """One merge call: ``cp`` ranks' partials of ``heads`` heads over ``rows`` rows."""

    name: str
    cp: int
    heads: int
    rows: int
    served: str


CASES = (
    Case("cp2_r2048", 2, 1, 2048, "CP 2, one head per rank, a 2048-row chunk"),
    Case("cp4_r2048", 4, 1, 2048, "CP 4, one head per rank, a 2048-row chunk"),
    Case("cp8_r2048", 8, 1, 2048, "CP 8, one head per rank, a 2048-row chunk"),
    Case("cp4_h2_r2048", 4, 2, 2048, "not served at TP 64: CP 4 at two heads per rank"),
)


def roofline(case: Case) -> dict:
    """The case's roofline in us, with its terms."""
    elements = case.cp * case.heads * case.rows * LATENT
    lses = case.cp * case.heads * case.rows
    nbytes = elements * 4 + lses * 4 + case.rows * case.heads * LATENT * 2
    flops = 2 * elements
    terms = {"FLOPs": flops / TE_PEAK_FLOPS * 1e6, "bytes": nbytes / HBM_BYTES_PER_S * 1e6,
             "floor": FLOOR_US}
    binds = max(terms, key=terms.get)
    bound = terms[binds]
    return {"bytes": nbytes, "flops": flops, "flop_us": terms["FLOPs"],
            "byte_us": terms["bytes"], "floor_us": FLOOR_US, "roofline_us": bound,
            "bar_us": BAR * bound, "binds": binds,
            "formula": "max(FLOPs / 79e12, bytes / 716e9, 2 us); FLOPs 2*CP*H*R*L; bytes "
                       "= f32 partials + f32 lses + bf16 output"}


def neuron_devices() -> list[str]:
    """The Neuron device nodes this process can see. None on a host without a device, or in
    a namespace that hides them."""
    return sorted(glob.glob("/dev/neuron[0-9]*"))


def md5_of(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def device_intervals(events_json: str) -> list[float]:
    """Device us of each execution, in execution order, the physical-core intervals merged."""
    starts, intervals = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            begin = starts.pop(key)
            intervals.setdefault(begin["data"]["exec_id"], []).append(
                (begin["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in spans) - min(begin for begin, _ in spans)) / 1e3
            for _, spans in sorted(intervals.items())]


def summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"n": len(ordered), "median_us": statistics.median(ordered),
            "min_us": ordered[0], "max_us": ordered[-1],
            "p90_us": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}


def time_calls(graph, inputs: tuple, iterations: int) -> dict:
    """One repeat: ``iterations`` calls, each output copied to CPU, timed by the system trace."""
    from nrtpy._nrtpy import SystemTraceSession

    with SystemTraceSession() as trace:
        for _ in range(iterations):
            graph(*inputs).to("cpu")
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != iterations:
        raise AssertionError(f"the trace has {len(device)} executions for {iterations} calls")
    return summary(device)


def run_case(case: Case, args) -> dict:
    import torch

    from test.vllm_neuron.functional.attention import test_dcp_merge as TM
    from vllm_neuron.functional.attention import dcp_merge as DM

    torch._dynamo.reset()
    graph = torch.compile(DM.dcp_lse_merge, backend="neuron_libtorch", fullgraph=True,
                          dynamic=False)
    numerics, timed_inputs = [], None
    for seed_index in range(args.seeds):
        seed = 1000 * seed_index + 10 * case.cp + case.heads
        partials, lse = TM.make_partials(case.cp, case.heads, case.rows, seed=seed,
                                         empty_fraction=EMPTY_FRACTION)
        inputs = (partials.to("neuron:0"), lse.to("neuron:0"))
        got = graph(*inputs).to("cpu")
        want = TM.llama3_merge_reference(partials, lse)
        bound = TM.merge_bound(partials, lse)
        err = (got.double() - want).abs()
        fp32_bound = TM.fp32_error_bound(partials, lse)
        clear = (TM.distance_to_a_bf16_tie(want) > fp32_bound) | (fp32_bound == 0)
        rne = want.float().to(torch.bfloat16)
        numerics.append({
            "seed": seed, "finite": bool(torch.isfinite(got.float()).all()),
            "within_bound": bool((err <= bound).all()),
            "max_err_over_bound": float((err / bound.clamp_min(1e-300)).max()),
            "clear_of_a_tie": float(clear.double().mean()),
            "differ_from_rne_clear_of_a_tie": int(((got != rne) & clear).sum()),
            "empty_slots": int((lse == DM.EMPTY_LSE).sum())})
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(got, args.save_dir / f"{case.name}_seed{seed}.pt")
        if timed_inputs is None:
            timed_inputs = inputs
    for _ in range(args.warmup):
        graph(*timed_inputs).to("cpu")
    reps = [time_calls(graph, timed_inputs, args.iterations) for _ in range(args.reps)]
    medians = [r["median_us"] for r in reps]
    median = statistics.median(medians)
    budget = roofline(case)
    ratio = median / budget["roofline_us"]
    timing = {"rep_medians_us": medians, "median_us": median,
              "rep_spread": (max(medians) - min(medians)) / median,
              "over_roofline": ratio, "meets_bar": ratio <= BAR}
    numerics_ok = all(n["finite"] and n["within_bound"]
                      and n["differ_from_rne_clear_of_a_tie"] == 0 for n in numerics)
    return {"case": asdict(case), "roofline": budget, "timing": timing, "reps": reps,
            "numerics": numerics, "numerics_ok": numerics_ok,
            "programs": DM.merge_programs(case.rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES],
                        help="case name (repeatable; default: every case)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20, help="calls per repeat")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3, help="operand sets checked per case")
    parser.add_argument("--save-dir", type=Path, help="keep every checked output as .pt here")
    args = parser.parse_args()
    if args.reps < 1 or args.iterations < 1 or args.warmup < 0 or args.seeds < 1:
        parser.error("use reps, iterations and seeds >= 1 and warmup >= 0")
    if not neuron_devices():
        raise SystemExit("hardware benchmark: no Neuron device is visible (no /dev/neuron* "
                         "node); run it on a Neuron host, on the cores your allocation gives it")
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise SystemExit("hardware benchmark: unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit("the served geometry is LNC2: NEURON_LOGICAL_NC_CONFIG must be 2")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise SystemExit("set NEURON_LIBTORCH_CACHE_ROOT to a compile cache directory")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    from vllm_neuron.functional.attention import dcp_merge as DM

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    modules = {"vllm_neuron": {"path": vllm_neuron.__file__, "md5": md5_of(vllm_neuron.__file__)},
               "dcp_merge": {"path": DM.__file__, "md5": md5_of(DM.__file__)}}
    print(f"vllm_neuron.__file__ {modules['vllm_neuron']['path']} "
          f"{modules['vllm_neuron']['md5']}", flush=True)
    print(f"dcp_merge {modules['dcp_merge']['path']} {modules['dcp_merge']['md5']}", flush=True)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_dir is not None:
        args.save_dir = args.save_dir.resolve()
    # The compiler writes its logs to the working directory: keep them with the cache.
    scratch = Path(cache_root) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "neuron_devices": len(neuron_devices()),
        "tree": str(ROOT),
        "modules": modules,
        "shape": {"latent": LATENT, "partials": "float32", "lse": "float32",
                  "out": "bfloat16", "empty_fraction": EMPTY_FRACTION},
        "rates": {"te_peak_flops": TE_PEAK_FLOPS, "hbm_bytes_per_s": HBM_BYTES_PER_S,
                  "floor_us": FLOOR_US, "bar": BAR,
                  "source": "glm53f-wt3/planner/configs/trn2_constants.json; "
                            "docs/kernel_ledger.md section 3"},
        "timing_method": {"reps": args.reps, "iterations": args.iterations,
                          "warmup": args.warmup,
                          "device": "system trace nc_exec_running, physical cores merged",
                          "synchronization": "output copied to CPU on every call",
                          "noise_floor": "(max - min) / median of the rep medians"},
        "cases": [],
    }
    for case in CASES:
        if args.case and case.name not in args.case:
            continue
        row = run_case(case, args)
        report["cases"].append(row)
        out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({"case": case.name,
                          "median_us": round(row["timing"]["median_us"], 2),
                          "roofline_us": round(row["roofline"]["roofline_us"], 2),
                          "over_roofline": round(row["timing"]["over_roofline"], 3),
                          "meets_bar": row["timing"]["meets_bar"],
                          "noise": round(row["timing"]["rep_spread"], 4),
                          "numerics_ok": row["numerics_ok"],
                          "max_err_over_bound": max(n["max_err_over_bound"]
                                                    for n in row["numerics"])}), flush=True)
    return 0 if all(row["numerics_ok"] for row in report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
