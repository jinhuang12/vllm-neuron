# SPDX-License-Identifier: Apache-2.0
"""``python -m test.kernel_ledger``: the GLM-5.3-Flash decode kernel ledger.

    PYTHONPATH=$PWD python -m test.kernel_ledger --tip 5938748 --bs 1 --ctx 1024
    PYTHONPATH=$PWD python -m test.kernel_ledger --tip current --bs 1 --ctx 1024
    PYTHONPATH=$PWD python -m test.kernel_ledger --tip current --bs 64 --ctx 8192
    PYTHONPATH=$PWD python -m test.kernel_ledger --emit-shapes

No device time: reads the team's benchmark and gate JSONs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .models.glm53f.calibration import Calibration, calibrate, calibrated_buckets, calibrated_ms
from .models.glm53f.configs import BASELINE, BS1_CTX1K, BS64_CTX8K, FAMILY_OF, NOT_WIRED, DecodePoint, point_for
from .models.glm53f.ledger import Ledger, build_ledger
from .models.glm53f.profile import profile_buckets
from .models.glm53f.references import GATE_BASELINE_LABEL, RECONCILED, REFERENCES, STEP_5938748, TOLERANCE
from .models.glm53f.shapes import emit_shapes
from .models.glm53f.tips import Tip, resolve_tip
from .readers.gate import GateStep, read_gates
from .readers.micro import MICRO_FILES, REPORTS_DIR, load_micro_results, mhc_batch_files, missing_micro_files

#: The point every gate run serves: one request, context about 1k, max_model_len 4096.
GATE_POINT = BS1_CTX1K
PREDICTOR = "PREDICTOR, not a test"


def _num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0


def _kernel_set_line(tip: Tip) -> str:
    ks = tip.kernel_set
    if tip.name == "current":
        what = 'every wave-1 kernel ("after" medians)'
    elif not ks.after:
        what = 'every unit at its 5938748 kernel ("before" medians)'
    else:
        what = f'wave-1 kernels ("after") for {", ".join(sorted(ks.after))}; 5938748 ("before") for the rest'
    line = f"kernel set {ks.name}: {what}"
    if tip.branches:
        line += f"; gated branches in this tree: {', '.join(tip.branches)}"
    wired = [k for k in NOT_WIRED if ks.is_after(FAMILY_OF[k])]
    if wired:
        line += f"; not wired, stays 5938748: {', '.join(wired)}"
    return line


def node_table(led: Ledger) -> List[str]:
    out = ["Per node (us per call; step ms = us x count; roofline = max(FLOPs / 79 TFLOP/s, bytes / 716 GB/s, 2 us))",
           f"{'node':<24} {'bucket':<11} {'kind':<10} {'count':>5} {'roofline_us':>11} {'measured_us':>11} "
           f"{'step_ms':>8} {'variant':<7} source / note"]
    for r in led.rows:
        src = r.source if not r.note else f"{r.source}; {r.note}"
        if r.location != "device":
            src = f"[{r.location}: outside the device step] {src}"
        out.append(f"{r.node:<24} {r.bucket:<11} {r.kind:<10} {r.layer_count:>5} {r.roofline_us:>11.2f} "
                   f"{_num(r.measured_us):>11} {_num(r.measured_ms, '{:.3f}'):>8} {r.variant or '':<7} {src}")
    return out


def bucket_table(led: Ledger, cal: Optional[Calibration]) -> List[str]:
    cb = calibrated_buckets(led, cal) if cal else {}
    head = f"{'bucket':<12} {'raw':>8} {'roofline':>9} {'unmeasured':>11}"
    if cal:
        head += f" {'k':>6} {'calibrated':>11}"
    out = ["Per bucket (ms per step; raw = benchmark medians x count"
           + (f"; calibrated = k x raw: {PREDICTOR})" if cal else ")"),
           head + "  units without a benchmark at this shape"]
    for b, t in led.buckets().items():
        line = f"{b:<12} {t.measured_ms:>8.3f} {t.roofline_ms:>9.3f} {t.unmeasured_roofline_ms:>11.3f}"
        if cal:
            line += f" {_num(cal.k.get(b), '{:.3f}') if b in cal.k else '1':>6} {cb[b]:>11.3f}"
        out.append(line + f"  {', '.join(t.missing) or '-'}")
    total = f"{'total':<12} {led.measured_ms:>8.3f} {led.roofline_ms:>9.3f} {led.unmeasured_roofline_ms:>11.3f}"
    if cal:
        total += f" {'':>6} {calibrated_ms(led, cal):>11.3f}"
    out.append(total)
    out.append("(unmeasured = roofline of compiler ops and DSA members fused into mla_sparse: no time of their own "
               "in 'raw')")
    if cal:
        src = Path(cal.gate.file).name if cal.gate else "no gate run"
        out.append(f"(k = in-model ms at 5938748 / raw 'before' ms; in-model = the {src} profile buckets, else the "
                   "breakdown reference; k = 1 where no in-model number exists)")
    return out


def _verdict(delta_pct: float) -> str:
    return "PASS" if abs(delta_pct) <= TOLERANCE * 100 else "FAIL"


def reconciliation(led: Ledger, cal: Optional[Calibration]) -> Dict:
    buckets = led.buckets()
    rows = []
    for b in RECONCILED:
        ref = REFERENCES[b]
        got = buckets[b].measured_ms
        delta = _pct(got, ref.reference_ms)
        verdict = _verdict(delta)
        rows.append({
            "bucket": b, "ledger_ms": got, "scope": ref.scope,
            "reference_ms": ref.reference_ms, "reference_kind": ref.reference_kind, "delta_pct": delta,
            "verdict": verdict, "expected": ref.expect, "cause": ref.cause if verdict == "FAIL" else "",
            "note": ref.note, "engine_active_ms": ref.engine_active_ms,
            "k": cal.k.get(b) if cal else None, "k_source": cal.source.get(b) if cal else None,
            "reference_terms": [{"ms": t.ms, "label": t.label, "file": t.file.name} for t in ref.reference],
            "engine_active_terms": [{"ms": t.ms, "label": t.label, "file": t.file.name} for t in ref.engine_active],
            "alternatives": [{"label": a.label, "reference_ms": a.ms, "delta_pct": _pct(got, a.ms),
                              "verdict": _verdict(_pct(got, a.ms)),
                              "terms": [{"ms": t.ms, "label": t.label, "file": t.file.name} for t in a.terms]}
                             for a in ref.alternatives],
            "unmodeled": [{"ms": t.ms, "label": t.label, "file": t.file.name} for t in ref.unmodeled],
            "unmodeled_ms": ref.unmodeled_ms,
        })
    return {"tolerance_pct": TOLERANCE * 100, "rows": rows}


def reconciliation_table(rec: Dict) -> List[str]:
    tol = rec["tolerance_pct"]
    out = [f"Reconciliation (5938748, bs=1, ctx about 1070, rank 0): raw ledger vs the same-scope in-model "
           f"reference, PASS within {tol:.0f}%",
           f"{'bucket':<12} {'ledger':>7} {'reference':>9} {'delta':>8} {'verdict':<7} {'engine-active':>13} {'k':>6}"]
    for r in rec["rows"]:
        flag = "" if r["verdict"] == r["expected"] else f" (expected {r['expected']})"
        out.append(f"{r['bucket']:<12} {r['ledger_ms']:>7.2f} {r['reference_ms']:>9.2f} {r['delta_pct']:>+7.1f}% "
                   f"{r['verdict']:<7} {r['engine_active_ms']:>13.2f} {_num(r['k'], '{:.3f}'):>6}{flag}")
    n = sum(r["verdict"] == "PASS" for r in rec["rows"])
    unexpected = [r["bucket"] for r in rec["rows"] if r["verdict"] != r["expected"]]
    out.append(f"PASS {n} of {len(rec['rows'])}; "
               + (f"UNEXPECTED: {', '.join(unexpected)}" if unexpected else "every verdict as expected"))
    for r in rec["rows"]:
        readings = [(r["reference_ms"], r["verdict"])] + [(a["reference_ms"], a["verdict"]) for a in r["alternatives"]]
        if len({v for _, v in readings}) > 1:
            by = {v: ", ".join(f"{ms:.2f}" for ms, vv in sorted(readings) if vv == v) for v in ("PASS", "FAIL")}
            out.append(f"scope-dependent verdict: {r['bucket']} (PASS at {by['PASS']} ms; FAIL at {by['FAIL']} ms); "
                       "see its alternatives below")
    out.append("reference = same-scope in-model time (wall where the breakdown gives one); engine-active = the "
               "DECODE_BREAKDOWN.md master-table rows; k = in-model / raw, the calibration of the PREDICTOR")
    for r in rec["rows"]:
        out.append(f"  {r['bucket']}: {r['scope']}")
        out.append(f"    reference ({r['reference_kind']}): "
                   + " + ".join(f"{t['ms']:.3f} {t['label']} ({t['file']})" for t in r["reference_terms"]))
        for a in r["alternatives"]:
            out.append(f"    alternative reference: {a['reference_ms']:.2f} ms {a['label']}: {a['delta_pct']:+.1f}% "
                       f"{a['verdict']}")
        for t in r["unmodeled"]:
            out.append(f"    un-modeled (lands in the residual): {t['ms']:.3f} ms {t['label']} ({t['file']})")
        if r["verdict"] == "FAIL":
            out.append(f"    FAIL cause: {r['cause']}")
        if r["note"]:
            out.append(f"    note: {r['note']}")
        if r["k_source"]:
            out.append(f"    k source: {r['k_source']}")
    return out


def profile_table(led: Ledger, cal: Calibration, gate: GateStep) -> List[str]:
    """The gate's measured in-model ms per bucket next to the calibrated PREDICTOR."""
    prof = profile_buckets(gate.buckets_ms)
    cb = calibrated_buckets(led, cal)
    out = [f"Measured in-model per bucket ({Path(gate.file).name} profile, tree {(gate.head or '?')[:7]}) vs "
           f"calibrated ({PREDICTOR})",
           f"{'bucket':<12} {'raw':>8} {'calibrated':>11} {'measured':>9} {'calib-meas':>11}"]
    for b in RECONCILED:
        meas = prof.by_bucket.get(b)
        diff = None if meas is None else cb[b] - meas
        out.append(f"{b:<12} {led.buckets()[b].measured_ms:>8.3f} {cb[b]:>11.3f} {_num(meas, '{:.3f}'):>9} "
                   f"{_num(diff, '{:+.3f}'):>11}")
    out.append(f"profile glue + waits (unnamed compiler ops, waits, DMA issue, idle): {prof.residual_ms:.2f} ms"
               + (f"; UNMAPPED kernel sources (in the residual): {', '.join(prof.unmapped)}" if prof.unmapped else ""))
    return out


def residual_lines(steps: List[tuple], calibrated: Optional[float], raw: float, led: Ledger,
                   rec: Optional[Dict] = None) -> List[str]:
    out = []
    for label, step in steps:
        line = f"  {label}: step {step:.2f} ms -> "
        if calibrated is not None:
            line += f"residual {step - calibrated:.2f} ms vs calibrated ({(step - calibrated) / step * 100:.1f}%); "
        line += f"{step - raw:.2f} ms vs raw"
        out.append(line)
    out.append(f"  residual = compiler glue (roofline of the unmeasured ops alone: {led.unmeasured_roofline_ms:.2f} ms), "
               "waits between kernels, launch skew")
    terms = [(r["bucket"], t) for r in (rec or {}).get("rows", ()) for t in r["unmodeled"]]
    if terms:
        out.append(f"  residual holds the listed un-modeled in-model terms: {sum(t['ms'] for _, t in terms):.3f} ms ("
                   + "; ".join(f"{b}: {t['label']} {t['ms']:.3f}" for b, t in terms) + ")")
    if not led.complete:
        out.append(f"  the sum is partial: the residual also holds {', '.join(led.missing)} (no benchmark at this shape)")
    return out


def run(tip_name: str, point: DecodePoint, reports_dir: Path = REPORTS_DIR,
        json_path: Optional[Path] = None) -> str:
    results = load_micro_results(reports_dir)
    gates = read_gates(reports_dir)
    tip = resolve_tip(tip_name, gates)
    led = build_ledger(point, tip.kernel_set, results)
    base_tip = resolve_tip("5938748", gates)
    cal = calibrate(build_ledger(point, BASELINE, results), base_tip.gate) if point == GATE_POINT else None
    lines = [f"GLM-5.3-Flash decode ledger | tip {tip.name} | {point.label} (max_model_len {point.max_model_len}) "
             f"| TP=64 EP=16, one rank",
             _kernel_set_line(tip),
             f"reports: {reports_dir} ({len(MICRO_FILES) - len(missing_micro_files(reports_dir))} of {len(MICRO_FILES)} "
             f"microbenchmark files; missing: {', '.join(missing_micro_files(reports_dir)) or 'none'})"
             + (f" + mHC batch files: {', '.join(mhc_batch_files(reports_dir))}" if mhc_batch_files(reports_dir) else ""),
             ""]
    lines += node_table(led) + [""] + bucket_table(led, cal) + [""]
    doc = {"tip": tip.name, "sha": tip.sha, "kernel_set": {"name": tip.kernel_set.name,
                                                             "after": sorted(tip.kernel_set.after)},
           "branches": list(tip.branches), "point": vars(point),
           "rows": [dict(vars(r), roofline_ms=r.roofline_ms, measured_ms=r.measured_ms) for r in led.rows],
           "buckets": {b: vars(t) for b, t in led.buckets().items()},
           "kernels_ms": led.kernels_ms, "collectives_ms": led.collectives_ms, "raw_ms": led.measured_ms,
           "host_ms": led.host_ms, "roofline_ms": led.roofline_ms, "unmeasured_roofline_ms": led.unmeasured_roofline_ms,
           "missing": led.missing}

    units = [r for r in led.rows if r.kind in ("measured", "missing")]
    partial = "" if led.complete else (f" PARTIAL: {len(units) - len(led.missing)} of {len(units)} measured units "
                                       "have a benchmark at this shape")
    cal_sum = calibrated_ms(led, cal) if cal else None
    if cal:
        doc["calibration"] = {"gate": cal.gate.file if cal.gate else None, "k": cal.k, "in_model_ms": cal.in_model_ms,
                              "micro_before_ms": cal.micro_before_ms, "source": cal.source}
        doc["calibrated_buckets"] = calibrated_buckets(led, cal)
        doc["calibrated_ms"] = cal_sum
        lines.append(f"sum of measured kernels + collectives, calibrated ({PREDICTOR}): {cal_sum:.2f} ms")
    lines.append(f"sum of measured kernels + collectives, raw benchmark medians: {led.measured_ms:.2f} ms "
                 f"(kernels {led.kernels_ms:.2f} + collectives {led.collectives_ms:.2f}){partial}")
    if cal:
        lines.append(f"scope gap raw - calibrated: {led.measured_ms - cal_sum:+.2f} ms (standalone call wall vs "
                     "in-model time)")
    lines.append(f"roofline of the whole step (every node + collectives): {led.roofline_ms:.2f} ms")
    if led.host_ms:
        lines.append(f"host time outside the device step (measured on the host): {led.host_ms:.2f} ms")
    if not led.complete:
        lines.append(f"no benchmark at this shape for {len(led.missing)} measured unit(s): {', '.join(led.missing)} "
                     "(roofline only; not in the sum)")

    if point != GATE_POINT:
        lines.append(f"no gate run serves {point.label}: no measured step, no residual and no calibration "
                     f"(gates serve {GATE_POINT.label})")
    elif tip.name == "current":
        base = build_ledger(point, BASELINE, results)
        res = {"breakdown_full_host": STEP_5938748.ms - calibrated_ms(base, cal)}
        if base_tip.gate is not None:
            res["gate_baseline_cpu_split"] = base_tip.gate.device_step_ms - calibrated_ms(base, cal)
        if tip.gate is not None and tip.gate_tip is not None:
            gt = tip.gate_tip
            at = build_ledger(point, gt.kernel_set, results)
            verdict = f", gate verdict {tip.gate.verdict}" if tip.gate.verdict else ""
            lines += ["", f"latest gate run: {Path(tip.gate.file).name} (tree {gt.sha[:7]}{verdict}): device step "
                          f"{tip.gate.device_step_ms:.2f} ms; gated branches in that tree: "
                          f"{', '.join(gt.branches) or 'none'}; wave-1 kernels in it: "
                          f"{', '.join(sorted(gt.kernel_set.after)) or 'none'}"]
            if tip.gate.buckets_ms:
                lines += profile_table(at, cal, tip.gate)
            lines.append(f"ledger at tree {gt.sha[:7]}: calibrated {calibrated_ms(at, cal):.2f} ms, "
                         f"raw {at.measured_ms:.2f} ms")
            lines += residual_lines([(Path(tip.gate.file).name, tip.gate.device_step_ms)],
                                    calibrated_ms(at, cal), at.measured_ms, at)
            res["latest_gate"] = tip.gate.device_step_ms - calibrated_ms(at, cal)
            doc["latest_gate"] = {"file": tip.gate.file, "step_ms": tip.gate.device_step_ms, "tree": gt.sha,
                                  "branches": list(gt.branches), "verdict": tip.gate.verdict,
                                  "calibrated_ms": calibrated_ms(at, cal), "raw_ms": at.measured_ms,
                                  "residual_ms": res["latest_gate"]}
        labels = {"latest_gate": "the latest gate run's residual (today's glue, waits and host path)",
                  "breakdown_full_host": f"the 5938748 residual vs {STEP_5938748.ms:.2f} ms ({STEP_5938748.label})",
                  "gate_baseline_cpu_split": (f"the 5938748 residual vs {base_tip.gate.device_step_ms:.2f} ms "
                                              f"({GATE_BASELINE_LABEL})") if base_tip.gate else ""}
        pred = {k: cal_sum + v for k, v in res.items()}
        lines += ["", f"predicted device step for current = {cal_sum:.2f} ms calibrated ({PREDICTOR}) + a measured residual:"]
        for k in ("latest_gate", "breakdown_full_host", "gate_baseline_cpu_split"):
            if k in pred:
                lines.append(f"  {pred[k]:.2f} ms with {labels[k]}: {res[k]:.2f} ms")
        doc["residual_ms"], doc["predicted_step_ms"] = res, pred
    else:
        steps, rec = [], None
        if not tip.kernel_set.after and tip.sha and tip.sha.startswith("5938748"):
            steps.append((f"{STEP_5938748.label} (DECODE_BREAKDOWN.md)", STEP_5938748.ms))
            rec = reconciliation(led, cal)
            doc["reconciliation"] = rec
            lines += [""] + reconciliation_table(rec)
        if tip.gate is not None:
            label = GATE_BASELINE_LABEL if tip.gate is base_tip.gate else "gate run"
            steps.append((f"{label} ({Path(tip.gate.file).name}, tree {(tip.gate.head or '?')[:7]})",
                          tip.gate.device_step_ms))
            if tip.gate.buckets_ms:
                lines += [""] + profile_table(led, cal, tip.gate)
        if steps:
            lines += ["", "measured device step vs sum of measured kernels + collectives:"]
            lines += residual_lines(steps, cal_sum, led.measured_ms, led, rec)
            doc["residual_ms"] = {label: step - cal_sum for label, step in steps}
            doc["residual_raw_ms"] = {label: step - led.measured_ms for label, step in steps}
        else:
            lines.append(f"no gate run measured tip {tip.name}: no residual")

    if json_path is not None:
        Path(json_path).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        lines.append(f"wrote {json_path}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m test.kernel_ledger", description=__doc__.split("\n")[0])
    ap.add_argument("--tip", default="current", help="'current', '5938748', or any commit of this repository")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--emit-shapes", nargs="?", const=REPORTS_DIR / "ledger_shapes.json", type=Path, default=None,
                    metavar="PATH", help="write the per-kernel decode shapes (default reports/ledger_shapes.json)")
    ap.add_argument("--json", type=Path, default=None, metavar="PATH", help="also write the ledger as JSON")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = ap.parse_args(argv)

    point = point_for(args.bs, args.ctx)
    if args.emit_shapes is not None:
        points = [BS1_CTX1K, BS64_CTX8K] + ([point] if point not in (BS1_CTX1K, BS64_CTX8K) else [])
        doc = emit_shapes(points)
        args.emit_shapes.write_text(json.dumps(doc, indent=1) + "\n")
        n = sum(len(p["entries"]) for p in doc["points"])
        print(f"wrote {args.emit_shapes}: {n} entries at {', '.join(p.label for p in points)}")
    print(run(args.tip, point, args.reports_dir, args.json))
    return 0
