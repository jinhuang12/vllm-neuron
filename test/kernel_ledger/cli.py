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

from .models.glm53f.configs import BASELINE, BS1_CTX1K, BS64_CTX8K, NOT_WIRED, DecodePoint, point_for
from .models.glm53f.ledger import Ledger, build_ledger
from .models.glm53f.references import RECONCILED, REFERENCES, STEP_5938748, TOLERANCE
from .models.glm53f.shapes import emit_shapes
from .models.glm53f.tips import Tip, resolve_tip
from .readers.gate import GateStep, read_gates
from .readers.micro import MICRO_FILES, REPORTS_DIR, load_micro_results, missing_micro_files

#: The point every gate run serves: one request, context about 1k, max_model_len 4096.
GATE_POINT = BS1_CTX1K


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
    if tip.merged:
        line += f"; merged gates at this tip: {', '.join(tip.merged)}"
    wired = [k for k in NOT_WIRED if ks.after]
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


def bucket_table(led: Ledger) -> List[str]:
    out = ["Per bucket (ms per step)",
           f"{'bucket':<12} {'measured':>9} {'roofline':>9} {'unmeasured':>11}  units without a benchmark at this shape"]
    for b, t in led.buckets().items():
        out.append(f"{b:<12} {t.measured_ms:>9.3f} {t.roofline_ms:>9.3f} {t.unmeasured_roofline_ms:>11.3f}  "
                   f"{', '.join(t.missing) or '-'}")
    out.append(f"{'total':<12} {led.measured_ms:>9.3f} {led.roofline_ms:>9.3f} {led.unmeasured_roofline_ms:>11.3f}")
    out.append("(unmeasured = roofline of compiler ops and DSA members fused into mla_sparse: no time of their own "
               "in 'measured')")
    return out


def reconciliation(led: Ledger) -> Dict:
    buckets = led.buckets()
    rows = []
    for b in RECONCILED:
        ref = REFERENCES[b]
        got = buckets[b].measured_ms
        row = {"bucket": b, "ledger_ms": got, "scope": ref.scope,
               "as_built_ms": ref.as_built_ms, "as_built_delta_pct": _pct(got, ref.as_built_ms),
               "as_built_terms": [{"ms": t.ms, "label": t.label, "file": t.file.name} for t in ref.as_built]}
        row["as_built_ok"] = abs(row["as_built_delta_pct"]) <= TOLERANCE * 100
        if ref.in_model_wall:
            row.update(wall_ms=ref.in_model_wall_ms, wall_delta_pct=_pct(got, ref.in_model_wall_ms),
                       wall_terms=[{"ms": t.ms, "label": t.label, "file": t.file.name} for t in ref.in_model_wall])
            row["wall_ok"] = abs(row["wall_delta_pct"]) <= TOLERANCE * 100
        rows.append(row)
    return {"tolerance_pct": TOLERANCE * 100, "rows": rows}


def reconciliation_table(rec: Dict) -> List[str]:
    tol = rec["tolerance_pct"]
    out = [f"Reconciliation with DECODE_BREAKDOWN.md (5938748, bs=1, ctx about 1070, rank 0), within {tol:.0f}%?",
           f"{'bucket':<12} {'ledger':>7} {'as-built':>9} {'delta':>8} {'ok':>3}  {'in-model wall':>13} {'delta':>8} "
           f"{'ok':>3}"]
    for r in rec["rows"]:
        wall = (f"{r['wall_ms']:>13.2f} {r['wall_delta_pct']:>+7.1f}% {'yes' if r['wall_ok'] else 'NO':>3}"
                if "wall_ms" in r else f"{'-':>13} {'-':>8} {'-':>3}")
        out.append(f"{r['bucket']:<12} {r['ledger_ms']:>7.2f} {r['as_built_ms']:>9.2f} {r['as_built_delta_pct']:>+7.1f}% "
                   f"{'yes' if r['as_built_ok'] else 'NO':>3}  {wall}")
    n_ab = sum(r["as_built_ok"] for r in rec["rows"])
    n_w = sum(r.get("wall_ok", False) for r in rec["rows"])
    out.append(f"within {tol:.0f}%: {n_ab} of {len(rec['rows'])} against as-built, {n_w} of {len(rec['rows'])} "
               "against in-model wall")
    out.append("as-built = master-table rows, engine-active time (waits are a separate bucket there); in-model wall = "
               "call spans / kernel wall. Benchmark medians are wall times of a kernel in its own graph.")
    for r in rec["rows"]:
        out.append(f"  {r['bucket']}: {r['scope']}")
        out.append("    as-built: " + " + ".join(f"{t['ms']:.3f} {t['label']} ({t['file']})" for t in r["as_built_terms"]))
        if "wall_terms" in r:
            out.append("    wall: " + " + ".join(f"{t['ms']:.3f} {t['label']} ({t['file']})" for t in r["wall_terms"]))
    return out


def _gate_label(g: GateStep) -> str:
    head = (g.head or "?")[:7]
    return f"{Path(g.file).name} (tree {head})"


def residual_lines(led: Ledger, steps: List[tuple]) -> List[str]:
    out = []
    for label, step in steps:
        res = led.residual_ms(step)
        out.append(f"  {label}: step {step:.2f} ms -> residual {res:.2f} ms ({res / step * 100:.1f}% of the step)")
    out.append(f"  residual = compiler glue (roofline of the unmeasured ops alone: {led.unmeasured_roofline_ms:.2f} ms), "
               "waits between kernels, launch skew")
    if not led.complete:
        out.append(f"  the sum is partial: the residual also holds {', '.join(led.missing)} (no benchmark at this shape)")
    return out


def run(tip_name: str, point: DecodePoint, reports_dir: Path = REPORTS_DIR,
        json_path: Optional[Path] = None) -> str:
    results = load_micro_results(reports_dir)
    gates = read_gates(reports_dir)
    tip = resolve_tip(tip_name, gates)
    led = build_ledger(point, tip.kernel_set, results)
    lines = [f"GLM-5.3-Flash decode ledger | tip {tip.name} | {point.label} (max_model_len {point.max_model_len}) "
             f"| TP=64 EP=16, one rank",
             _kernel_set_line(tip),
             f"reports: {reports_dir} ({len(MICRO_FILES) - len(missing_micro_files(reports_dir))} of {len(MICRO_FILES)} "
             f"microbenchmark files; missing: {', '.join(missing_micro_files(reports_dir)) or 'none'})",
             ""]
    lines += node_table(led) + [""] + bucket_table(led) + [""]
    doc = {"tip": tip.name, "sha": tip.sha, "kernel_set": {"name": tip.kernel_set.name,
                                                             "after": sorted(tip.kernel_set.after)},
           "merged": list(tip.merged), "point": vars(point),
           "rows": [dict(vars(r), roofline_ms=r.roofline_ms, measured_ms=r.measured_ms) for r in led.rows],
           "buckets": {b: vars(t) for b, t in led.buckets().items()},
           "kernels_ms": led.kernels_ms, "collectives_ms": led.collectives_ms, "measured_ms": led.measured_ms,
           "host_ms": led.host_ms, "roofline_ms": led.roofline_ms, "unmeasured_roofline_ms": led.unmeasured_roofline_ms,
           "missing": led.missing}

    units = [r for r in led.rows if r.kind in ("measured", "missing")]
    partial = "" if led.complete else (f" PARTIAL: {len(units) - len(led.missing)} of {len(units)} measured units "
                                       "have a benchmark at this shape")
    lines.append(f"sum of measured kernels + collectives: {led.measured_ms:.2f} ms "
                 f"(kernels {led.kernels_ms:.2f} + collectives {led.collectives_ms:.2f}){partial}")
    lines.append(f"roofline of the whole step (every node + collectives): {led.roofline_ms:.2f} ms")
    if led.host_ms:
        lines.append(f"host time outside the device step (measured on the host): {led.host_ms:.2f} ms")
    if not led.complete:
        lines.append(f"no benchmark at this shape for {len(led.missing)} measured unit(s): {', '.join(led.missing)} "
                     "(roofline only; not in the sum)")

    if point != GATE_POINT:
        lines.append(f"no gate run serves {point.label}: no measured step, no residual (gates serve {GATE_POINT.label})")
    elif tip.name == "current":
        base = build_ledger(point, BASELINE, results)
        base_gate = resolve_tip("5938748", gates).gate
        res_bd = base.residual_ms(STEP_5938748.ms)
        pred = {"quiet_host": led.measured_ms + res_bd}
        lines.append("predicted device step = sum above + the 5938748 residual (glue and waits as at 5938748):")
        lines.append(f"  {pred['quiet_host']:.2f} ms with the DECODE_BREAKDOWN.md residual {res_bd:.2f} ms "
                     f"(step {STEP_5938748.ms:.2f}, quiet host)")
        if base_gate is not None:
            res_g = base.residual_ms(base_gate.device_step_ms)
            pred["gate_placement"] = led.measured_ms + res_g
            lines.append(f"  {pred['gate_placement']:.2f} ms with the {Path(base_gate.file).name} residual {res_g:.2f} ms "
                         f"(step {base_gate.device_step_ms:.2f}, the gate's CPU placement)")
        doc["predicted_step_ms"] = pred
        if tip.gate is not None:
            gt = tip.gate_tip
            lines.append(f"latest gate run: {_gate_label(tip.gate)}: device step {tip.gate.device_step_ms:.2f} ms")
            if gt is not None:
                at = build_ledger(point, gt.kernel_set, results)
                lines.append(f"  that tree merged {', '.join(gt.merged) or 'nothing'}; wave-1 kernels in it: "
                             f"{', '.join(sorted(gt.kernel_set.after)) or 'none'}")
                lines += residual_lines(at, [(f"ledger at tree {gt.sha[:7]} ({at.measured_ms:.2f} ms)",
                                              tip.gate.device_step_ms)])
                doc["latest_gate"] = {"file": tip.gate.file, "step_ms": tip.gate.device_step_ms,
                                      "tree": gt.sha, "merged": list(gt.merged),
                                      "ledger_measured_ms": at.measured_ms,
                                      "residual_ms": at.residual_ms(tip.gate.device_step_ms)}
    else:
        steps = []
        if not tip.kernel_set.after and tip.sha and tip.sha.startswith("5938748"):
            steps.append((f"DECODE_BREAKDOWN.md (quiet host)", STEP_5938748.ms))
            rec = reconciliation(led)
            doc["reconciliation"] = rec
            lines += [""] + reconciliation_table(rec) + [""]
        if tip.gate is not None:
            steps.append((_gate_label(tip.gate), tip.gate.device_step_ms))
        if steps:
            lines.append("measured device step vs sum of measured kernels + collectives:")
            lines += residual_lines(led, steps)
            doc["residual_ms"] = {label: led.residual_ms(step) for label, step in steps}
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
