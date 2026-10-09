# SPDX-License-Identifier: Apache-2.0
"""CPU harness: worker GC collections under a decode-like allocation pattern, per policy.

CPU only, no device, no model. Each policy runs in its own child process (fresh
interpreter, so the GC state of one policy never leaks into the next):

1. Import ``vllm_neuron`` (torch, vLLM and the plugin: the worker's own module heap).
2. Build a static heap of ``--heap`` objects (default 5 M, one tracked list each with
   a pointer to an earlier one), then ``gc.collect()``: the end-of-warmup state. The
   TP=64 worker tracked 5.39 M gen-2 objects at its first ``execute_model``.
3. Apply the policy. The worker's policies (``off``, ``freeze``, ``freeze_rare_gen2``)
   run through the worker's own ``gc_policy.apply_post_warmup_gc_policy``; the
   other two need a per-step hook the worker does not have and are emulated here.
4. Run ``--steps`` decode-like steps and record every collection with
   ``gc.callbacks`` (generation, pause, step), and RSS before and after.

One step, calibrated to the bs=64 worker diag (``profile/runs/b64v2/diag``: after the
freeze ~0.07 full passes per step = one per 11 gen-1 triggers, so ~1 gen-1 and ~9-11
gen-0 passes per step; census: gen-0/1 passes free ~600-700 objects per step):

* ``--requests`` x (``--inflight`` / ``--requests``) tracked lists alive for the whole
  step (per-request host metadata); they drive the gen-0 count and are what each
  gen-1 pass promotes to gen-2 (``long_lived_pending``);
* ``--cycles`` two-list reference cycles dropped at once (600 objects only the GC
  frees);
* ``--long-lived`` lists kept for ``--lifetime`` steps (per-request state).

Policies:

* ``off``: CPython default GC, no freeze (0a08ff4).
* ``freeze``: ``freeze_gc_heap()`` alone (e9aa679).
* ``freeze_rare_gen2`` (a): freeze, then ``threshold2`` = ``GEN2_THRESHOLD``
  (the worker default, :data:`CHOSEN_POLICY`).
* ``b_disable_safe_point`` (b): freeze, ``gc.disable()``, then ``gc.collect(1)`` every
  16 steps and ``gc.collect(2)`` every 4096 steps, at the step boundary.
* ``c_periodic_refreeze`` (c): freeze, then ``gc.collect(); gc.freeze()`` every 1000
  steps.

The 64-rank lock-step estimate gives each rank the same per-step GC series at a
random cyclic offset (step counts are shared, so (b)'s offsets are 0) and takes the
slowest rank per step. Real ranks partly share their GC phase, so this spreads the
passes more than the server does; it is an estimate of the barrier tax, not a
measurement.

Usage (from the worktree root; one child at a time, ~1-3 min each):

    PYTHONPATH=$PWD python test/perf/gc_decode_pattern.py \\
        --json <reports>/gcfreeze2_harness.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: The policy this harness recommends. ``test_gc_policy.py`` asserts that it is the
#: worker's default (``gc_policy.DEFAULT_POLICY``), so the two cannot drift apart.
CHOSEN_POLICY = "freeze_rare_gen2"
NO_FREEZE = "off"
FREEZE_ONLY = "freeze"
SAFE_POINT = "b_disable_safe_point"
REFREEZE = "c_periodic_refreeze"
#: Policies the worker itself can apply (``gc_policy.POLICIES``).
WORKER_POLICIES = (CHOSEN_POLICY, FREEZE_ONLY, NO_FREEZE)
ORDER = (NO_FREEZE, FREEZE_ONLY, CHOSEN_POLICY, SAFE_POINT, REFREEZE)
LABELS = {
    NO_FREEZE: "no freeze (CPython default)",
    FREEZE_ONLY: "freeze only (e9aa679)",
    CHOSEN_POLICY: "(a) freeze + gen-2 threshold",
    SAFE_POINT: "(b) freeze + gc.disable + safe-point collect",
    REFREEZE: "(c) freeze + re-freeze every 1000 steps",
}
SAFE_POINT_GEN1_EVERY = 16
SAFE_POINT_GEN2_EVERY = 4096
REFREEZE_EVERY = 1000
RESULT_MARK = "GC_HARNESS_RESULT "

#: Acceptance for the chosen policy (task packet).
MAX_GEN2_PER_1000 = 1.0
MAX_PAUSE_MS = 20.0
MAX_RSS_GROWTH_VS_NO_FREEZE = 2.0


def apply_policy(name: str):
    """Apply one policy to this process; return the per-step hook (or ``None``).

    The worker's policies go through the worker's own code, looked up at call time.
    """
    from vllm_neuron.vllm.worker import gc_policy

    if name in WORKER_POLICIES:
        gc_policy.apply_post_warmup_gc_policy(name)
        return None
    if name == SAFE_POINT:
        gc_policy.apply_post_warmup_gc_policy(FREEZE_ONLY)
        gc.disable()

        def safe_point(step: int) -> None:
            if (step + 1) % SAFE_POINT_GEN2_EVERY == 0:
                gc.collect(2)
            elif (step + 1) % SAFE_POINT_GEN1_EVERY == 0:
                gc.collect(1)

        return safe_point
    if name == REFREEZE:
        gc_policy.apply_post_warmup_gc_policy(FREEZE_ONLY)

        def refreeze(step: int) -> None:
            if (step + 1) % REFREEZE_EVERY == 0:
                gc.collect()
                gc.freeze()

        return refreeze
    raise ValueError(f"unknown policy {name!r}; known: {', '.join(ORDER)}")


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _build_heap(n: int) -> list:
    """``n`` tracked lists, each pointing at an earlier one (pointer chasing)."""
    heap: list = [None] * n
    if n:
        heap[0] = [None]
    for i in range(1, n):
        heap[i] = [heap[(i * 7919) % i]]
    return heap


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[k]


def run_child(policy: str, args) -> dict:
    """One policy, in this process. Returns the measurement as a dict."""
    import vllm_neuron  # noqa: F401  (the worker's module heap: torch, vLLM, plugin)

    rng = random.Random(args.seed)
    t_build = time.perf_counter()
    gc.disable()
    heap = _build_heap(args.heap)
    gc.enable()
    gc.collect()  # end of warmup: the static heap sits in gen 2
    build_s = time.perf_counter() - t_build

    steps = args.steps
    per_step_ms = [[0.0] * steps for _ in range(3)]
    events: list[tuple[int, float, int, int]] = []  # (gen, ms, step, collected)
    state = {"t0": 0.0, "step": -1, "on": False}

    def on_gc(phase: str, info: dict) -> None:
        if not state["on"]:
            return
        if phase == "start":
            state["t0"] = time.perf_counter()
            return
        ms = (time.perf_counter() - state["t0"]) * 1e3
        g = info["generation"]
        s = state["step"]
        events.append((g, ms, s, info.get("collected", 0)))
        if 0 <= s < steps:
            per_step_ms[g][s] += ms

    gc.callbacks.append(on_gc)

    t_policy = time.perf_counter()
    hook = apply_policy(policy)
    policy_ms = (time.perf_counter() - t_policy) * 1e3
    frozen = gc.get_freeze_count()
    threshold_after = gc.get_threshold()

    per_request = max(1, args.inflight // args.requests)
    cycles_per_request = max(0, args.cycles // args.requests)
    ring: list = [None] * args.lifetime
    rss_start = _rss_mb()
    state["on"] = True
    t_steps = time.perf_counter()
    for s in range(steps):
        state["step"] = s
        inflight = []
        for r in range(args.requests):
            inflight.append([s, r])
            for j in range(per_request):
                inflight.append([r, j])
            for _ in range(cycles_per_request):
                a: list = []
                b = [a]
                a.append(b)
        ring[s % args.lifetime] = [[s, i] for i in range(args.long_lived)]
        del inflight
        if hook is not None:
            hook(s)
    steps_s = time.perf_counter() - t_steps
    state["on"] = False
    gc.callbacks.remove(on_gc)
    rss_end = _rss_mb()

    counts = [sum(1 for e in events if e[0] == g) for g in range(3)]
    pauses = [e[1] for e in events]
    gen2 = [e for e in events if e[0] == 2]
    # gen-1 passes between consecutive full passes (CPython: threshold2 when the 25 %
    # rule does not hold them back).
    between: list[int] = []
    n1 = None
    for g, _, _, _ in events:
        if g == 1 and n1 is not None:
            n1 += 1
        elif g == 2:
            if n1 is not None:
                between.append(n1)
            n1 = 0

    # 64-rank lock-step estimate (see module docstring).
    offsets = [0] * args.ranks if policy == SAFE_POINT else [
        rng.randrange(steps) for _ in range(args.ranks)
    ]
    total_ms = [per_step_ms[0][s] + per_step_ms[1][s] + per_step_ms[2][s] for s in range(steps)]
    tax_all, tax_gen2, steps_gen2_any = [], [], 0
    for s in range(steps):
        idx = [(s + o) % steps for o in offsets]
        tax_all.append(max(total_ms[i] for i in idx))
        g2 = max(per_step_ms[2][i] for i in idx)
        tax_gen2.append(g2)
        steps_gen2_any += g2 > 0
    del heap

    per_1000 = {f"gen{g}": counts[g] * 1000.0 / steps for g in range(3)}
    return {
        "policy": policy,
        "label": LABELS.get(policy, policy),
        "python": sys.version.split()[0],
        "steps": steps,
        "heap_objects": args.heap,
        "heap_build_s": round(build_s, 2),
        "policy_apply_ms": round(policy_ms, 1),
        "frozen_objects": frozen,
        "threshold_after": list(threshold_after),
        "gc_enabled_after": gc.isenabled(),
        "counts": {f"gen{g}": counts[g] for g in range(3)},
        "per_1000_steps": {k: round(v, 3) for k, v in per_1000.items()},
        "pause_ms": {
            "max": round(max(pauses, default=0.0), 3),
            "p99": round(_percentile(pauses, 0.99), 3),
            "median": round(_percentile(pauses, 0.5), 3),
        },
        "gen2_pause_ms": {
            "max": round(max((e[1] for e in gen2), default=0.0), 3),
            "median": round(_percentile([e[1] for e in gen2], 0.5), 3),
            "first_step": gen2[0][2] if gen2 else None,
            "collected_total": sum(e[3] for e in gen2),
            "gen1_between_median": statistics.median(between) if between else None,
        },
        "gc_ms_per_step_mean": round(sum(total_ms) / steps, 4),
        "step_ms_mean": round(steps_s * 1e3 / steps, 3),
        "rss_mb": {
            "start": round(rss_start, 1),
            "end": round(rss_end, 1),
            "delta": round(rss_end - rss_start, 2),
        },
        "lockstep": {
            "ranks": args.ranks,
            "aligned": policy == SAFE_POINT,
            "slowest_rank_gc_ms_per_step_mean": round(sum(tax_all) / steps, 4),
            "slowest_rank_gen2_ms_per_step_mean": round(sum(tax_gen2) / steps, 4),
            "steps_with_a_gen2_on_some_rank_frac": round(steps_gen2_any / steps, 4),
        },
        "garbage_uncollectable": len(gc.garbage),
    }


def parse_child_output(text: str) -> dict:
    """The child's result line (the plugin logs to stdout too)."""
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_MARK):
            return json.loads(line[len(RESULT_MARK):])
    raise ValueError("no result line in child output")


def _child_argv(policy: str, args) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child", policy,
        "--heap", str(args.heap), "--steps", str(args.steps), "--ranks", str(args.ranks),
        "--seed", str(args.seed), "--requests", str(args.requests),
        "--inflight", str(args.inflight), "--cycles", str(args.cycles),
        "--long-lived", str(args.long_lived), "--lifetime", str(args.lifetime),
    ]


def _check(chosen: dict, baseline: dict) -> dict:
    base_growth = baseline["rss_mb"]["delta"]
    growth = chosen["rss_mb"]["delta"]
    return {
        "gen2_per_1000_steps": {
            "value": chosen["per_1000_steps"]["gen2"], "limit": MAX_GEN2_PER_1000,
            "pass": chosen["per_1000_steps"]["gen2"] <= MAX_GEN2_PER_1000,
        },
        "max_pause_ms": {
            "value": chosen["pause_ms"]["max"], "limit": MAX_PAUSE_MS,
            "pass": chosen["pause_ms"]["max"] <= MAX_PAUSE_MS,
        },
        "rss_growth_mb": {
            "value": growth, "no_freeze_growth": base_growth,
            "limit": MAX_RSS_GROWTH_VS_NO_FREEZE * max(base_growth, 0.0),
            "pass": growth <= MAX_RSS_GROWTH_VS_NO_FREEZE * max(base_growth, 0.0),
        },
        "gen0_gen1_still_run": {
            "value": [chosen["per_1000_steps"]["gen0"], chosen["per_1000_steps"]["gen1"]],
            "pass": chosen["per_1000_steps"]["gen0"] > 0 and chosen["per_1000_steps"]["gen1"] > 0,
        },
    }


def _table(results: dict) -> str:
    head = (
        f"{'policy':<46} {'gen0/1k':>8} {'gen1/1k':>8} {'gen2/1k':>8} {'max ms':>8} "
        f"{'p99 ms':>7} {'gen2 med':>8} {'RSS dMB':>8} {'gc ms/st':>8} {'64-rank gen2':>12}"
    )
    rows = [head, "-" * len(head)]
    for name in ORDER:
        r = results.get(name)
        if r is None:
            continue
        p = r["per_1000_steps"]
        rows.append(
            f"{r['label']:<46} {p['gen0']:>8.1f} {p['gen1']:>8.1f} {p['gen2']:>8.2f} "
            f"{r['pause_ms']['max']:>8.2f} {r['pause_ms']['p99']:>7.3f} "
            f"{r['gen2_pause_ms']['median']:>8.2f} {r['rss_mb']['delta']:>8.2f} "
            f"{r['gc_ms_per_step_mean']:>8.3f} "
            f"{r['lockstep']['slowest_rank_gen2_ms_per_step_mean']:>12.3f}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, help="write all results here")
    ap.add_argument("--policies", default=",".join(ORDER))
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--heap", type=int, default=5_000_000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--ranks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--inflight", type=int, default=6336, help="in-flight objects per step")
    ap.add_argument("--cycles", type=int, default=320, help="2-object cycles per step")
    ap.add_argument("--long-lived", type=int, default=64)
    ap.add_argument("--lifetime", type=int, default=128)
    args = ap.parse_args(argv)

    if args.child:
        result = run_child(args.child, args)
        print(RESULT_MARK + json.dumps(result), flush=True)
        return 0

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    )
    env.pop("VLLM_NEURON_GC_POLICY", None)
    env.pop("VLLM_GC_DEBUG", None)
    results: dict[str, dict] = {}
    for name in [p for p in args.policies.split(",") if p]:
        t = time.perf_counter()
        out = subprocess.run(
            _child_argv(name, args), env=env, cwd=root, capture_output=True, text=True,
            check=False,
        )
        if out.returncode != 0:
            sys.stderr.write(out.stdout[-4000:] + out.stderr[-4000:])
            raise SystemExit(f"child {name} failed with rc={out.returncode}")
        results[name] = parse_child_output(out.stdout)
        print(f"[{name}] done in {time.perf_counter() - t:.0f} s", flush=True)

    print(_table(results))
    verdict = None
    if CHOSEN_POLICY in results and NO_FREEZE in results:
        verdict = _check(results[CHOSEN_POLICY], results[NO_FREEZE])
        for key, v in verdict.items():
            print(f"chosen {CHOSEN_POLICY}: {key} = {v['value']} -> {'PASS' if v['pass'] else 'FAIL'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "harness": "test/perf/gc_decode_pattern.py",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "cpython_default_threshold": list(gc.get_threshold()),
            "loadavg": os.getloadavg(),
            "args": {k: v for k, v in vars(args).items() if k not in ("json", "child")},
            "chosen_policy": CHOSEN_POLICY,
            "criteria": verdict,
            "policies": results,
        }, indent=2) + "\n")
        print(f"wrote {args.json}")
    if verdict is not None and not all(v["pass"] for v in verdict.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
