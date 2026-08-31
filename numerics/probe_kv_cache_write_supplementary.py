# SPDX-License-Identifier: Apache-2.0
"""Supplementary numerics probe for ``kv_cache_write`` (triad LD-79).

Covers what scripts/triad_numerics.py's input vocabulary cannot express
(tensor keyword args) plus the in-place contract: ``valid_mask`` mask-skip
(mixed and all-False), mixed -1/valid slot vectors, duplicate sentinel rows,
the full per-class width sweep {64,128,224,512,672,896,1344,2688} with
BITWISE post-call cache-byte compare, rows dtype sweep {fp8 identity, f32
cast, bf16 cast}, int32 slot_ids, and the eager-simulator in-place contract
(after the PUBLIC op on the kernel leg the CALLER's cache object carries the
written bytes — the Z0 §D-copyback). Declared in
numerics/kv_cache_write.declaration.json ``_uncovered_by_harness``;
tolerances are READ FROM that declaration and never restated here (prereg
TRIADS79-Z0-PREREGISTRATION.md §D7, sealed before any edit).

Legs (subprocess-per-leg, mirroring the harness's env-leg design):
  --leg fallback   plain CPU eager; ``can_run_kernel`` FORCE-FALSE wrapped and
                   counted in every vllm_neuron module (the family-19
                   instrument form — the forbidden env knob is never exported
                   by this script). Gate False + CPU tensor -> the torch
                   fallback (the incumbent composition).
  --leg kernel     requires VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1 in the
                   environment (set by the caller); the REAL gate must accept
                   every case (all probe cases sit inside the D8 envelope; a
                   fallback-vs-itself comparison is an instrument failure,
                   not a pass).
  --compare        loads both legs' outputs; returned tensors graded with the
                   declaration's rtol/atol (0/0 = exact) in fp32; post-call
                   cache bytes graded BITWISE (uint8 view, bitwise_neq=0);
                   kernel-leg gate acceptance, in-place byte carry, and the
                   s02 unchanged-cache ZERO with its s01 firing control all
                   asserted from the leg records.

Geometry notes: every writing case uses UNIQUE valid slot ids (stride-5
residues mod the slot count; gcd(5, 64) = gcd(5, 128) = gcd(5, 256) = 1), so
no two writes in a case ever target the same slot — write order can never
differ between the legs. No case both validly writes slot 0 and carries a
sentinel row (the recorded contract-exclusion zone, declaration
``_domain_notes``); sentinel duplicates are no-ops on every spelling.

Run from a scratch checkout root (PYTHONPATH=tree) in the pinned instance
venv, device-free. Exit codes: 0 PASS, 1 FAIL, 2 usage/instrument error.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

BS = 32  # block size (every production cache class)


def log(*a):
    print(*a, flush=True)


def _stamp():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "<no-git>"


def build_cases(torch):
    """Deterministic cases. Same tensors in both legs (rebuilt from seeds)."""

    def g(s):
        return torch.Generator().manual_seed(s)

    def rnd(s, *sh):
        return torch.randn(*sh, generator=g(s), dtype=torch.float32)

    def fp8(t):
        return t.clamp(-240.0, 240.0).to(torch.float8_e4m3fn)

    def cache_of(seed, nb, w):
        return fp8(rnd(seed, nb, 1, BS, w))

    def ids_unique(t, slots, offset=3):
        # stride-5 residues: unique for t < slots when gcd(5, slots) == 1
        return ((torch.arange(t, dtype=torch.int64) * 5 + offset) % slots)

    cases = {}

    # ---- (a) valid_mask: mixed True/False ------------------------------
    cases["s01-mask-mixed"] = dict(
        cache=cache_of(401, 4, 224),
        slot_ids=ids_unique(32, 128),
        rows=rnd(402, 32, 224),
        valid_mask=(torch.arange(32) % 3 != 0),
    )
    # ---- (a) valid_mask: all-False -> cache bytes UNCHANGED -------------
    cases["s02-mask-allfalse"] = dict(
        cache=cache_of(403, 4, 224),
        slot_ids=ids_unique(32, 128),
        rows=rnd(404, 32, 224),
        valid_mask=torch.zeros(32, dtype=torch.bool),
    )
    # ---- (b) mixed -1/valid ids (valid unique, none 0) ------------------
    ids = torch.arange(48, dtype=torch.int64) * 5 + 1  # 1..236, unique, no 0
    ids[::2] = -1
    cases["s03-mixed-sentinel"] = dict(
        cache=cache_of(405, 8, 512),
        slot_ids=ids,
        rows=fp8(rnd(406, 48, 512)),
        valid_mask=None,
    )
    # ---- (c) duplicate sentinel rows ------------------------------------
    ids = torch.full((40,), -1, dtype=torch.int64)
    ids[8], ids[16], ids[24], ids[32] = 5, 9, 17, 33
    cases["s04-dup-sentinel"] = dict(
        cache=cache_of(407, 4, 128),
        slot_ids=ids,
        rows=fp8(rnd(408, 40, 128)),
        valid_mask=None,
    )
    # ---- (d) per-class width sweep --------------------------------------
    sweep = [
        ("s05-w64-pad7", 4, 64, 7, "f32", 410),
        ("s06-w128-full", 4, 128, 128, "fp8", 412),
        ("s07-w224-full", 4, 224, 224, "f32", 414),
        ("s08-w512-full", 4, 512, 512, "fp8", 416),
        ("s09-w672-pad512", 4, 672, 512, "f32", 418),
        ("s10-w896-pad512", 4, 896, 512, "fp8", 420),
        ("s11-w1344-full", 2, 1344, 1344, "f32", 422),
        ("s12-w2688-pad2080", 2, 2688, 2080, "f32", 424),
    ]
    for name, nb, w, n, rk, seed in sweep:
        rows = rnd(seed + 1, 24, n)
        cases[name] = dict(
            cache=cache_of(seed, nb, w),
            slot_ids=ids_unique(24, nb * BS),
            rows=fp8(rows) if rk == "fp8" else rows,
            valid_mask=None,
        )
    # ---- (e) rows dtype: bf16 (fp8 identity + f32 covered above) --------
    cases["s13-dtype-bf16"] = dict(
        cache=cache_of(426, 4, 224),
        slot_ids=ids_unique(24, 128),
        rows=rnd(427, 24, 224).to(torch.bfloat16),
        valid_mask=None,
    )
    # ---- (g) int32 slot ids ---------------------------------------------
    cases["s14-int32-ids"] = dict(
        cache=cache_of(428, 4, 64),
        slot_ids=ids_unique(24, 128).to(torch.int32),
        rows=rnd(429, 24, 7),
        valid_mask=None,
    )
    return cases


def run_leg(leg, outdir):
    import torch

    if leg == "kernel":
        if not (os.environ.get("VLLM_NEURON_CPU_MODE") == "1"
                and os.environ.get("NKI_SIMULATOR") == "1"):
            log("INSTRUMENT_ERROR kernel leg needs VLLM_NEURON_CPU_MODE=1 "
                "and NKI_SIMULATOR=1 in the environment")
            return 2
    else:
        for v in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
            if os.environ.get(v):
                log(f"INSTRUMENT_ERROR fallback leg must not run under {v}")
                return 2

    import importlib

    import vllm_neuron.functional as NF

    # `from vllm_neuron.functional import kv_cache_write` binds the re-exported
    # FUNCTION (functional/__init__.py), not the module — S2-r1 refusing line:
    # "AttributeError: 'function' object has no attribute
    # '_can_use_kv_cache_write'" (probe :198). Import the module explicitly.
    kvw_mod = importlib.import_module("vllm_neuron.functional.kv_cache_write")

    counter = {"calls": 0}
    if leg == "fallback":
        patched = 0
        for mn, m in list(sys.modules.items()):
            if mn.startswith("vllm_neuron") and m is not None \
                    and callable(getattr(m, "can_run_kernel", None)):
                def _mk():
                    def w(*a, **k):
                        counter["calls"] += 1
                        return False
                    return w
                setattr(m, "can_run_kernel", _mk())
                patched += 1
        log(f"leg=fallback can_run_kernel FORCE-FALSE wrapped in {patched} modules")
        if patched == 0:
            log("INSTRUMENT_ERROR gate wrap matched 0 modules")
            return 2

    torch.set_grad_enabled(False)
    os.makedirs(outdir, exist_ok=True)
    fn = getattr(NF, "kv_cache_write")
    gate = getattr(kvw_mod, "_can_use_kv_cache_write")
    record = {"leg": leg, "stamp_commit": _stamp(),
              "clock": datetime.datetime.utcnow().isoformat() + "Z",
              "cases": {}}
    rc = 0
    for name, case in build_cases(torch).items():
        cache = case["cache"].clone()
        pre = cache.clone()
        ids = case["slot_ids"].clone()
        rows = case["rows"].clone()
        mask = case["valid_mask"]
        mask = mask.clone() if isinstance(mask, torch.Tensor) else None

        verdict = bool(gate(cache, ids, rows, mask))
        if leg == "kernel" and not verdict:
            log(f"{name} GATE_DECLINED on kernel leg — instrument failure "
                "(every probe case sits inside the D8 envelope; "
                "fallback-vs-itself is not a comparison)")
            rc = 2
        try:
            out = fn(cache, ids, rows, valid_mask=mask)
        except Exception as e:
            log(f"{name} RAISED {type(e).__name__}: {e!r}")
            record["cases"][name] = {"raised": f"{type(e).__name__}: {e}"}
            rc = max(rc, 1)
            continue
        np.save(os.path.join(outdir, f"{name}.out.npy"),
                out.to(torch.float32).numpy())
        np.save(os.path.join(outdir, f"{name}.cache.bytes.npy"),
                cache.view(torch.uint8).numpy())
        in_place_neq = int(
            (out.view(torch.uint8) != cache.view(torch.uint8)).sum())
        pre_post_neq = int(
            (pre.view(torch.uint8) != cache.view(torch.uint8)).sum())
        record["cases"][name] = {
            "gate": verdict,
            "out_shape": list(out.shape),
            "ret_is_cache": bool(out is cache),
            "in_place_bytes_neq": in_place_neq,
            "pre_post_neq": pre_post_neq,
        }
        log(f"{name} ok gate={verdict} ret_is_cache={out is cache} "
            f"in_place_neq={in_place_neq} pre_post_neq={pre_post_neq}")
    record["force_false_calls"] = counter["calls"]
    if leg == "fallback" and counter["calls"] == 0:
        log("INSTRUMENT_ERROR FORCE-FALSE wrap never consulted")
        rc = 2
    with open(os.path.join(outdir, "leg.json"), "w") as f:
        json.dump(record, f, indent=1)
    log(f"leg={leg} record saved rc={rc}")
    return rc


def compare(decl_path, fdir, kdir):
    decl = json.load(open(decl_path))
    tol = decl["cases"][0]["tolerances"]
    rtol, atol = float(tol["rtol"]), float(tol["atol"])
    log(f"tolerances from declaration: rtol={rtol} atol={atol}")
    fleg = json.load(open(os.path.join(fdir, "leg.json")))
    kleg = json.load(open(os.path.join(kdir, "leg.json")))
    log(f"stamps fallback={fleg['stamp_commit']} kernel={kleg['stamp_commit']}")
    if fleg["stamp_commit"] != kleg["stamp_commit"]:
        log("FAIL stamp mismatch between legs")
        return 1
    if fleg.get("force_false_calls", 0) <= 0:
        log("FAIL fallback leg FORCE-FALSE wrap consulted 0 times")
        return 1
    rc = 0
    for name in sorted(set(fleg["cases"]) | set(kleg["cases"])):
        fc, kc = fleg["cases"].get(name), kleg["cases"].get(name)
        if not fc or not kc or "raised" in fc or "raised" in kc:
            log(f"{name} FAIL missing/raised leg: fallback={fc} kernel={kc}")
            rc = 1
            continue
        if not kc["gate"]:
            log(f"{name} FAIL kernel-leg gate declined (in-envelope case)")
            rc = 1
        # (f) the in-place carry: returned bytes == caller-cache bytes, BOTH
        # legs (on the kernel leg this IS the D-copyback grading).
        for legname, c in (("fallback", fc), ("kernel", kc)):
            if c.get("in_place_bytes_neq", -1) != 0:
                log(f"{name} FAIL {legname} leg: returned tensor and caller "
                    f"cache bytes differ ({c.get('in_place_bytes_neq')})")
                rc = 1
        a = np.load(os.path.join(fdir, f"{name}.out.npy"))
        b = np.load(os.path.join(kdir, f"{name}.out.npy"))
        err = np.abs(a - b)
        bound = atol + rtol * np.abs(a)
        bad = int((err > bound).sum())
        ca = np.load(os.path.join(fdir, f"{name}.cache.bytes.npy"))
        cb = np.load(os.path.join(kdir, f"{name}.cache.bytes.npy"))
        neq = int((ca != cb).sum())
        log(f"{name} OUT max_abs={err.max():.6g} viol={bad}/{err.size} "
            f"CACHE bitwise_neq={neq} gateK={kc['gate']}")
        if bad or neq:
            rc = 1
    # s02 all-False mask: cache UNCHANGED (zero) with s01 as its firing
    # control (same geometry, mixed mask, MUST have changed bytes).
    for legname, legrec in (("fallback", fleg), ("kernel", kleg)):
        s01 = legrec["cases"].get("s01-mask-mixed", {})
        s02 = legrec["cases"].get("s02-mask-allfalse", {})
        if s02.get("pre_post_neq", -1) != 0:
            log(f"s02 FAIL {legname} leg: all-False mask changed cache bytes "
                f"({s02.get('pre_post_neq')})")
            rc = 1
        if s01.get("pre_post_neq", 0) <= 0:
            log(f"s01 FAIL {legname} leg: firing control did not fire "
                f"(pre_post_neq={s01.get('pre_post_neq')}) — the s02 zero "
                "is unwitnessed")
            rc = 1
    log("COMPARE " + ("PASS" if rc == 0 else "FAIL"))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=("fallback", "kernel"))
    ap.add_argument("--out")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--declaration")
    ap.add_argument("--fallback-dir")
    ap.add_argument("--kernel-dir")
    args = ap.parse_args()
    if args.compare:
        if not (args.declaration and args.fallback_dir and args.kernel_dir):
            log("usage: --compare --declaration D --fallback-dir A --kernel-dir B")
            return 2
        return compare(args.declaration, args.fallback_dir, args.kernel_dir)
    if not (args.leg and args.out):
        log("usage: --leg {fallback,kernel} --out DIR")
        return 2
    return run_leg(args.leg, args.out)


if __name__ == "__main__":
    sys.exit(main())
