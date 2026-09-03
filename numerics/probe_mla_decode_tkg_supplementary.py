# SPDX-License-Identifier: Apache-2.0
"""Supplementary numerics probe for ``mla_decode_tkg`` (triad LD-75).

Covers what scripts/triad_numerics.py's input vocabulary cannot express
(tensor keyword args): 4-D paged fp8 caches with block tables, ue8m0 scale
caches, current-row provenance merge, the in-kernel must-alias cache write
(bitwise byte compare), sink, ratio>0 causal caps including the ratio-128
group-closing decode step (C128 F-240 coverage at the kernel rung), and the
[B,1]-PAD-topk SWA-only decode form. Declared in
numerics/mla_decode_tkg.declaration.json ``_uncovered_by_harness``; tolerances
are READ FROM that declaration and never restated here (prereg
TRIADS-Z0-PREREGISTRATION.md §D7, sealed before any edit).

Legs (subprocess-per-leg, mirroring the harness's env-leg design):
  --leg fallback   plain CPU eager; ``can_run_kernel`` FORCE-FALSE wrapped and
                   counted in every vllm_neuron module (the family-19
                   instrument form — the forbidden env knob is never exported
                   by this script).
  --leg kernel     requires VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1 in the
                   environment (set by the caller); the real gate must accept
                   every case (a fallback-vs-itself comparison is an
                   instrument failure, not a pass).
  --compare        loads both legs' outputs; attention outputs graded with the
                   declaration's rtol/atol in fp32; post-call cache bytes
                   graded BITWISE.

Geometry notes: every sequence owns DISTINCT physical blocks (block table =
arange, no sharing) and each row's three write slots live in that row's own
current block at distinct in-block offsets, so no two writes in a case ever
target the same slot of the same cache — write order can never differ between
the legs.

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

NOPE, ROPE, LAT = 448, 64, 512
NSG = 7  # nope scale groups (448 / 64)
BS = 32  # block size


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
    """Deterministic family-geometry cases. Same tensors in both legs."""

    def g(s):
        return torch.Generator().manual_seed(s)

    def rnd(s, *sh):
        return torch.randn(*sh, generator=g(s), dtype=torch.float32)

    def fp8(t):
        return t.clamp(-240.0, 240.0).to(torch.float8_e4m3fn)

    def pow2(s, shape):
        k = torch.randint(-3, 4, shape, generator=g(s), dtype=torch.int64)
        return torch.pow(2.0, k.to(torch.float32))

    def paged_caches(seed, nb):
        return {
            "latent_k": fp8(rnd(seed + 0, nb, 1, BS, 224)),
            "latent_v": fp8(rnd(seed + 1, nb, 1, BS, 224)),
            "rope": fp8(rnd(seed + 2, nb, 1, BS, 64)),
            "scale": fp8(torch.cat(
                (pow2(seed + 3, (nb, 1, BS, NSG)),
                 torch.zeros(nb, 1, BS, 64 - NSG)), dim=-1)),
            "swa_k": fp8(rnd(seed + 4, nb, 1, BS, 512)),
            "swa_v": fp8(torch.cat(
                (pow2(seed + 5, (nb, 1, BS, NSG)),
                 torch.zeros(nb, 1, BS, 512 - NSG)), dim=-1)),
        }

    def bundle(seed, b):
        return fp8(torch.cat(
            (fp8(rnd(seed + 0, b, NOPE)).to(torch.float32),
             fp8(rnd(seed + 1, b, ROPE)).to(torch.float32),
             fp8(pow2(seed + 2, (b, NSG))).to(torch.float32)), dim=-1))

    def slots_for(bt, pos, deltas=(0, 5, 9)):
        """Three per-row slots in the row's own current block, distinct
        in-block offsets (the three columns write three DIFFERENT cache
        pairs, so only within-column distinctness matters — given by the
        per-sequence-unique block tables)."""
        col = (pos // BS).clamp(0, bt.shape[1] - 1)
        blk = bt.to(torch.int64).gather(1, col.reshape(-1, 1)).reshape(-1)
        base = pos % BS
        return torch.stack(
            [blk * BS + (base + d) % BS for d in deltas], dim=1)

    H = 2
    scale = LAT ** -0.5
    cases = {}

    # ---- p1: paged-fp8 full — ratio 4, window 8, current rows + write, sink.
    B, MB = 8, 4
    nb = B * MB
    seed = 300
    caches = paged_caches(seed, nb)
    bt = torch.arange(B * MB, dtype=torch.int32).reshape(B, MB)
    pos = torch.tensor([3, 7, 11, 15, 19, 23, 27, 30], dtype=torch.int64)
    slots = slots_for(bt, pos)
    slots[1] = -1  # one padded sequence: no write, no merge (F-13)
    topk = torch.full((B, 8), -1, dtype=torch.int32)
    for i in range(B):
        n = min(int(pos[i] + 1) // 4, 8)
        topk[i, :n] = torch.arange(n, dtype=torch.int32)
    cases["p1-paged-full"] = dict(
        args=[rnd(seed + 20, B, H, LAT).to(torch.bfloat16), None, None,
              scale, rnd(seed + 21, H)],
        arg_cache={1: "latent_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, window=8, compress_ratio=4, topk_indices=topk,
            max_compressed_slots=nb * BS,
            latent_v_cache="latent_v", latent_rope_cache="rope",
            latent_scale_cache="scale", latent_widths=(224, 224, 64),
            latent_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            current_latent_rows=fp8(rnd(seed + 22, B, 512)),
            current_scale_rows=fp8(pow2(seed + 23, (B, NSG))),
            current_compressed_rows=bundle(seed + 24, B),
            current_slot_ids=slots, update_cache=True,
        ),
        caches=caches,
    )

    # ---- p2: SWA-only class — swa cache in both slots, [B,1] PAD topk.
    B, MB = 8, 4
    nb = B * MB
    seed = 330
    caches = paged_caches(seed, nb)
    bt = torch.arange(B * MB, dtype=torch.int32).reshape(B, MB)
    pos = torch.tensor([0, 1, 5, 9, 13, 21, 29, 31], dtype=torch.int64)
    slots = slots_for(bt, pos)
    slots[:, 1] = -1
    slots[:, 2] = -1
    cases["p2-swa-only"] = dict(
        args=[rnd(seed + 20, B, H, LAT).to(torch.bfloat16), None, None,
              scale, rnd(seed + 21, H)],
        arg_cache={1: "swa_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, window=8, compress_ratio=0,
            topk_indices=torch.full((B, 1), -1, dtype=torch.int32),
            max_compressed_slots=1, latent_block_table=bt,
            latent_scale_cache="swa_v", latent_widths=(512,),
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            current_latent_rows=fp8(rnd(seed + 22, B, 512)),
            current_scale_rows=fp8(pow2(seed + 23, (B, NSG))),
            current_slot_ids=slots, update_cache=True,
        ),
        caches={"swa_k": caches["swa_k"], "swa_v": caches["swa_v"]},
    )

    # ---- p3: ratio-128 — the C128 F-240 class at the kernel rung. Rows 0/1
    # close a 128-group at THIS decode step (pos 127/255: use_cur_c fires),
    # row 2 is mid-group (pos 20: compressed leg empty), row 3 has one prior
    # closed group AND closes another (pos 383).
    B, MB = 4, 16
    nb = B * MB
    seed = 360
    caches = paged_caches(seed, nb)
    bt = torch.arange(B * MB, dtype=torch.int32).reshape(B, MB)
    pos = torch.tensor([127, 255, 20, 383], dtype=torch.int64)
    slots = slots_for(bt, pos)
    closes = (pos + 1) % 128 == 0
    slots[:, 1] = torch.where(closes, slots[:, 1], torch.full_like(slots[:, 1], -1))
    slots[:, 2] = torch.where(closes, slots[:, 2], torch.full_like(slots[:, 2], -1))
    topk = torch.full((B, 4), -1, dtype=torch.int32)
    topk[0, 0] = 0
    topk[1, :2] = torch.tensor([0, 1], dtype=torch.int32)
    topk[3, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    cases["p3-ratio128-close"] = dict(
        args=[rnd(seed + 20, B, H, LAT).to(torch.bfloat16), None, None,
              scale, rnd(seed + 21, H)],
        arg_cache={1: "latent_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, window=8, compress_ratio=128, topk_indices=topk,
            max_compressed_slots=nb * BS,
            latent_v_cache="latent_v", latent_rope_cache="rope",
            latent_scale_cache="scale", latent_widths=(224, 224, 64),
            latent_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            current_latent_rows=fp8(rnd(seed + 22, B, 512)),
            current_scale_rows=fp8(pow2(seed + 23, (B, NSG))),
            current_compressed_rows=bundle(seed + 24, B),
            current_slot_ids=slots, update_cache=True,
        ),
        caches=caches,
    )

    # ---- p4: no current rows — write skipped, pure prior-cache attention.
    B, MB = 8, 4
    nb = B * MB
    seed = 390
    caches = paged_caches(seed, nb)
    bt = torch.arange(B * MB, dtype=torch.int32).reshape(B, MB)
    pos = torch.tensor([3, 7, 11, 15, 19, 23, 27, 30], dtype=torch.int64)
    topk = torch.full((B, 8), -1, dtype=torch.int32)
    for i in range(B):
        n = min(int(pos[i] + 1) // 4, 8)
        topk[i, :n] = torch.arange(n, dtype=torch.int32)
    cases["p4-no-current"] = dict(
        args=[rnd(seed + 20, B, H, LAT).to(torch.bfloat16), None, None,
              scale, rnd(seed + 21, H)],
        arg_cache={1: "latent_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, window=8, compress_ratio=4, topk_indices=topk,
            max_compressed_slots=nb * BS,
            latent_v_cache="latent_v", latent_rope_cache="rope",
            latent_scale_cache="scale", latent_widths=(224, 224, 64),
            latent_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            update_cache=True,
        ),
        caches=caches,
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

    import vllm_neuron.functional as NF
    from vllm_neuron.functional.attention import mla_decode_tkg as tkg_mod

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
    fn = getattr(NF, "mla_decode_tkg")
    gate = getattr(tkg_mod, "_can_use_mla_decode_tkg")
    record = {"leg": leg, "stamp_commit": _stamp(),
              "clock": datetime.datetime.utcnow().isoformat() + "Z",
              "cases": {}}
    rc = 0
    for name, case in build_cases(torch).items():
        cache_clones = {k: v.clone() for k, v in case["caches"].items()}
        args = [a.clone() if isinstance(a, torch.Tensor) else a
                for a in case["args"]]
        for idx, cname in case["arg_cache"].items():
            args[idx] = cache_clones[cname]
        kwargs = {}
        for k, v in case["kwargs"].items():
            if isinstance(v, str) and v in cache_clones:
                kwargs[k] = cache_clones[v]
            elif isinstance(v, torch.Tensor):
                kwargs[k] = v.clone()
            else:
                kwargs[k] = v

        verdict = bool(gate(*args, **kwargs))
        if leg == "kernel" and not verdict:
            log(f"{name} GATE_DECLINED on kernel leg — instrument failure "
                "(fallback-vs-itself is not a comparison)")
            rc = 2
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            log(f"{name} RAISED {type(e).__name__}: {e!r}")
            record["cases"][name] = {"raised": f"{type(e).__name__}: {e}"}
            rc = max(rc, 1)
            continue
        np.save(os.path.join(outdir, f"{name}.out.npy"),
                out.to(torch.float32).numpy())
        for cname, cval in cache_clones.items():
            np.save(os.path.join(outdir, f"{name}.{cname}.bytes.npy"),
                    cval.view(torch.uint8).numpy())
        record["cases"][name] = {"gate": verdict, "out_shape": list(out.shape)}
        log(f"{name} ok gate={verdict} out={tuple(out.shape)}")
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
    rc = 0
    for name in sorted(set(fleg["cases"]) | set(kleg["cases"])):
        fc, kc = fleg["cases"].get(name), kleg["cases"].get(name)
        if not fc or not kc or "raised" in fc or "raised" in kc:
            log(f"{name} FAIL missing/raised leg: fallback={fc} kernel={kc}")
            rc = 1
            continue
        a = np.load(os.path.join(fdir, f"{name}.out.npy"))
        b = np.load(os.path.join(kdir, f"{name}.out.npy"))
        err = np.abs(a - b)
        bound = atol + rtol * np.abs(a)
        bad = int((err > bound).sum())
        log(f"{name} OUT max_abs={err.max():.6g} viol={bad}/{err.size} "
            f"gateK={kc['gate']}")
        if bad:
            rc = 1
        for f in sorted(os.listdir(fdir)):
            if f.startswith(name + ".") and f.endswith(".bytes.npy"):
                ca = np.load(os.path.join(fdir, f))
                kb = os.path.join(kdir, f)
                if not os.path.exists(kb):
                    log(f"{name} CACHE {f} MISSING on kernel leg")
                    rc = 1
                    continue
                neq = int((ca != np.load(kb)).sum())
                log(f"{name} CACHE {f} bitwise_neq={neq}")
                if neq:
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
