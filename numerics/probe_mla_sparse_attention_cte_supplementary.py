# SPDX-License-Identifier: Apache-2.0
"""Supplementary numerics probe for ``mla_sparse_attention_cte`` (triad LD-76).

Covers what scripts/triad_numerics.py's input vocabulary cannot express
(tensor keyword args): 4-D paged fp8 caches with block tables, ue8m0 scale
caches, the F-240 current-chunk provenance split (current_kv_rows /
current_compressed_rows + writer-frame slot ids), ratio>0 pooling including
ratio-128 groups closing in-chunk (prefill-side C128 coverage at the kernel
rung), swa width splits, and chunked query processing. Declared in
numerics/mla_sparse_attention_cte.declaration.json ``_uncovered_by_harness``;
tolerances are READ FROM that declaration (prereg TRIADS-Z0-PREREGISTRATION.md
§D7, sealed before any edit).

Legs identical in design to probe_mla_decode_tkg_supplementary.py:
  --leg fallback   can_run_kernel FORCE-FALSE wrapped+counted (family-19
                   instrument form; the forbidden env knob never exported).
  --leg kernel     requires VLLM_NEURON_CPU_MODE=1 + NKI_SIMULATOR=1 in the
                   caller's environment; the real gate must accept every case.
  --compare        outputs graded with the declaration's rtol/atol in fp32;
                   post-call cache bytes graded BITWISE (this op writes
                   nothing — byte inequality means the kernel corrupted a
                   cache).

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
NSG = 7
BS = 32


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
            "comp_k": fp8(rnd(seed + 0, nb, 1, BS, 224)),
            "comp_v": fp8(rnd(seed + 1, nb, 1, BS, 224)),
            "rope": fp8(rnd(seed + 2, nb, 1, BS, 64)),
            "scale": fp8(torch.cat(
                (pow2(seed + 3, (nb, 1, BS, NSG)),
                 torch.zeros(nb, 1, BS, 64 - NSG)), dim=-1)),
            "swa_k": fp8(rnd(seed + 4, nb, 1, BS, 512)),
            "swa_v": fp8(torch.cat(
                (pow2(seed + 5, (nb, 1, BS, NSG)),
                 torch.zeros(nb, 1, BS, 512 - NSG)), dim=-1)),
        }

    def kv_rows(seed, t):
        """current_kv_rows: [T, 519] fp8 = 512 cache-form row + 7 scale cols."""
        return fp8(torch.cat(
            (fp8(rnd(seed + 0, t, LAT)).to(torch.float32),
             fp8(pow2(seed + 1, (t, NSG))).to(torch.float32)), dim=-1))

    def comp_bundle(seed, t):
        """current_compressed_rows: [T, 519] fp8 codes+rope+scales."""
        return fp8(torch.cat(
            (fp8(rnd(seed + 0, t, NOPE)).to(torch.float32),
             fp8(rnd(seed + 1, t, ROPE)).to(torch.float32),
             fp8(pow2(seed + 2, (t, NSG))).to(torch.float32)), dim=-1))

    def close_ids(pos, ratio):
        """Writer frame: group id closed at each token, -1 when none closes."""
        closes = (pos + 1) % ratio == 0
        gid = (pos + 1) // ratio - 1
        return torch.where(closes, gid, torch.full_like(gid, -1))

    def topk_causal(pos, ratio, k, extra_seed):
        """Group ids < (pos+1)//ratio, padded -1; deterministic prefix."""
        t = pos.shape[0]
        topk = torch.full((t, k), -1, dtype=torch.int32)
        for i in range(t):
            n = min(int(pos[i] + 1) // ratio, k)
            topk[i, :n] = torch.arange(n, dtype=torch.int32)
        return topk

    H = 2
    scale = LAT ** -0.5
    cases = {}

    # ---- s1: paged-fp8 chunk at first_pos=64, ratio 4, full provenance split.
    T = 64
    nb = 8
    seed = 400
    caches = paged_caches(seed, nb)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pos = torch.arange(64, 64 + T, dtype=torch.int64)
    cases["s1-paged-split"] = dict(
        args=[rnd(seed + 20, T, H, LAT).to(torch.bfloat16), None, None,
              topk_causal(pos, 4, 8, seed), rnd(seed + 21, H), scale, 16],
        arg_cache={1: "comp_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, compress_ratio=4,
            compressed_v_cache="comp_v", compressed_rope_cache="rope",
            compressed_scale_cache="scale", compressed_widths=(224, 224, 64),
            compressed_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            current_kv_rows=kv_rows(seed + 22, T),
            current_kv_slot_ids=pos.clone(),
            current_compressed_rows=comp_bundle(seed + 24, T),
            current_compressed_slot_ids=close_ids(pos, 4),
            chunk_size=64,
        ),
        caches=caches,
    )

    # ---- s2: ratio-128 groups closing IN-CHUNK (C128 F-240, prefill side):
    # fresh sequence, T=256, groups 0 and 1 close at pos 127 / 255 —
    # frontier = 0, every compressed read is operand-sourced.
    T = 256
    nb = 8
    seed = 430
    caches = paged_caches(seed, nb)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pos = torch.arange(T, dtype=torch.int64)
    cases["s2-ratio128-inchunk"] = dict(
        args=[rnd(seed + 20, T, H, LAT).to(torch.bfloat16), None, None,
              topk_causal(pos, 128, 2, seed), rnd(seed + 21, H), scale, 64],
        arg_cache={1: "comp_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, compress_ratio=128,
            compressed_v_cache="comp_v", compressed_rope_cache="rope",
            compressed_scale_cache="scale", compressed_widths=(224, 224, 64),
            compressed_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            current_kv_rows=kv_rows(seed + 22, T),
            current_kv_slot_ids=pos.clone(),
            current_compressed_rows=comp_bundle(seed + 24, T),
            current_compressed_slot_ids=close_ids(pos, 128),
            chunk_size=64,
        ),
        caches=caches,
    )

    # ---- s3: degenerate topk (all -1), no current operands — pure
    # strictly-prior cache window attention mid-sequence.
    T = 64
    nb = 8
    seed = 460
    caches = paged_caches(seed, nb)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pos = torch.arange(160, 160 + T, dtype=torch.int64)
    cases["s3-window-only"] = dict(
        args=[rnd(seed + 20, T, H, LAT).to(torch.bfloat16), None, None,
              torch.full((T, 4), -1, dtype=torch.int32),
              rnd(seed + 21, H), scale, 16],
        arg_cache={1: "comp_k", 2: "swa_k"},
        kwargs=dict(
            positions=pos, compress_ratio=4,
            compressed_v_cache="comp_v", compressed_rope_cache="rope",
            compressed_scale_cache="scale", compressed_widths=(224, 224, 64),
            compressed_block_table=bt,
            swa_scale_cache="swa_v",
            swa_widths=(512,), swa_block_table=bt,
            chunk_size=64,
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
    from vllm_neuron.functional.attention import (
        mla_sparse_attention_cte as cte_mod,
    )

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
    fn = getattr(NF, "mla_sparse_attention_cte")
    gate = getattr(cte_mod, "_can_use_mla_sparse_attention_cte")
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
                log(f"{name} CACHE {f} bitwise_neq={neq} (op writes nothing; "
                    "any inequality is corruption)")
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
