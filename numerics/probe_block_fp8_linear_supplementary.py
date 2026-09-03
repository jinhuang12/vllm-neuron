# SPDX-License-Identifier: Apache-2.0
"""Graded probe for the three LD-11 encoding cases the harness cannot express.

``scripts/triad_numerics.py`` generates inputs from a fixed vocabulary
(``randn``/``rand``/``ones``/``zeros``/``arange``/``randint``), which cannot
build a weight block whose bytes carried exponent field 15 in the checkpoint's
OCP grid, nor its paired doubled block scale. Those inputs must be the ACTUAL
output of the LD-24 loader transforms. So the declaration
(``numerics/block_fp8_linear.declaration.json``) splits the eight declared
cases across two instruments and names this file as the second one. It runs the
SAME two legs against the SAME plain-torch reference and READS ITS THRESHOLDS
FROM THE DECLARATION -- it invents none.

This file is COMMITTED, with the declaration, before any graded number exists,
so the probe itself is pre-registered: it cannot be tuned after seeing a
result. Every emitted record carries ``git rev-parse HEAD`` of the tree it ran
in.

Usage (two processes, because both knobs are read from the environment at
import time; the runner script sequences them):

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        VLLM_NEURON_DISABLE_NKI_KERNELS=1 python -m \
        numerics.probe_block_fp8_linear_supplementary --leg fallback --out DIR

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        NKI_SIMULATOR=1 python -m \
        numerics.probe_block_fp8_linear_supplementary --leg simulator --out DIR

``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` must be exported BEFORE the first
import (R-24): the fork freezes ``FP8_CLAMP_MAX`` at import time and falls back
to 448.0 in bare CPU mode, so a leg that imported first would grade against the
wrong fp8 ceiling and could pass for the wrong reason. The probe asserts the
resolved value rather than trusting the export.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TREE = _HERE.parent
DECLARATION = _HERE / "block_fp8_linear.declaration.json"


def _assert_env() -> str:
    """Refuse to produce a number under the wrong environment."""
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") != "trn2":
        raise SystemExit(
            "NEURON_PLATFORM_TARGET_OVERRIDE must be 'trn2' and must be "
            "exported before the first import (R-24)."
        )
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise SystemExit("VLLM_NEURON_CPU_MODE=1 is required: this probe is device-free.")
    sim = os.environ.get("NKI_SIMULATOR") == "1"
    dis = os.environ.get("VLLM_NEURON_DISABLE_NKI_KERNELS") == "1"
    if sim == dis:
        raise SystemExit(
            "exactly one of NKI_SIMULATOR=1 (simulator leg) or "
            "VLLM_NEURON_DISABLE_NKI_KERNELS=1 (fallback leg) must be set; "
            f"got NKI_SIMULATOR={sim}, VLLM_NEURON_DISABLE_NKI_KERNELS={dis}."
        )
    return "simulator" if sim else "fallback"


def _stamp() -> str:
    """``git rev-parse HEAD`` of the tree this file lives in."""
    return subprocess.run(
        ["git", "-C", str(_TREE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Measurement, with the definitions the declared thresholds were derived under
# ---------------------------------------------------------------------------


def _max_rel(got, ref) -> float:
    """``max|got-ref| / max|ref|`` -- iteration 1's definition, unchanged.

    A NORMALIZED max error, not a per-element relative error. This is the
    definition the measured 1.32114e-06 was computed under
    (``artifacts/repairs/author_kernel_triads-iter1/probe_ld11_remedy.py:120``),
    and therefore the definition the declared 2e-05 threshold is calibrated
    against. It also avoids the per-element form's pathology: a GEMM output
    position where accumulation cancels to near zero has an enormous
    per-element relative error while being numerically fine.
    """
    import torch

    g = got.to(torch.float32)
    r = ref.to(torch.float32)
    return float((g - r).abs().max() / max(float(r.abs().max()), 1e-12))


def _bf16_ulp_stats(got, ref) -> dict:
    """Exact bf16 ULP distance per position, by total-ordered bit pattern."""
    import torch

    assert got.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16

    def key(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, torch.full_like(i, -32768) - i, i)

    d = (key(got) - key(ref)).abs()
    n = int(d.numel())
    differing = int((d > 0).sum())
    return {
        "positions": n,
        "differing_positions": differing,
        "differing_fraction": (differing / n) if n else 0.0,
        "max_ulp_distance": int(d.max()) if n else 0,
        "positions_beyond_one_ulp": int((d > 1).sum()),
        "exact": differing == 0,
    }


def _non_finite(t) -> int:
    import torch

    return int((~torch.isfinite(t.to(torch.float32))).sum())


# ---------------------------------------------------------------------------
# The three declared encoding cases, built through the LD-24 loader's own code
# ---------------------------------------------------------------------------

_OCP_FP8_MAX = 448.0  # the CHECKPOINT's grid, before LD-24. Never this op's.


def _ocp_block_quantize(block, fp8_max=_OCP_FP8_MAX):
    """Quantize one 128x128 block under the checkpoint's own recipe.

    ue8m0 block scale against the OCP ceiling, code clamped to +-fp8_max. This
    is what produced the field-15 bytes that made the unremedied op emit
    non-finite outputs on this venue (iteration-1 leg E1).

    Returns ``(ocp_bytes_uint8, e8m0_code_uint8)``.
    """
    import torch

    amax = block.abs().amax().clamp(min=1e-12)
    ceil_log2 = int(torch.ceil(torch.log2(amax / fp8_max)).item())
    s = 2.0**ceil_log2
    codes = (block / s).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    e8m0 = torch.tensor([[ceil_log2 + 127]], dtype=torch.uint8)
    return codes.view(torch.uint8), e8m0


def _field15_count(bytes_u8) -> int:
    """Number of bytes whose e4m3 exponent field is 15 (inf/NaN on this venue)."""
    return int(((bytes_u8 & 0x7F) >= 0x78).sum())


def _through_ld24(ocp_bytes, e8m0_codes, name):
    """Run the LD-24 loader transforms -- the loader's OWN functions."""
    wl = importlib.import_module("vllm_neuron.model.deepseek_v4.weight_loaders")
    w = wl._reencode_ocp_to_legacy_halved(ocp_bytes, name, f"{name}.weight")
    ws = wl._e8m0_to_fp32_doubled(e8m0_codes, name, f"{name}.weight_scale_inv")
    return w, ws


def _case_field15_preimage(seed: int):
    """Declared case 1/3: a weight block with a field-15 OCP preimage byte.

    Drawn under the checkpoint's own recipe and REDRAWN until a field-15 byte
    appears, exactly as the declaration states. The draw counter is reported so
    the search is visible rather than implied.
    """
    import torch

    draws = 0
    while True:
        draws += 1
        g = torch.Generator().manual_seed(seed + draws)
        block = torch.randn(128, 128, generator=g, dtype=torch.float32) * 100.0
        ocp, e8m0 = _ocp_block_quantize(block)
        n15 = _field15_count(ocp)
        if n15 > 0:
            break
        if draws > 64:
            raise SystemExit(
                "no field-15 OCP byte appeared in 64 draws; the case cannot be "
                "built as declared and this is a probe defect, not a pass."
            )
    gx = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 128, generator=gx, dtype=torch.float32).to(torch.bfloat16)
    w, ws = _through_ld24(ocp, e8m0, "case_field15_preimage")
    meta = {
        "draws_until_field15": draws,
        "ocp_field15_bytes": n15,
        "legacy_field15_bytes_after_ld24": _field15_count(w.view(torch.uint8)),
        "e8m0_code": int(e8m0[0, 0]),
        "doubled_scale": float(ws[0, 0]),
    }
    return x, w, ws, meta


def _case_field01_odd_mantissa(seed: int):
    """Declared case 2/3: field 0 and field 1 bytes with ODD mantissas.

    The halving's entire inexact zone: field 1 halves into the subnormals and
    field 0 halves within them, both round-to-nearest-even, so this is where
    LD-24 is NOT exact and the 2e-05 fp32 clause is actually exercised rather
    than trivially passed. Byte patterns are PLACED, not drawn -- an odd
    mantissa at field <= 1 is far too rare in a random draw to rely on.
    """
    import torch

    odd = [0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F]
    patterns = odd + [0x80 | b for b in odd]  # both signs
    flat = torch.tensor(
        [patterns[i % len(patterns)] for i in range(128 * 128)], dtype=torch.uint8
    )
    ocp = flat.reshape(128, 128)
    # Scale code 127 -> 2**0 = 1.0, doubling to 2.0 exactly.
    e8m0 = torch.tensor([[127]], dtype=torch.uint8)
    gx = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 128, generator=gx, dtype=torch.float32).to(torch.bfloat16)
    w, ws = _through_ld24(ocp, e8m0, "case_field01_odd_mantissa")
    halved = w.view(torch.uint8)
    exact_halves = int(
        (
            ocp.to(torch.float8_e4m3fn).to(torch.float32) / 2.0
            == w.to(torch.float32)
        ).sum()
    )
    meta = {
        "distinct_ocp_bytes": int(torch.unique(ocp).numel()),
        "all_inputs_field_le_1": bool(int(((ocp & 0x7F) >= 0x10).sum()) == 0),
        "legacy_field15_bytes_after_ld24": _field15_count(halved),
        "positions_halved_exactly": exact_halves,
        "positions_total": int(ocp.numel()),
        "doubled_scale": float(ws[0, 0]),
    }
    return x, w, ws, meta


def _case_activation_240_448(seed: int):
    """Declared case 3/3: the activation groups the 240 ceiling exists for.

    K = 256, i.e. two activation groups per row:

    * group 0 carries a maximum placed so that ``amax / s`` lands in
      (240, 448] under the OLD 448 ceiling -- the COMMON case, ~90% of ue8m0
      groups (assessment section 2.7), where a 448-divisor scale emits a
      field-15 byte that IS inf/NaN on this venue;
    * group 1 spans more than 13 binades of internal dynamic range, so the
      small end of the group quantizes to subnormal codes or to zero.

    The op's own ceiling is ``FP8_CLAMP_MAX`` = 240 on trn2, so neither group
    may emit a field-15 byte. That count is measured, not assumed.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 256, generator=g, dtype=torch.float32)
    # group 0: place the per-row maximum at 300.0 -> s = 2**ceil(log2(300/448))
    # = 2**0 = 1, so amax/s = 300, inside (240, 448].
    x[:, 0:128] = x[:, 0:128] / x[:, 0:128].abs().amax(dim=1, keepdim=True) * 300.0
    # group 1: 14 binades, 2**-6 .. 2**8
    ramp = torch.logspace(-6, 8, 128, base=2.0, dtype=torch.float32)
    x[:, 128:256] = torch.sign(x[:, 128:256]) * ramp.reshape(1, 128)
    x = x.to(torch.bfloat16)

    gw = torch.Generator().manual_seed(seed + 1)
    ocp_blocks = []
    codes = []
    for kb in range(2):
        block = torch.randn(128, 128, generator=gw, dtype=torch.float32) * 10.0
        b, c = _ocp_block_quantize(block)
        ocp_blocks.append(b)
        codes.append(int(c[0, 0]))
    ocp = torch.cat(ocp_blocks, dim=1)
    e8m0 = torch.tensor([codes], dtype=torch.uint8)
    w, ws = _through_ld24(ocp, e8m0, "case_activation_240_448")

    # What the op emits for these activations, under BOTH ceilings, computed
    # with the reference's own recipe. The 448 column is the contrast that
    # shows the ceiling choice is load-bearing, not a style preference.
    ref_mod = importlib.import_module("numerics.block_fp8_linear_reference")
    emitted = {}
    for label, fp8_max in (("240", 240.0), ("448", 448.0)):
        grouped = x.to(torch.float32).reshape(128, 2, 128)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=ref_mod.AMAX_FLOOR)
        s = ref_mod._pow2_ceil_scale(amax, fp8_max)
        q = (grouped / s).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        emitted[label] = _field15_count(q.view(torch.uint8))
        if label == "240":
            emitted["group0_amax_over_s"] = float((amax / s).reshape(128, 2)[0, 0])
        else:
            emitted["group0_amax_over_s_448"] = float((amax / s).reshape(128, 2)[0, 0])
    meta = {
        "e8m0_codes": codes,
        "doubled_scales": [float(v) for v in ws.reshape(-1)],
        "emitted_activation_field15_count_at_240": emitted["240"],
        "emitted_activation_field15_count_at_448": emitted["448"],
        "group0_amax_over_scale_at_240": emitted["group0_amax_over_s"],
        "group0_amax_over_scale_at_448": emitted["group0_amax_over_s_448"],
        "binades_group1": 14,
    }
    return x, w, ws, meta


_SUPPLEMENTARY = {
    "encoding_field15_preimage_block": (_case_field15_preimage, 21001),
    "encoding_field01_odd_mantissa_block": (_case_field01_odd_mantissa, 22001),
    "activation_group_240_448_and_wide_dynamic_range": (
        _case_activation_240_448,
        23001,
    ),
}


# ---------------------------------------------------------------------------
# Gate verdicts for every declared case, harness cases included
# ---------------------------------------------------------------------------


def _harness_case_gate_verdicts(decl, gate) -> list:
    """Evaluate the gate on the five harness cases' declared input signatures.

    Only shapes, dtypes and ranks reach the gate, so the input VALUES are
    irrelevant here and zeros are used; the declared shape/dtype triple is what
    is being graded. This makes "the out-of-envelope case took the fallback" a
    measurement rather than an inference.
    """
    import torch

    name_to_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    out = []
    for case in decl["cases"]:
        ins = case["inputs"]
        t = [
            torch.zeros(*spec["shape"], dtype=name_to_dtype[spec["dtype"]])
            for spec in ins
        ]
        verdict = gate(
            t[0], t[1], t[2], (128, 128), 128, torch.float32, torch.bfloat16, None
        )
        out.append({"case": case["name"], "kernel_eligible": bool(verdict)})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=("fallback", "simulator"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    leg = _assert_env()
    if leg != args.leg:
        raise SystemExit(f"--leg {args.leg} contradicts the environment ({leg}).")
    stamp = _stamp()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    decl = json.loads(DECLARATION.read_text())
    clauses = decl["clauses"]

    import torch

    from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

    if FP8_CLAMP_MAX != 240.0:
        raise SystemExit(
            f"FP8_CLAMP_MAX resolved to {FP8_CLAMP_MAX}, not 240.0. Plan section 2 "
            "step 0: on trn2 the legacy nl.float8_e4m3 amax is 240 and a leg "
            "graded against 448 would pass for the wrong reason."
        )

    import vllm_neuron.functional as NF

    bfl = importlib.import_module("vllm_neuron.functional.block_fp8_linear")
    ref_mod = importlib.import_module("numerics.block_fp8_linear_reference")
    reference = ref_mod.block_fp8_linear_reference

    record = {
        "probe": "numerics/probe_block_fp8_linear_supplementary.py",
        "op": decl["op"],
        "leg": leg,
        "stamp_git_rev_parse_HEAD": stamp,
        "declaration_clauses": clauses,
        "declaration_path": "numerics/block_fp8_linear.declaration.json",
        "fp8_clamp_max_resolved": float(FP8_CLAMP_MAX),
        "env": {
            k: os.environ.get(k)
            for k in (
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "VLLM_NEURON_CPU_MODE",
                "NKI_SIMULATOR",
                "VLLM_NEURON_DISABLE_NKI_KERNELS",
            )
        },
        "nf_block_fp8_linear_resolves": hasattr(NF, "block_fp8_linear"),
        "gate_verdicts_harness_cases": _harness_case_gate_verdicts(
            decl, bfl._can_use_block_fp8_linear
        ),
        "cases": [],
    }

    failures = []
    for name, (builder, seed) in _SUPPLEMENTARY.items():
        x, w, ws, meta = builder(seed)
        gate_verdict = bool(
            bfl._can_use_block_fp8_linear(
                x, w, ws, (128, 128), 128, torch.float32, torch.bfloat16, None
            )
        )
        expected_gate = leg == "simulator"
        got_bf16 = NF.block_fp8_linear(x, w, ws, out_dtype=torch.bfloat16)
        got_fp32 = NF.block_fp8_linear(x, w, ws, out_dtype=torch.float32)
        ref_bf16 = reference(x, w, ws, out_dtype=torch.bfloat16)
        ref_fp32 = reference(x, w, ws, out_dtype=torch.float32)

        entry = {
            "case": name,
            "shape": {"m": int(x.shape[0]), "k": int(x.shape[1]), "n": int(w.shape[0])},
            "construction": meta,
            "kernel_eligible": gate_verdict,
            "kernel_eligible_expected_for_this_leg": expected_gate,
            "non_finite_outputs_bf16": _non_finite(got_bf16),
            "non_finite_outputs_fp32": _non_finite(got_fp32),
            "fp32_max_rel_vs_reference": _max_rel(got_fp32, ref_fp32),
            "bf16_vs_reference": _bf16_ulp_stats(got_bf16, ref_bf16),
        }

        # Cross-leg comparison: the simulator leg grades the kernel against the
        # fallback bytes the fallback leg actually produced, not against a
        # re-derivation of them.
        npz = outdir / f"fallback-{name}.npz"
        if leg == "fallback":
            import numpy as np

            np.savez(
                npz,
                bf16=got_bf16.to(torch.float32).numpy(),
                fp32=got_fp32.numpy(),
            )
            entry["saved"] = npz.name
        else:
            if npz.exists():
                import numpy as np

                z = np.load(npz)
                fb_bf16 = torch.from_numpy(z["bf16"]).to(torch.bfloat16)
                fb_fp32 = torch.from_numpy(z["fp32"])
                entry["fp32_max_rel_vs_fallback"] = _max_rel(got_fp32, fb_fp32)
                entry["bf16_vs_fallback"] = _bf16_ulp_stats(got_bf16, fb_bf16)
            else:
                entry["fp32_max_rel_vs_fallback"] = None
                entry["bf16_vs_fallback"] = None
                failures.append(f"{name}: fallback-leg output missing ({npz})")

        # ---- clause enforcement, read from the declaration ------------------
        checks = []
        checks.append(
            (
                "gate_verdict",
                gate_verdict == expected_gate,
                f"gate={gate_verdict}, expected {expected_gate} on the {leg} leg",
            )
        )
        checks.append(
            (
                "non_finite_outputs_max",
                entry["non_finite_outputs_bf16"] <= clauses["non_finite_outputs_max"]
                and entry["non_finite_outputs_fp32"]
                <= clauses["non_finite_outputs_max"],
                f"bf16={entry['non_finite_outputs_bf16']}, "
                f"fp32={entry['non_finite_outputs_fp32']}, "
                f"cap={clauses['non_finite_outputs_max']}",
            )
        )
        checks.append(
            (
                "fp32_accumulator_max_rel",
                entry["fp32_max_rel_vs_reference"]
                <= clauses["fp32_accumulator_max_rel"],
                f"{entry['fp32_max_rel_vs_reference']:.6g} vs "
                f"{clauses['fp32_accumulator_max_rel']:.6g}",
            )
        )
        u = entry["bf16_vs_reference"]
        checks.append(
            (
                "bf16_max_ulp_distance",
                u["max_ulp_distance"] <= clauses["bf16_max_ulp_distance"],
                f"{u['max_ulp_distance']} vs {clauses['bf16_max_ulp_distance']}",
            )
        )
        checks.append(
            (
                "bf16_one_ulp_position_fraction_max",
                u["differing_fraction"]
                <= clauses["bf16_one_ulp_position_fraction_max"],
                f"{u['differing_fraction']:.6g} vs "
                f"{clauses['bf16_one_ulp_position_fraction_max']:.6g}",
            )
        )
        if leg == "simulator" and entry.get("bf16_vs_fallback"):
            uf = entry["bf16_vs_fallback"]
            checks.append(
                (
                    "bf16_max_ulp_distance_vs_fallback",
                    uf["max_ulp_distance"] <= clauses["bf16_max_ulp_distance"],
                    f"{uf['max_ulp_distance']} vs {clauses['bf16_max_ulp_distance']}",
                )
            )
            checks.append(
                (
                    "fp32_accumulator_max_rel_vs_fallback",
                    entry["fp32_max_rel_vs_fallback"]
                    <= clauses["fp32_accumulator_max_rel"],
                    f"{entry['fp32_max_rel_vs_fallback']:.6g} vs "
                    f"{clauses['fp32_accumulator_max_rel']:.6g}",
                )
            )
        if name == "activation_group_240_448_and_wide_dynamic_range":
            cap = decl["supplementary_legs"]["emitted_activation_field15_count"]["max"]
            checks.append(
                (
                    "emitted_activation_field15_count",
                    meta["emitted_activation_field15_count_at_240"] <= cap,
                    f"{meta['emitted_activation_field15_count_at_240']} vs cap {cap} "
                    f"(contrast: {meta['emitted_activation_field15_count_at_448']} "
                    f"under the old 448 ceiling)",
                )
            )

        entry["checks"] = [
            {"clause": c, "passed": bool(p), "measured": m} for c, p, m in checks
        ]
        for c, p, m in checks:
            if not p:
                failures.append(f"{name}: {c}: {m}")
        record["cases"].append(entry)

        print(f"--- {name} [{leg}] ---", flush=True)
        for c, p, m in checks:
            print(f"    {'PASS' if p else 'FAIL'}  {c}: {m}", flush=True)

    record["failures"] = failures
    record["result"] = "PASS" if not failures else "FAIL"
    out_json = outdir / f"supplementary-{leg}.json"
    out_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"stamp (git rev-parse HEAD) = {stamp}", flush=True)
    print(f"record = {out_json}", flush=True)
    print(f"RESULT = {record['result']}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
