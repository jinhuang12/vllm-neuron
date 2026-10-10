# SPDX-License-Identifier: Apache-2.0
"""Device time and numerics of the sparse MLA call, gathered and masked, at served contexts.

Run it ONLY through the device lease, which pins the cores and sets LNC2; the script refuses
to run without them and selects no cores itself::

    export NEURON_LIBTORCH_CACHE_ROOT=<an empty directory>
    python3 /home/ubuntu/glm53f-wt/devlease.py slice <slice> -- python3 \\
        test/hardware/benchmark_mla_gatherfree.py --case ctx8192 --out bench.json \\
        --save-dir <dir>

A case is one call of :data:`CHUNK` queries at rows ``ctx - CHUNK .. ctx - 1`` over a window of
``ctx`` rows (the last chunk at that context), one head (TP=64), latent 512, the indexer's
selection of ``SELECT_K`` pools of ``KPOOL`` rows. ``mla_sparse.mla_sparse_attention`` is
called twice:

* ``gather``: without a pool size, the per-query gathers of the selection;
* ``masked``: with ``pool_size=KPOOL``, as ``model_fp8.py`` calls it, which takes
  ``mla_dense_window.mla_masked_window_attention`` (the selection as a bias over the resident
  window) where ``masked_window_serves`` says so; a case where it does not is a failure.

Numerics, per seed: both outputs against a float64 oracle within the low-precision body's
derived budget (``benchmark_sparse_mla_prefill.error_budget``), and masked against gathered
within :func:`order_budget`, the derived bound of the summation and rounding order that
changes between them; every output is kept as ``.pt`` with ``--save-dir``. ``--capture``
replaces the random operands by a recorded model call's (:func:`captured_operands`), one
query head per seed.

Timing: the two graphs and a one-element floor, interleaved, ``--reps`` repeats of
``--iterations`` calls each, every emission compared bit for bit with the graph's first
output; a sample is the runtime system trace's ``nc_exec_running`` interval with the two
physical cores merged (:mod:`benchmark_mla_sparse_8k`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import benchmark_sparse_mla_prefill as sparse_bench
from benchmark_mla_sparse_8k import (
    BANK_PAGES_B64,
    CHUNK,
    build_floor,
    compiled,
    repeat_groups,
    to_device,
)
from benchmark_sparse_mla_prefill import (
    BF16_UNIT,
    FP32_UNIT,
    HBM_BYTES_PER_S,
    HEADS,
    KPOOL,
    LATENT,
    LEASE_MARKER,
    LNC2_MEASURED_PEAK_FLOPS,
    PAGE,
    SCALE,
    SELECT_K,
    TE_PEAK_FLOPS,
    WIDTH,
    Case,
)

#: The contexts measured: the std line's sparse graphs (3072 = KV 2048 + one chunk, 4096),
#: one between, and the 8k line's graph.
CONTEXTS = (3072, 4096, 6144, 8192)


def context_case(ctx: int) -> Case:
    """The last chunk at context ``ctx``: rows ``ctx - CHUNK .. ctx - 1`` over ``ctx`` rows."""
    return Case(f"ctx{ctx}", CHUNK, ctx // PAGE, BANK_PAGES_B64, ctx - CHUNK, ctx,
                f"{CHUNK} queries at rows {ctx - CHUNK}..{ctx - 1}, window {ctx} rows")


CASES = tuple(context_case(ctx) for ctx in CONTEXTS)


def captured_operands(case: Case, capture: Path, head: int):
    """``(q, bank, indices, table, written, offset, window)`` from a recorded model call.

    ``capture`` holds, for context ``C``, ``q_ctx{C}.pt`` ``[CHUNK, H, L]`` (the absorbed
    query), ``window_ctx{C}.pt`` ``[C, L]`` (the window's latent rows) and ``sel_ctx{C}.pt``
    ``[CHUNK, cols]`` (the indexer's expanded selection), as the model passed them. One head
    of the query is taken, as one rank holds at TP=64. The window's pages go into a zeroed
    bank of ``case.bank_pages`` pages in a seeded random order, and this chunk's rows are
    also passed as ``written`` at ``case.start``, as the model passes them.
    """
    import torch

    ctx = case.window_pages * PAGE
    q_all = torch.load(capture / f"q_ctx{ctx}.pt")
    window = torch.load(capture / f"window_ctx{ctx}.pt").to(torch.bfloat16)
    indices = torch.load(capture / f"sel_ctx{ctx}.pt").to(torch.int32)
    want = {"q": (case.tokens, LATENT), "window": (ctx, LATENT), "selection": (case.tokens, WIDTH)}
    got = {"q": (q_all.shape[0], q_all.shape[2]), "window": tuple(window.shape),
           "selection": tuple(indices.shape)}
    if q_all.ndim != 3 or got != want or not 0 <= head < q_all.shape[1]:
        raise SystemExit(f"{capture}: want {want} and a head below {q_all.shape[1]}; got {got}")
    q = q_all[:, head:head + 1, :].to(torch.bfloat16).contiguous()
    gen = torch.Generator().manual_seed(case.start + head)
    pages = torch.randperm(case.bank_pages, generator=gen)[:case.window_pages]
    bank = torch.zeros(case.bank_pages * PAGE, LATENT, dtype=torch.bfloat16)
    bank.view(-1, PAGE, LATENT)[pages] = window.view(-1, PAGE, LATENT)
    table = pages.reshape(case.window_pages, 1).to(torch.int32)
    written = window[case.start:case.start + case.tokens].clone()
    offset = torch.tensor([[case.start]], dtype=torch.int32)
    return q, bank, indices, table, written, offset, window


def gamma(n: int, unit: float = FP32_UNIT) -> float:
    """The worst-case relative error of an ``n``-term fp32 sum of non-negative terms."""
    return n * unit / (1 - n * unit)


def order_budget(q, window, indices) -> dict:
    """The derived bound of ``|masked - gather|`` per output element, from the arithmetic.

    Both paths multiply the same exact products and take the same exp of the same scaled
    score against the same global maximum, normalise ``p`` before splitting it into bf16
    hi/lo halves and contract the halves with the stored cache in fp32. What changes is the
    order of three sums and the values the roundings after them act on. With ``x_j`` the
    normalised probabilities, ``n`` the most rows a query selects, ``u`` the fp32 unit,
    ``ub`` the bf16 unit and ``V_e = sum_j x_j |v_j[e]|``, each path's deviation from exact
    arithmetic is bounded below, and the two bounds add:

    * the row sum over ``n`` non-negative exps: ``gamma(n - 1)``, relative, on every x;
    * its reciprocal and the normalising multiply: ``2 u``;
    * the hi/lo split, ``hi + lo = x (1 + d)``, ``|d| <= ub ** 2``;
    * MM2 over ``2 n`` products and the add of the halves: ``gamma(2 n)`` of ``V_e``;
    * MM1 over the latent: the same products in the same order of latent tiles on both
      paths, but the order inside a tile is the PE's; an absolute ``gamma(L) * A`` per score
      (``A`` the largest ``|q| . |k|``), on a score and on the maximum, which the exp turns
      into ``2 * scale * gamma(L) * A`` relative on ``p`` and twice that on ``x``;
    * the exp's argument, three roundings of values at most ``scale * A``: ``6 u scale A``
      relative on ``x``.

    So ``|masked - gather|[e] <= 2 * F * V_e`` with ``F = gamma(n - 1) + 2 u + ub**2 +
    gamma(2 n) + 4 scale gamma(L) A + 6 u scale A``. ``F_order`` is ``F`` without the two
    MM1 terms: the bound when both paths' scores are bit-equal.
    """
    import torch

    qd, cd = q.double(), window.double()
    rows = indices.to(torch.int64)
    keep = rows >= 0
    n = int(keep.sum(dim=1).max())
    weighted = torch.zeros(q.shape, dtype=torch.float64)
    largest_abs_dot = 0.0
    for s in range(q.shape[0]):
        picked = cd[rows[s][keep[s]]]
        weights = torch.softmax(torch.einsum("hl,kl->hk", qd[s], picked) * SCALE, dim=-1)
        weighted[s] = weights @ picked.abs()
        largest_abs_dot = max(largest_abs_dot,
                              float(torch.einsum("hl,kl->hk", qd[s].abs(), picked.abs()).max()))
    latent = int(q.shape[2])
    f_order = gamma(n - 1) + 2 * FP32_UNIT + BF16_UNIT ** 2 + gamma(2 * n)
    mm1 = (4 * SCALE * gamma(latent) * largest_abs_dot
           + 6 * FP32_UNIT * SCALE * largest_abs_dot)
    return {"selected": n, "largest_abs_dot": largest_abs_dot, "F_order": f_order,
            "F_mm1": mm1, "F": f_order + mm1, "weighted": weighted}


def compare(masked, gather, budget: dict) -> dict:
    """``max |masked - gather|`` against :func:`order_budget`, per element and per tensor."""
    delta = (masked.double() - gather.double()).abs()
    weighted = budget["weighted"]
    floor = weighted.max() * 1e-30
    out = {"max_abs_delta": float(delta.max()), "bit_equal": bool((delta == 0).all())}
    for key in ("F", "F_order"):
        bound = 2 * budget[key] * weighted
        out[f"ratio_elementwise_{key}"] = float((delta / bound.clamp_min(floor)).max())
        out[f"ratio_tensor_{key}"] = float(delta.max() / bound.max())
    return out


def entitlement(case: Case) -> dict:
    """The masked form's roofline: every window row scored and contracted, in us."""
    t, w, d = case.tokens * HEADS, case.window_pages * PAGE, LATENT
    flops = 2 * 2 * t * w * d
    nbytes = (w * d * 2 + t * d * 2 + case.tokens * WIDTH * 4 + case.window_pages * 4
              + case.tokens * d * 2 + 4 + t * d * 4)
    byte_us = nbytes / HBM_BYTES_PER_S * 1e6
    return {"T": t, "W": w, "D": d, "flops": flops, "bytes": nbytes, "byte_us": byte_us,
            "entitlement_us": max(flops / LNC2_MEASURED_PEAK_FLOPS * 1e6, byte_us),
            "entitlement_us_79e12": max(flops / TE_PEAK_FLOPS * 1e6, byte_us),
            "formula": "max(bytes / 716e9, 2*2*T*W*D / peak), peak 153.0e12 (MEASURED LNC2, "
                       "ship bar) and 79e12 (inventory); bytes = window + q + indices + table "
                       "+ written + offset + fp32 output"}


def graphs():
    """The sparse seam without and with the selection's pool size, compiled."""
    from vllm_neuron.functional.attention.mla_sparse import mla_sparse_attention

    def gather(q, bank, indices, table, written, offset):
        return mla_sparse_attention(q, bank, indices, SCALE, block_table_row=table,
                                    written=written, write_offset=offset, page_size=PAGE)

    def masked(q, bank, indices, table, written, offset):
        return mla_sparse_attention(q, bank, indices, SCALE, block_table_row=table,
                                    written=written, write_offset=offset, page_size=PAGE,
                                    pool_size=KPOOL)

    return {"gather": compiled(gather), "masked": compiled(masked)}


def run_case(case: Case, floor, args, failures: list) -> dict:
    """Numerics over ``--seeds`` operand sets, then interleaved timing of each set with identity.

    Each operand set is timed on its own: the gather's time depends on the selected rows'
    addresses, the masked kernel's does not, so ``set_spread`` (largest minus smallest set
    median, over the smallest) states the spread between index sets.
    """
    import statistics

    import torch

    from vllm_neuron.functional.attention import mla_dense_window as DW

    torch._dynamo.reset()
    DW.reset_mla_dense_window_dispatch_counters()
    paths = graphs()
    numerics, timed, first_call_s = [], [], {}
    for seed_index in range(args.seeds):
        if args.capture is None:
            seed = 1000 * seed_index + case.tokens + case.start
            operands = sparse_bench.make_operands(case, seed)
            selection = "random top-512 (benchmark_sparse_mla_prefill.selection)"
        else:
            seed = seed_index
            operands = captured_operands(case, args.capture, seed_index)
            selection = f"CPU indexer selection, query head {seed_index}: {args.capture}"
        q, bank, indices, table, written, offset, window = operands
        inputs = to_device((q, bank, indices, table, written, offset))
        outputs = {}
        for name, graph in paths.items():
            started = time.perf_counter()
            outputs[name] = graph(*inputs).to("cpu")
            first_call_s.setdefault(name, time.perf_counter() - started)
        want = sparse_bench.oracle(q, window, indices)
        bar = sparse_bench.operand_budget(q, window, indices)
        budget = order_budget(q, window, indices)
        row = {"seed": seed, "selection": selection, "float64_budget": bar,
               "order_budget": {k: v for k, v in budget.items() if k != "weighted"},
               "masked_vs_gather": compare(outputs["masked"], outputs["gather"], budget)}
        for name, got in outputs.items():
            row[f"rel_l2_{name}_vs_float64"] = sparse_bench.rel_l2(got, want)
            if not torch.isfinite(got).all() or row[f"rel_l2_{name}_vs_float64"] > bar["rel_l2"]:
                failures.append(f"{case.name} seed {seed}: {name} outside its float64 budget")
        if row["masked_vs_gather"]["ratio_elementwise_F"] > 1:
            failures.append(f"{case.name} seed {seed}: masked vs gather above the order budget")
        if args.save_dir is not None:
            for name, got in outputs.items():
                torch.save(got, args.save_dir / f"{case.name}_seed{seed}_{name}.pt")
        numerics.append(row)
        print(json.dumps({"case": case.name, "seed": seed,
                          "masked_vs_gather": row["masked_vs_gather"],
                          "rel_l2_vs_float64": {n: row[f"rel_l2_{n}_vs_float64"] for n in outputs},
                          "float64_bar": bar["rel_l2"]}), flush=True)
        timed.append((seed, inputs, {name: [out] for name, out in outputs.items()}))
    masked_calls = DW.mla_masked_window_dispatch_count()
    if masked_calls < 1:
        failures.append(f"{case.name}: the pool-size call never took the masked kernel")
    timing_graphs = dict(paths)
    timing_graphs["floor"] = floor["graph"]
    sets = []
    for seed, inputs, reference in timed:
        feeds = {name: inputs for name in paths}
        feeds["floor"] = floor["inputs"]
        one = repeat_groups(timing_graphs, feeds, args, reference)
        for name in paths:
            if one[name]["identical"] != one[name]["emissions"]:
                failures.append(f"{case.name} seed {seed}: {name} emissions differ from its "
                                f"first output")
        sets.append({"seed": seed, **one})
    timing = {"sets": sets}
    for name in timing_graphs:
        medians = [one[name]["median_us"] for one in sets]
        timing[name] = {"set_medians_us": medians, "median_us": statistics.median(medians),
                        "set_spread": (max(medians) - min(medians)) / min(medians),
                        "rep_spread": max(one[name]["rep_spread"] for one in sets)}
        if name in paths:
            timing[name]["emissions"] = sum(one[name]["emissions"] for one in sets)
            timing[name]["identical"] = sum(one[name]["identical"] for one in sets)
    timing["masked_over_gather"] = timing["masked"]["median_us"] / timing["gather"]["median_us"]
    timing["masked_over_gather_per_set"] = [one["masked"]["median_us"] / one["gather"]["median_us"]
                                            for one in sets]
    timing["noise_floor"] = max(timing[name]["rep_spread"] for name in paths)
    ent = entitlement(case)
    timing["masked_over_entitlement"] = timing["masked"]["median_us"] / ent["entitlement_us"]
    return {"case": asdict(case), "entitlement": ent, "gather_entitlement":
            sparse_bench.entitlement(case), "timing": timing, "numerics": numerics,
            "first_call_s": first_call_s, "masked_dispatches": masked_calls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES],
                        help="case (repeatable; default: every case)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=4, help="calls per graph per repeat")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3, help="operand sets checked per case")
    parser.add_argument("--capture", type=Path,
                        help="a recorded model call per context (q_ctx{C}.pt, window_ctx{C}.pt, "
                             "sel_ctx{C}.pt); seed i takes query head i")
    parser.add_argument("--save-dir", type=Path, help="keep every checked output as .pt here")
    args = parser.parse_args()
    if min(args.reps, args.iterations, args.seeds) < 1 or args.warmup < 0:
        parser.error("use reps, iterations and seeds >= 1 and warmup >= 0")
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise SystemExit("hardware benchmark: unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR")
    if not os.environ.get(LEASE_MARKER):
        raise SystemExit(f"run under devlease.py slice <name>, which sets {LEASE_MARKER}")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit("the served geometry is LNC2: NEURON_LOGICAL_NC_CONFIG must be 2")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise SystemExit("set NEURON_LIBTORCH_CACHE_ROOT to a compile cache directory")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    import vllm_neuron  # registers the Neuron compilation backend
    from vllm_neuron.functional.attention import mla_dense_window, mla_sparse

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_dir is not None:
        args.save_dir = args.save_dir.resolve()
        args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.capture is not None:
        args.capture = args.capture.resolve()
    # The compiler writes its logs to the working directory: keep them with the cache.
    scratch = Path(cache_root) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            LEASE_MARKER, "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "tree": str(ROOT),
        "modules": {m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(
            Path(m.__file__).read_bytes()).hexdigest()} for m in (mla_sparse, mla_dense_window)},
        "geometry": {"chunk": CHUNK, "heads": HEADS, "latent": LATENT, "page": PAGE,
                     "softmax_scale": SCALE, "select_k": SELECT_K, "kpool": KPOOL,
                     "index_cols": WIDTH, "bank_pages": BANK_PAGES_B64},
        "timing_method": {"reps": args.reps, "iterations": args.iterations,
                          "warmup": args.warmup, "order": "rotated every call within a group",
                          "device": "system trace nc_exec_running, physical cores merged",
                          "synchronization": "outputs copied to CPU on every call",
                          "noise_floor": "largest (max - min) / median of a graph's rep medians",
                          "set_spread": "(max - min) / min of a graph's medians over the operand "
                                        "sets, each set timed on its own"},
        "cases": [], "failures": [],
    }
    floor = build_floor()

    def save():
        out.write_text(json.dumps(report, indent=1) + "\n")

    for case in CASES:
        if args.case and case.name not in args.case:
            continue
        row = run_case(case, floor, args, report["failures"])
        report["cases"].append(row)
        save()
        t = row["timing"]
        print(json.dumps({"case": case.name, "gather_us": round(t["gather"]["median_us"], 1),
                          "masked_us": round(t["masked"]["median_us"], 1),
                          "floor_us": round(t["floor"]["median_us"], 1),
                          "masked_over_gather": round(t["masked_over_gather"], 4),
                          "masked_over_entitlement": round(t["masked_over_entitlement"], 3),
                          "noise": round(t["noise_floor"], 4),
                          "set_spread": [round(t[n]["set_spread"], 4) for n in ("gather", "masked")],
                          "identical": [t[n]["identical"] for n in ("gather", "masked")],
                          "emissions": [t[n]["emissions"] for n in ("gather", "masked")]}),
              flush=True)
    save()
    for failure in report["failures"]:
        print(f"FAIL {failure}", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
