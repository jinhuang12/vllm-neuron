# SPDX-License-Identifier: Apache-2.0
"""Time the sparse MLA kernel at its served shapes on Neuron, base module against this tree.

Run it ONLY through the device lease, which pins the cores and sets LNC2; the script refuses
to run without them and selects no cores itself::

    export NEURON_LIBTORCH_CACHE_ROOT=<an empty directory>
    python3 /home/ubuntu/glm53f-wt/devlease.py slice <slice> -- python3 \\
        test/hardware/benchmark_sparse_mla_prefill.py --base-module <base>/mla_sparse.py \\
        --out bench.json

``--base-module`` is the base revision's ``vllm_neuron/functional/attention/mla_sparse.py``
(for example from ``git show <rev>:<path>``); the script loads it under its own module
name, so both kernels run in one process and one lease.

Geometry: the per-rank TP=64 shape of GLM-5.3-Flash DSA attention. One head, latent 512,
page 128, bf16 query and cache, softmax scale 1/16, the selector's 2176-column index rows
(512 pools of 4 rows in score order, the tail of the last open pool, then -1 to a whole
number of 128-column chunks). A case is one call of the model's seam,
``mla_sparse_attention(q, bank, indices, scale, block_table_row=..., written=...,
write_offset=..., page_size=128)``, as ``Glm5NextMLAAttention.attend`` makes it: the window
is ``window_pages`` pages of a bank of ``bank_pages`` pages named by a random block table,
and the call's own ``tokens`` rows are overlaid at window row ``start``. Query ``t`` sits at
position ``start + t`` and sees rows ``0 .. start + t``. The served cases (:data:`CASES`):

* ``p1k_chunk1``: the 1024-token prompt of the 8k line (``max_model_len`` 8192): 64 window
  pages, chunk 1, so most index columns of the early queries are -1 (the chunk-1 case).
* ``p64k_first`` / ``p64k_last``: the first and the last 2048-row chunk of a 65534-token
  prompt on the 64k line (512 window pages).
* ``decode_8k`` / ``decode_64k``: one decode token of a single request at context 8192 and
  65534, the shape the one-request decode step hands this kernel when it selects.
* ``rows64_8k``: 64 query rows at context 8192. No served graph makes this call (a decode
  step of 64 requests runs ``mla_decode_attention``); it covers the multi-query path at a
  decode-sized row count.

Per case: both graphs are compiled and checked against a float64 oracle, within the
low-precision body's derived error budget (:func:`error_budget`), then timed in
``--reps`` repeats of ``--iterations`` calls each, the two graphs interleaved (the order
flips every call). A sample is one dispatch with the output copied to CPU; its device time
is the runtime system trace's ``nc_exec_running`` interval with the two physical cores
merged. ``--seeds`` more operand sets per case are run once each and the outputs compared
bit for bit; ``--save-dir`` keeps each output as a ``.pt`` file.

Each case carries its ENTITLEMENT, computed by the inventory's rule
(``glm53f-wt3/inventory/entitle.py`` ``kernel_cost``, branch
``mla_sparse_attention_nope_row_tiled_kernel``): FLOPs ``2 * 2 * T * S * D`` and bytes = the
non-cache operands and the fp32 output plus the unique cache rows the call can touch,
``min(bank rows, valid KV) * D * 2``, at the per-logical-core rates below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

HEADS, LATENT, PAGE, SCALE = 1, 512, 128, 0.0625
#: The DSA selector's dials at GLM-5.3-Flash: ``index_topk`` 2048 rows in pools of 4.
KPOOL, SELECT_K = 4, 512
#: Index columns the selector emits: 512 pools of 4, the 3-row tail, -1 to 17 x 128.
WIDTH = ((SELECT_K * KPOOL + KPOOL - 1 + 127) // 128) * 128
#: Rates as the inventory cites them (``glm53f-wt3/planner/configs/trn2_constants.json``
#: ``te_peak_flops``, ``hbm_bw_bytes_per_s``). 79e12 is one physical TensorEngine's peak.
TE_PEAK_FLOPS = 79e12
HBM_BYTES_PER_S = 716e9
#: The MEASURED bf16 matmul peak of one logical core (two physical cores at LNC2), the
#: campaign's ship-bar rate; a FLOP-bound case is shown against both peaks.
LNC2_MEASURED_PEAK_FLOPS = 153.0e12
#: The fp32 unit roundoff.
FP32_UNIT = 2.0 ** -24
#: The bf16 unit roundoff: 8 significand bits, round to nearest.
BF16_UNIT = 2.0 ** -8
LEASE_MARKER = "NEURON_RT_VISIBLE_CORES"


@dataclass(frozen=True)
class Case:
    """One seam call: ``tokens`` query rows at window rows ``start ..``, ``kv_valid`` rows seen."""

    name: str
    tokens: int
    window_pages: int
    bank_pages: int
    start: int
    kv_valid: int
    served: str


#: ``bank_pages`` is the rank's latent bank at each line (the graph operand
#: ``bf16[bank_pages * 128, 512]`` of glm53f-wt3/reports/op_inventory.json P-A and P-B).
CASES = (
    Case("p1k_chunk1", 1024, 64, 69, 0, 1024, "prefill, 8k line, chunk 1 of a 1024-token prompt"),
    Case("p64k_first", 2048, 512, 517, 0, 2048, "prefill, 64k line, chunk 1 of 32"),
    Case("p64k_last", 2048, 512, 517, 65534 - 2048, 65534, "prefill, 64k line, chunk 32 of 32"),
    Case("decode_8k", 1, 64, 69, 8191, 8192, "decode, one request, context 8192"),
    Case("decode_64k", 1, 512, 517, 65533, 65534, "decode, one request, context 65534"),
    Case("rows64_8k", 64, 64, 69, 8192 - 64, 8192, "not served: 64 rows at context 8192"),
)


def selection(case: Case, gen):
    """``[tokens, WIDTH]`` int32 index rows laid out as ``dsa_index_expand`` emits them.

    Query ``t`` sees ``seen = start + t + 1`` rows: the complete pools
    ``0 .. seen // KPOOL - 1`` (all of them when there are at most ``SELECT_K``, else
    ``SELECT_K`` drawn at random), each expanded to its ``KPOOL`` rows, in a random (score)
    order; then the rows of the open pool; then -1.
    """
    import torch

    out = torch.full((case.tokens, WIDTH), -1, dtype=torch.int32)
    for t in range(case.tokens):
        seen = case.start + t + 1
        n_pools = seen // KPOOL
        chosen = torch.randperm(n_pools, generator=gen)[:SELECT_K]
        rows = (chosen.reshape(-1, 1) * KPOOL + torch.arange(KPOOL).reshape(1, -1)).reshape(-1)
        row = torch.cat([rows, torch.arange(n_pools * KPOOL, seen)])
        out[t, :row.numel()] = row.to(torch.int32)
    return out


def make_operands(case: Case, seed: int):
    """``(q, bank, indices, table, written, offset, window)`` for one case and seed.

    ``window`` is the cache the call attends (pages in table order, this call's rows
    overlaid), for the oracle only.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(case.tokens, HEADS, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    bank = torch.randn(case.bank_pages * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(case.bank_pages, generator=gen)[:case.window_pages]
    table = table.reshape(case.window_pages, 1).to(torch.int32)
    written = torch.randn(case.tokens, LATENT, generator=gen).to(torch.bfloat16)
    offset = torch.tensor([[case.start]], dtype=torch.int32)
    indices = selection(case, gen)
    window = bank.reshape(-1, PAGE, LATENT)[table[:, 0].long()]
    window = window.reshape(case.window_pages * PAGE, LATENT).clone()
    window[case.start:case.start + case.tokens] = written
    return q, bank, indices, table, written, offset, window


def entitlement(case: Case) -> dict:
    """The inventory's per-call roofline for this case, in us, with its terms."""
    t, s, d = case.tokens * HEADS, WIDTH, LATENT
    flops = 2 * 2 * t * s * d
    kv_rows = min(case.bank_pages * PAGE, case.kv_valid)
    kv_bytes = kv_rows * d * 2
    other = (t * d * 2 + case.tokens * WIDTH * 4 + case.window_pages * 4 + case.tokens * d * 2
             + 4 + t * d * 4)
    nbytes = kv_bytes + other
    flop_us, byte_us = flops / TE_PEAK_FLOPS * 1e6, nbytes / HBM_BYTES_PER_S * 1e6
    flop_us_measured = flops / LNC2_MEASURED_PEAK_FLOPS * 1e6
    return {"T": t, "S": s, "D": d, "kv_rows": kv_rows, "bytes": nbytes, "flops": flops,
            "flop_us": flop_us, "byte_us": byte_us, "entitlement_us": max(flop_us, byte_us),
            "binds": "FLOPs" if flop_us >= byte_us else "bytes",
            "flop_us_measured_peak": flop_us_measured,
            "entitlement_us_measured_peak": max(flop_us_measured, byte_us),
            "binds_measured_peak": "FLOPs" if flop_us_measured >= byte_us else "bytes",
            "formula": "max(bytes / 716e9, 2*2*T*S*D / peak), peak 79e12 (inventory) and "
                       "153.0e12 (MEASURED LNC2, ship bar); bytes = min(bank rows, valid KV)"
                       " x D x 2 + q + indices + table + written + offset + fp32 output"}


def oracle(q, window, indices):
    """Float64 reference: softmax over each query's selected rows, -1 masked, duplicates kept."""
    import torch

    qd, cd = q.double(), window.double()
    rows = indices.to(torch.int64)
    out = torch.zeros(q.shape, dtype=torch.float64)
    for s in range(q.shape[0]):
        keep = rows[s] >= 0
        if not bool(keep.any()):
            continue
        gathered = cd[rows[s][keep]]
        weights = torch.softmax(torch.einsum("hl,kl->hk", qd[s], gathered) * SCALE, dim=-1)
        out[s] = weights @ gathered
    return out


def error_budget(selected: int, latent: int, max_abs_logit: float) -> float:
    """The low-precision body's error against float64 attention: relative L2 over the output.

    One term per step that rounds, each the size of that step's error as a fraction of
    the output. ``n`` roundings of relative size ``u`` in one sum move it by about
    ``sqrt(n) * u`` (random-walk rounding), and a relative error of every probability is
    the same relative error of the output, which is their weighted mean. The sum terms
    are that root-mean-square model, which a relative L2 over the whole output measures;
    the split term is a worst-case bound on every element. ``selected`` is
    the most rows a query selects, ``max_abs_logit`` the largest scaled score a selected
    row gets, ``|softmax_scale * q . c|``. The products of MM1 and MM2 are exact: both
    take 2-byte operands and accumulate in fp32.

    * MM2's moving operand, p split into two bf16 halves: hi + lo is p to within
      ``BF16_UNIT ** 2`` relative (hi rounds p, lo rounds p - hi, which is exact in fp32);
    * MM2's fp32 sums over the ``selected`` rows, and the add of the hi and lo sums;
    * the softmax's fp32 sum over the same rows, which scales the whole output row;
    * MM1's fp32 sum over the latent: an absolute error of ``sqrt(latent) * u`` times the
      score in each scaled score, which the exp turns into that relative error of p;
    * the exp's argument ``scale * x - scale * max``: three roundings of values at most
      ``max_abs_logit``, absolute;
    * the exp, the reciprocal of the sum and the normalising multiply: one ``u`` each.

    A change of summation order inside any of these sums stays inside the budget. The
    fp32 body, which serves one-row calls, is held to the same bar.
    """
    split = BF16_UNIT ** 2
    mm2 = selected ** 0.5 * FP32_UNIT + FP32_UNIT
    row_sum = selected ** 0.5 * FP32_UNIT
    mm1 = latent ** 0.5 * FP32_UNIT * max_abs_logit
    exp_argument = 3 * FP32_UNIT * max_abs_logit
    pointwise = 3 * FP32_UNIT
    return split + mm2 + row_sum + mm1 + exp_argument + pointwise


def operand_budget(q, window, indices) -> dict:
    """:func:`error_budget` of these operands, with its two data terms."""
    import torch

    qd, cd = q.double(), window.double()
    rows = indices.to(torch.int64)
    keep = rows >= 0
    largest = 0.0
    for s in range(q.shape[0]):
        if bool(keep[s].any()):
            scores = torch.einsum("hl,kl->hk", qd[s], cd[rows[s][keep[s]]]) * SCALE
            largest = max(largest, float(scores.abs().max()))
    selected = int(keep.sum(dim=1).max())
    return {"selected": selected, "max_abs_logit": largest,
            "rel_l2": error_budget(selected, LATENT, largest)}


def rel_l2(got, want) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def time_interleaved(graphs: dict, inputs: tuple, iterations: int) -> dict:
    """One repeat: ``iterations`` calls of each graph, the order flipped every call."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    order = []
    with SystemTraceSession() as trace:
        for it in range(iterations):
            for name in (names if it % 2 == 0 else names[::-1]):
                graphs[name](*inputs).to("cpu")
                order.append(name)
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != len(order):
        raise AssertionError(f"the trace has {len(device)} executions for {len(order)} calls")
    per = {name: [] for name in names}
    for name, value in zip(order, device):
        per[name].append(value)
    return {name: summary(values) for name, values in per.items()}


def run_case(case: Case, modules: dict, args) -> dict:
    import torch

    # Every case compiles the same Python function again; dynamo's per-function recompile
    # limit would otherwise end a run of all cases.
    torch._dynamo.reset()

    def graph_of(module):
        attend = module.mla_sparse_attention

        def call(q, bank, indices, table, written, offset):
            return attend(q, bank, indices, SCALE, block_table_row=table, written=written,
                          write_offset=offset, page_size=PAGE)

        return torch.compile(call, backend="neuron_libtorch", fullgraph=True, dynamic=False)

    graphs = {name: graph_of(module) for name, module in modules.items()}
    numerics, first_call_s = [], {}
    timed_inputs = None
    for seed_index in range(args.seeds):
        seed = 1000 * seed_index + case.tokens + case.kv_valid
        q, bank, indices, table, written, offset, window = make_operands(case, seed)
        inputs = tuple(t.to("neuron:0") for t in (q, bank, indices, table, written, offset))
        outputs = {}
        for name, graph in graphs.items():
            started = time.perf_counter()
            outputs[name] = graph(*inputs).to("cpu")
            first_call_s.setdefault(name, time.perf_counter() - started)
        want = oracle(q, window, indices)
        bar = operand_budget(q, window, indices)
        row = {"seed": seed, "error_budget": bar,
               "bit_equal": bool(torch.equal(outputs["base"], outputs["after"])),
               "max_abs_after_vs_base": float((outputs["after"] - outputs["base"]).abs().max()),
               "rel_l2_after_vs_base": rel_l2(outputs["after"], outputs["base"])}
        for name, got in outputs.items():
            row[f"rel_l2_{name}_vs_float64"] = rel_l2(got, want)
            if not torch.isfinite(got).all() or row[f"rel_l2_{name}_vs_float64"] > bar["rel_l2"]:
                raise AssertionError(f"{case.name} seed {seed}: {name} {row}")
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for name, got in outputs.items():
                torch.save(got, args.save_dir / f"{case.name}_seed{seed}_{name}.pt")
        numerics.append(row)
        if timed_inputs is None:
            timed_inputs = inputs
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph(*timed_inputs).to("cpu")
    reps = [time_interleaved(graphs, timed_inputs, args.iterations) for _ in range(args.reps)]
    medians = {name: [r[name]["median_us"] for r in reps] for name in graphs}
    budget = entitlement(case)
    timing = {}
    for name, values in medians.items():
        median = statistics.median(values)
        timing[name] = {"rep_medians_us": values, "median_us": median,
                        "rep_spread": (max(values) - min(values)) / median,
                        "over_entitlement": median / budget["entitlement_us"],
                        "over_entitlement_measured_peak":
                            median / budget["entitlement_us_measured_peak"]}
    timing["after_over_base"] = timing["after"]["median_us"] / timing["base"]["median_us"]
    timing["noise_floor"] = max(timing["base"]["rep_spread"], timing["after"]["rep_spread"])
    return {"case": asdict(case), "entitlement": budget, "timing": timing, "reps": reps,
            "numerics": numerics, "first_call_s": first_call_s}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-module", type=Path, required=True,
                        help="the base revision's mla_sparse.py")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES],
                        help="case name (repeatable; default: every case)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6, help="calls per graph per repeat")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3, help="operand sets checked per case")
    parser.add_argument("--save-dir", type=Path, help="keep every checked output as .pt here")
    args = parser.parse_args()
    if args.reps < 1 or args.iterations < 1 or args.warmup < 0 or args.seeds < 1:
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
    from vllm_neuron.functional.attention import mla_sparse as live

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    base_path = args.base_module.resolve()
    modules = {"base": load_module(base_path, "mla_sparse_benchmark_base"), "after": live}
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
            LEASE_MARKER, "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "tree": str(ROOT),
        "modules": {name: {"path": m.__file__,
                           "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
                    for name, m in modules.items()},
        "shape": {"heads": HEADS, "latent": LATENT, "page": PAGE, "softmax_scale": SCALE,
                  "index_width": WIDTH, "kpool": KPOOL, "select_k": SELECT_K,
                  "dtype": "bfloat16"},
        "rates": {"te_peak_flops": TE_PEAK_FLOPS, "hbm_bytes_per_s": HBM_BYTES_PER_S,
                  "lnc2_measured_peak_flops": LNC2_MEASURED_PEAK_FLOPS,
                  "source": "glm53f-wt3/planner/configs/trn2_constants.json; LNC2 peak MEASURED"
                            " (team-lead ruling, final)"},
        "timing_method": {"reps": args.reps, "iterations": args.iterations,
                          "warmup": args.warmup, "order": "base/after flipped every call",
                          "device": "system trace nc_exec_running, physical cores merged",
                          "synchronization": "output copied to CPU on every call",
                          "noise_floor": "largest (max - min) / median of a variant's rep medians"},
        "cases": [],
    }
    for case in CASES:
        if args.case and case.name not in args.case:
            continue
        row = run_case(case, modules, args)
        report["cases"].append(row)
        out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({"case": case.name,
                          "base_us": round(row["timing"]["base"]["median_us"], 1),
                          "after_us": round(row["timing"]["after"]["median_us"], 1),
                          "after_over_base": round(row["timing"]["after_over_base"], 4),
                          "noise": round(row["timing"]["noise_floor"], 4),
                          "bit_equal": [n["bit_equal"] for n in row["numerics"]]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
