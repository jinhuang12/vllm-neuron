# SPDX-License-Identifier: Apache-2.0
"""Time one DSA/MLA layer's decode step for ``B`` requests: batched against 75090b9.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores. For each context (4096 and 8192, ``max_seq_len`` equal to it, so 1024 and
2048 candidate pools) and each ``B`` (1, 4, 16; ``--batches`` adds 64):

* ``before`` -- 75090b9's ``Glm5NextMLAAttention.forward``, once per request: the
  one-request indexer chain (ring step, candidate gather, score GEMM, causal bound,
  top-k, sentinel, order, expand) and ``mla_sparse_attention``. This is how 75090b9
  serves ``B`` requests, and what the earlier batch carrier did for the indexer before
  this branch.
* ``after`` -- this tree's batched layer (``dsa_batch_case.batched_layer``): the
  projections once on ``B`` rows, ``Glm5NextDSAIndexer.forward_requests`` (one ring-step
  launch and one score launch for all requests, then the existing top-k, sentinel,
  order and expand at ``B`` rows) and ``mla_decode_attention`` for all requests.
* ``after_per_request`` (``B = 1`` only) -- this tree's ``forward`` at one request, the
  route a one-request decode step takes: ``forward_requests`` at one request, then the
  one-request attention (``attend`` at ``batch_size`` 1, ``mla_sparse_attention``).

The geometry is the per-rank TP=64 shape (``dsa_decode_case.decode_config``). Request
``b`` is at length ``ctx - b % 4``, so positions mix pool completions and open pools, in
a full ``ctx``-row window of its own shuffled pages. One compiled graph chains
``--layers`` distinct layers (default 11, the model's DSA layer count), each layer's
output feeding the next one's input, and the reported figure is the graph's time
divided by the layer count. ``before`` chains ``min(layers, 16 // B)`` layers (at least
one) so its graph of ``B`` forwards per layer compiles inside one job; above B=16 it is
not run and ``after`` chains ``--wide-layers`` (default 4). Every graph also
returns layer 0's output, which is computed from the same operands in both trees and is
the agreement check.

Each timed call copies the graph's output to CPU to synchronise. Compilation and warmup
are excluded. Pass conditions (asserted at the end, after the JSON is written):
``after(B=16) <= 6 * after(B=1)`` and ``after(B=1) <= 1.1 * before(B=1)`` at each context.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

#: The worktree root, ahead of any installed copy: the lease command sets no PYTHONPATH,
#: and the venv's own ``vllm_neuron`` is another tree. Bytecode is not written, so a run
#: leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.dsa import decode_batch as DB  # noqa: E402
from vllm_neuron.model.glm5_next import model_fp8  # noqa: E402
from test.vllm_neuron.functional.dsa import dsa_batch_case as batch_case  # noqa: E402
from test.vllm_neuron.functional.dsa import dsa_decode_case as case  # noqa: E402

DEVICE = "neuron:0"
CONTEXTS = (4096, 8192)
BATCHES = (1, 4, 16)
#: The forwards one ``before`` graph may chain (``B`` per layer).
BEFORE_FORWARDS = 16
SCALING_LIMIT = 6.0
B1_LIMIT = 1.1
#: Before vs after agreement on the same operands over the same chained layers. Both
#: select the same pools; the residue is fp8 projection rounding at one row against
#: ``B`` rows and the bf16 cast of two attention outputs that agree to ~1e-7.
AGREE_REL_L2 = 3e-2
NAMES = ("latent_cache", "pool_bank", "tail_bank", "state_slots", "seq_lens", "position",
         "block_table", "latent_slots")
BEFORE_NAMES = ("latent_cache", "pool_cache", "seq_lens", "start_position",
                "block_table_row", "latent_slots", "tail", "position")


def load_baseline(directory: Path):
    """The baseline package's ``load()``: 75090b9's modules, from its committed copies."""
    spec = importlib.util.spec_from_file_location(
        "dsa_batched_baseline_loader", directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load()


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def first_call(fn, inputs):
    started = time.perf_counter()
    out = fn(*inputs).to("cpu")
    return out, time.perf_counter() - started


def measure(fn, inputs, per: int, warmup: int, iterations: int) -> dict:
    """Median and p90 of one unit (graph time / ``per``), in microseconds."""
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0 / per)
    samples.sort()
    return {
        "iterations": iterations,
        "units_per_sample": per,
        "median_us": statistics.median(samples),
        "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_us": samples[0],
        "max_us": samples[-1],
    }


def _fresh(tensor: torch.Tensor) -> torch.Tensor:
    """A private device copy: every graph writes its banks in place, and no two graphs
    may share a bank or start from another's writes."""
    return tensor.to(DEVICE, copy=True)


def _lengths(ctx: int, batch: int) -> list[int]:
    return [ctx - b % 4 for b in range(batch)]


def _layer_operands(cfg, ctx, batch, layers):
    """One batch of operands per layer, CPU-side, same draws for both trees."""
    return [batch_case.batch_operands(cfg, _lengths(ctx, batch), max_seq_len=ctx,
                                      seed=99 + 7 * index)
            for index in range(layers)]


def _after_graph(layers, statics):
    """``step(hidden, per-layer operands...)`` -> ``[2, B, hidden]``: layer 0's output and
    the last layer's."""
    def step(hidden, *flat):
        first = None
        for index, layer in enumerate(layers):
            ops = dict(zip(NAMES, flat[len(NAMES) * index:len(NAMES) * (index + 1)]))
            ops.update(statics)
            ops["hidden"] = hidden
            hidden = batch_case.batched_layer(layer, ops)
            if first is None:
                first = hidden
        return torch.stack([first, hidden], 0)
    return step


def _after_inputs(per_layer):
    flat = [_fresh(per_layer[0]["hidden"])]
    for ops in per_layer:
        flat.extend(_fresh(ops[name]) for name in NAMES)
    return tuple(flat)


def _before_graph(layers, statics, batch):
    """``step(hidden..., per-layer-per-request operands...)`` -> ``[2, B, hidden]``, as
    :func:`_after_graph`."""
    def step(*flat):
        hidden = list(flat[:batch])
        rest = flat[batch:]
        at = 0
        first = None
        for layer in layers:
            for b in range(batch):
                ops = dict(zip(BEFORE_NAMES, rest[at:at + len(BEFORE_NAMES)]))
                at += len(BEFORE_NAMES)
                hidden[b] = layer.forward(hidden[b], **ops, **statics)
            if first is None:
                first = torch.cat(hidden, 0)
        return torch.stack([first, torch.cat(hidden, 0)], 0)
    return step


def _before_inputs(per_layer, batch):
    """Each request's own views of the same draws: its ring and its store, its table."""
    hidden = [_fresh(per_layer[0]["hidden"][b:b + 1]) for b in range(batch)]
    rest = []
    for ops in per_layer:
        cache = _fresh(ops["latent_cache"])
        for b in range(batch):
            slot = int(ops["state_slots"][b])
            rest.extend([
                cache,
                _fresh(ops["pool_bank"][slot]),
                _fresh(ops["seq_lens"][b:b + 1]),
                _fresh(ops["position"][b]),
                _fresh(ops["block_table"][b].reshape(-1, 1).contiguous()),
                _fresh(ops["latent_slots"][b:b + 1]),
                _fresh(ops["tail_bank"][slot]),
                _fresh(ops["position"][b]),
            ])
    return tuple(hidden + rest)


def _timed(name, graph, inputs, per, args, timing, outputs):
    torch._dynamo.reset()
    fn = compiled(graph)
    outputs[name], seconds = first_call(fn, inputs)
    timing[name] = measure(fn, inputs, per, args.warmup, args.iterations)
    timing[name]["first_call_s"] = seconds
    timing[name]["layers_chained"] = per


def layer_case(ctx, batch, base, args) -> dict:
    cfg = case.decode_config()
    statics = {"softmax_scale": float(int(cfg.qk_nope_head_dim)
                                      + int(cfg.qk_rope_head_dim)) ** -0.5,
               "max_seq_len": ctx}
    after_layers = args.layers if batch <= BEFORE_FORWARDS else args.wide_layers
    before_layers = max(1, min(args.layers, BEFORE_FORWARDS // batch))
    with_before = batch <= BEFORE_FORWARDS
    per_layer = _layer_operands(cfg, ctx, batch, after_layers)
    timing, outputs = {}, {}

    live = [case.build_attention(model_fp8, cfg, seed=4242 + 31 * index, device=DEVICE)
            for index in range(after_layers)]
    DB.reset_decode_batch_dispatch_counters()
    _timed("after", _after_graph(live, statics), _after_inputs(per_layer), after_layers,
           args, timing, outputs)
    dispatched = DB.decode_batch_dispatch_counters()
    if batch == 1:
        _timed("after_per_request",
               _before_graph(live, {**statics, "page_size": case.PAGE}, batch),
               _before_inputs(per_layer, batch), after_layers, args, timing, outputs)
    del live

    if with_before:
        old = [case.build_attention(base.model_fp8, cfg, seed=4242 + 31 * index,
                                    device=DEVICE)
               for index in range(before_layers)]
        _timed("before", _before_graph(old, {**statics, "page_size": case.PAGE}, batch),
               _before_inputs(per_layer[:before_layers], batch), before_layers, args,
               timing, outputs)
        del old

    for name, out in outputs.items():
        if not torch.isfinite(out.float()).all():
            raise AssertionError(f"ctx {ctx} B={batch}: {name} returned nonfinite values")
    # Layer 0 runs on the same operands in every graph, so its outputs are comparable;
    # past it each tree feeds its own output forward, and where selection keeps half the
    # pools a bf16 ulp in the index query moves pools across the cut, so the chained
    # outputs are recorded, not bounded (``benchmark_dsa_decode.py`` says the same).
    agreement = chained = None
    if with_before:
        agreement = rel_l2(outputs["after"][0], outputs["before"][0])
        if agreement > AGREE_REL_L2:
            raise AssertionError(f"ctx {ctx} B={batch}: layer 0 after vs before relative "
                                 f"L2 {agreement}")
        if before_layers == after_layers:
            chained = rel_l2(outputs["after"][1], outputs["before"][1])
    row = {
        "table": "layer",
        "ctx": ctx,
        "max_seq_len": ctx,
        "candidates": ctx // int(cfg.index_kpool),
        "batch": batch,
        "lengths": _lengths(ctx, batch),
        "unit": "one layer's decode step for all `batch` requests",
        "layer0_after_vs_before_relative_l2": agreement,
        "chained_after_vs_before_relative_l2": chained,
        "decode_batch_dispatch_counters_while_tracing_after": list(dispatched),
        "before": None,
        **timing,
        "after_us_per_request": timing["after"]["median_us"] / batch,
    }
    if with_before:
        row["after_over_before_median"] = (timing["after"]["median_us"]
                                           / timing["before"]["median_us"])
        row["before_us_per_request"] = timing["before"]["median_us"] / batch
    if batch == 1:
        row["after_over_after_per_request_median"] = (
            timing["after"]["median_us"] / timing["after_per_request"]["median_us"])
    return row


def verdicts(cases: list[dict]) -> dict:
    """The pass conditions, per context, from whatever rows the report holds."""
    out = {}
    for ctx in sorted({row["ctx"] for row in cases}):
        by_batch = {row["batch"]: row for row in cases if row["ctx"] == ctx}
        one = by_batch.get(1)
        entry = {}
        if one is not None:
            entry["after_b1_over_before_b1"] = (
                one["after"]["median_us"] / one["before"]["median_us"])
            entry["b1_pass"] = entry["after_b1_over_before_b1"] <= B1_LIMIT
            for batch, row in sorted(by_batch.items()):
                if batch > 1:
                    ratio = row["after"]["median_us"] / one["after"]["median_us"]
                    entry[f"after_b{batch}_over_after_b1"] = ratio
                    if row["before"] is not None:
                        before = row["before"]["median_us"] / one["before"]["median_us"]
                        entry[f"before_b{batch}_over_before_b1"] = before
            if 16 in by_batch:
                entry["scaling_pass"] = entry["after_b16_over_after_b1"] <= SCALING_LIMIT
        out[str(ctx)] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--merge", action="store_true",
                        help="keep the output file's rows for cases this run skips")
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--wide-layers", type=int, default=4,
                        help="layers chained above B=16, where 'before' is not run")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    if args.layers < 1 or args.wide_layers < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use positive chain lengths and iteration counts")
    baseline_dir = args.baseline_module.resolve()
    output = args.output.resolve()
    base = load_baseline(baseline_dir)
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    kept = []
    if args.merge and output.exists():
        wanted = {(c, b) for c in args.contexts for b in args.batches}
        kept = [row for row in json.loads(output.read_text()).get("cases", [])
                if (row["ctx"], row["batch"]) not in wanted]
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in ("NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                        "NEURON_CC_FLAGS", "NEURON_PLATFORM_TARGET_OVERRIDE",
                        "NEURON_LIBTORCH_CACHE_ROOT")
        },
        "tree": str(ROOT),
        "baseline": {"commit": base.commit, "directory": base.directory,
                     "loader": str(baseline_dir)},
        "args": {"layers": args.layers, "wide_layers": args.wide_layers,
                 "warmup": args.warmup,
                 "iterations": args.iterations, "contexts": args.contexts,
                 "batches": args.batches, "merge": args.merge},
        "kernels": {"ring_step": "decode_batch.dsa_decode_ring_step_kernel",
                    "scores": "decode_batch.dsa_decode_scores_kernel"},
        "synchronization": "graph output copied to CPU on every timed call",
        "cases": list(kept),
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        report["cases"].append(row)
        report["cases"].sort(key=lambda r: (r["ctx"], r["batch"]))
        report["verdicts"] = verdicts(report["cases"])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"ctx": row["ctx"], "batch": row["batch"],
                          "before_median_us": (row["before"] or {}).get("median_us"),
                          "after_median_us": row["after"]["median_us"],
                          "after_over_before": row.get("after_over_before_median"),
                          "agreement": row["layer0_after_vs_before_relative_l2"]}),
              flush=True)

    for ctx in args.contexts:
        for batch in args.batches:
            record(layer_case(ctx, batch, base, args))
    report["verdicts"] = verdicts(report["cases"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdicts"]), flush=True)
    failed = [f"ctx {ctx}: {name}" for ctx, entry in report["verdicts"].items()
              for name in ("b1_pass", "scaling_pass") if entry.get(name) is False]
    if failed:
        raise AssertionError(f"pass conditions failed: {failed}")


if __name__ == "__main__":
    main()
