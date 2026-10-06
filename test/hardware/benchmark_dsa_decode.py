# SPDX-License-Identifier: Apache-2.0
"""Time one DSA/MLA layer's decode step on Neuron against commit 5938748.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores. Three tables are written, each as before (5938748) and after (this tree):

* ``layer`` -- ``Glm5NextMLAAttention.forward`` (indexer, projections, attention,
  ``o_proj``) at the per-rank TP=64 shape: ctx 1024 in a 2048-row window (the bypass)
  and ctx 4096 in a 4096-row window (selection), both with ``max_seq_len`` 4096, the
  runner's model length; plus ctx 1024 with 5938748 in the 4096-row window the runner
  uses when no decode context bucket is set. One compiled graph chains ``--layers``
  distinct layers (default 11, the model's DSA layer count), each layer's output
  feeding the next one's input, and the reported figure is the graph's time divided by
  the layer count. B=4 is four requests, one forward each per layer: the layer seam
  takes one request in both trees (the runner's batch-one refusal is wave 2's).
* ``attention`` -- the decode attention kernel alone, batched: ``mla_decode_attention``
  at B in {1, 4} in one call against 5938748's ``mla_sparse_attention`` once per
  request.
* ``projection`` -- each DSA projection site at M in {1, 4} rows, and at the 1024-row
  prefill chunk (the prefill leg takes the same route): the fp8/bf16 kernel against
  5938748's fp32 kernel on the dequantised weight.

The default run writes the layer table; ``--tables attention projection`` writes the
other two (a separate leased job, each under ten minutes once the cache is warm).
Compilation and warmup are excluded. Each timed call copies the graph's output to CPU
to synchronise, so a sample includes one dispatch; chaining amortises it. Before and
after outputs are compared on device data before any timing.
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
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402
from vllm_neuron.functional.attention import mla_decode as MD  # noqa: E402
from vllm_neuron.functional.attention import mla_projections as MP  # noqa: E402
from vllm_neuron.model.glm5_next import model_fp8  # noqa: E402
from test.vllm_neuron.functional.dsa import dsa_decode_case as case  # noqa: E402

DEVICE = "neuron:0"
MAX_SEQ_LEN = 4096
#: (label, ctx, before's window pages, after's window pages, B=1 only). The first two
#: compare one graph shape. The third is what serving changes: 5938748 decodes in the
#: runner's default max_model_len window (32 pages), and the bypass needs a 2048-row
#: decode context bucket.
LAYER_CASES = (("bypass", 1024, 16, 16, False), ("selected", 4096, 32, 32, False),
               ("bypass_vs_default_window", 1024, 32, 16, True))
#: (site, in, out, fp8) at TP=64, one MLA head per rank.
SITES = (("q_a_proj", 4096, 1536, True), ("q_b_proj", 1536, 256, True),
         ("kv_a_proj_with_mqa", 4096, 512, True), ("o_proj", 256, 4096, True),
         ("wq_b", 1536, 4096, False), ("wk", 4096, 128, False),
         ("weights_proj", 4096, 32, False))
#: Before vs after agreement, chained, asserted where both trees attend the same rows
#: (the bypass cases: selection there keeps every token): each layer carries ~5e-3
#: relative L2 from the fp8 weights read as stored (``test_decode_bypass.py``). Where
#: selection drops half the pools the index query's fp8 rounding moves a few pools
#: across the cut, and over eleven chained layers of random weights the outputs part
#: ways; that case records its agreement and asserts only finite outputs
#: (``test_selected_decode_keeps_5938748s_selection`` bounds one layer).
LAYER_REL_L2 = 3e-2
KERNEL_REL_L2 = 4e-3


def load_baseline(directory: Path):
    """The baseline package's ``load()``: 5938748's modules, read by ``git show``."""
    spec = importlib.util.spec_from_file_location(
        "dsa_baseline_loader", directory / "__init__.py",
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
    """The first call compiles (or loads from the cache); its wall time is reported."""
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


def compare(before: dict, after: dict) -> dict:
    return {
        "before": before,
        "after": after,
        "median_speedup": before["median_us"] / after["median_us"],
        "after_over_before_median": after["median_us"] / before["median_us"],
    }


def _one_program(timing: dict) -> dict:
    if "after_one_program" not in timing:
        return {}
    one = timing["after_one_program"]
    return {"after_one_program": one,
            "two_over_one_program_median": timing["after"]["median_us"] / one["median_us"]}


# --- layer ---------------------------------------------------------------------------

def _layer_graph(layers, statics, batch):
    """``step(hidden..., per-layer-per-request operands...)`` -> ``[batch, hidden]``."""
    names = ("latent_cache", "pool_cache", "seq_lens", "start_position", "block_table_row",
             "latent_slots", "tail", "position")

    def step(*flat):
        hidden = list(flat[:batch])
        rest = flat[batch:]
        at = 0
        for layer in layers:
            for b in range(batch):
                ops = dict(zip(names, rest[at:at + len(names)]))
                at += len(names)
                hidden[b] = layer.forward(hidden[b], **ops, **statics)
        return torch.cat(hidden, 0)

    return step, names


def layer_case(label, context, pages_before, pages_after, batch, base, args) -> dict:
    torch._dynamo.reset()
    cfg = case.decode_config()
    statics = {"softmax_scale": float(int(cfg.qk_nope_head_dim)
                                      + int(cfg.qk_rope_head_dim)) ** -0.5,
               "max_seq_len": MAX_SEQ_LEN, "page_size": case.PAGE}
    per_tree = {}
    outputs = {}
    for tree, module, pages in (("before", base.model_fp8, pages_before),
                                ("after", model_fp8, pages_after)):
        layers = [case.build_attention(module, cfg, seed=4242 + 31 * index, device=DEVICE)
                  for index in range(args.layers)]
        step, names = _layer_graph(layers, statics, batch)
        hidden, rest = [], []
        for index in range(args.layers):
            for b in range(batch):
                # One draw at the wider window for both trees; the narrower one is
                # the same table cut short (its tail is -1 padding either way).
                ops = case.decode_operands(cfg, context,
                                           window_pages=max(pages_before, pages_after),
                                           max_seq_len=MAX_SEQ_LEN,
                                           seed=99 + 7 * index + 1009 * b)
                ops["block_table_row"] = ops["block_table_row"][:pages].contiguous()
                if index == 0:
                    hidden.append(ops["normed_hidden_states"].to(DEVICE))
                rest.extend(ops[name].to(DEVICE) for name in names)
        inputs = tuple(hidden + rest)
        graph = compiled(step)
        outputs[tree], seconds = first_call(graph, inputs)
        per_tree[tree] = measure(graph, inputs, args.layers, args.warmup, args.iterations)
        per_tree[tree]["first_call_s"] = seconds
        del graph, layers
    for tree, out in outputs.items():
        if not torch.isfinite(out.float()).all():
            raise AssertionError(f"{label} B={batch}: {tree} returned nonfinite values")
    agreement = rel_l2(outputs["after"], outputs["before"])
    if label.startswith("bypass") and agreement > LAYER_REL_L2:
        raise AssertionError(f"{label} B={batch}: after vs before relative L2 {agreement}")
    return {
        "table": "layer",
        "case": label,
        "ctx": context,
        "window_rows": {"before": pages_before * case.PAGE,
                        "after": pages_after * case.PAGE},
        "max_seq_len": MAX_SEQ_LEN,
        "batch": batch,
        "layers_chained": args.layers,
        "unit": "one layer's decode step for all `batch` requests",
        "after_vs_before_relative_l2": agreement,
        "agreement_asserted": label.startswith("bypass"),
        **compare(per_tree["before"], per_tree["after"]),
    }


# --- attention -----------------------------------------------------------------------

def _attention_operands(batch, pages, context, selected, seed):
    gen = torch.Generator().manual_seed(seed)
    bank_pages = pages * batch + 3
    bank = (torch.randn(bank_pages * case.PAGE, 512, generator=gen) * 0.5).to(torch.bfloat16)
    order = torch.randperm(bank_pages, generator=gen)
    table = order[:pages * batch].reshape(batch, pages).to(torch.int32)
    used = -(-context // case.PAGE)
    table[:, used:] = -1
    q = torch.randn(batch, 1, 512, generator=gen).to(torch.bfloat16)
    written = (torch.randn(batch, 512, generator=gen) * 0.5).to(torch.bfloat16)
    pos = torch.full((batch,), context - 1, dtype=torch.int32)
    cols = torch.arange(2048, dtype=torch.int32).expand(batch, 2048)
    if selected:
        pick = torch.stack([torch.randperm(context, generator=gen)[:2048]
                            for _ in range(batch)])
        indices = pick.to(torch.int32)
    else:
        indices = torch.where(cols < context, cols, torch.full_like(cols, -1))
    return q, bank, table, pos, written, indices


def attention_case(mode, context, pages, batch, base, args) -> dict:
    torch._dynamo.reset()
    selected = mode == "selected"
    chain = args.kernel_chain
    sets = [_attention_operands(batch, pages, context, selected, seed=5 + index)
            for index in range(chain)]
    scale = 256.0 ** -0.5
    sparse = base.mla_sparse

    def after(*flat):
        outs = []
        for index in range(chain):
            q, bank, table, pos, written, indices = flat[6 * index:6 * index + 6]
            outs.append(MD.mla_decode_attention(
                q, bank, table, pos, written, scale, case.PAGE,
                indices if selected else None))
        return torch.stack(outs, 0)

    def before(*flat):
        outs = []
        for index in range(chain):
            q, bank, table, pos, written, indices = flat[6 * index:6 * index + 6]
            rows = []
            for b in range(batch):
                rows.append(sparse.mla_sparse_attention(
                    q[b:b + 1], bank, indices[b:b + 1], scale,
                    block_table_row=table[b].reshape(-1, 1), written=written[b:b + 1],
                    write_offset=pos[b].reshape(1, 1), page_size=case.PAGE))
            outs.append(torch.cat(rows, 0))
        return torch.stack(outs, 0)

    def after_one_program(*flat):
        """The seam's kernel on one program, to show what the second core buys."""
        call = wrap_nki(MD.mla_decode_selected_kernel if selected
                        else MD.mla_decode_dense_kernel)
        outs = []
        for index in range(chain):
            q, bank, table, pos, written, indices = flat[6 * index:6 * index + 6]
            head = (q, bank, table, pos, written)
            outs.append(call(*head, indices, scale, case.PAGE, MD.SOURCE_DIGEST)
                        if selected else call(*head, scale, case.PAGE, MD.SOURCE_DIGEST))
        return torch.stack(outs, 0)

    inputs = tuple(t.to(DEVICE) for one in sets for t in one)
    variants = [("before", before), ("after", after)]
    if MD._programs(batch) == 2:
        variants.append(("after_one_program", after_one_program))
    timing, outputs = {}, {}
    for tree, fn in variants:
        graph = compiled(fn)
        outputs[tree], seconds = first_call(graph, inputs)
        timing[tree] = measure(graph, inputs, chain, args.warmup, args.iterations)
        timing[tree]["first_call_s"] = seconds
    oracle = torch.stack([MD.mla_decode_attention_torch_oracle(
        one[0], one[1], one[2], one[3], one[4], scale, case.PAGE,
        one[5] if selected else None) for one in sets], 0)
    checks = {tree: rel_l2(out, oracle) for tree, out in outputs.items()}
    if max(checks.values()) > KERNEL_REL_L2:
        raise AssertionError(f"attention {mode} B={batch}: vs CPU oracle {checks}")
    return {
        "table": "attention",
        "case": mode,
        "ctx": context,
        "window_rows": pages * case.PAGE,
        "indices": 2048,
        "batch": batch,
        "calls_chained": chain,
        "programs_after": MD._programs(batch),
        "unit": "one decode attention for all `batch` requests",
        "vs_cpu_oracle_relative_l2": checks,
        **compare(timing["before"], timing["after"]),
        **_one_program(timing),
    }


# --- projection ----------------------------------------------------------------------

def projection_case(site, idim, odim, fp8, rows, base, args) -> dict:
    torch._dynamo.reset()
    chain = args.kernel_chain
    gen = torch.Generator().manual_seed(idim + odim + rows)
    x = torch.randn(rows, idim, generator=gen).to(torch.bfloat16)
    lowp, dense = [], []
    for _ in range(chain):
        if fp8:
            raw = (torch.randn(odim, idim, generator=gen) * 48).clamp(-224, 224)
            w_out_in = raw.to(torch.float8_e4m3fn)
            s_out_in = (torch.rand(odim // 128, idim // 128, generator=gen) * 0.5 + 0.75) \
                * idim ** -0.5 / 48
            fp32 = (w_out_in.float() * s_out_in.repeat_interleave(128, 0)
                    .repeat_interleave(128, 1)).t().contiguous()
        else:
            w_out_in = (torch.randn(odim, idim, generator=gen) * idim ** -0.5).to(torch.bfloat16)
            s_out_in = None
            fp32 = w_out_in.float().t().contiguous()
        lowp.append(MP.prepare_lowp_projection(w_out_in, s_out_in))
        dense.append(fp32)
    old = base.mla_projections.mla_projection
    # A prefill-sized output is megabytes per call; copying it all to CPU would time
    # the copy. Each graph returns its first rows only (every kernel still writes its
    # whole output), and the check reads those rows.
    head = rows if rows <= 8 else 1

    def before(a, *weights):
        a32 = a.to(torch.float32)
        return torch.stack([old(a32, w)[:head] for w in weights], 0)

    def after(a, *operands):
        weights, scales = operands[:chain], operands[chain:]
        return torch.stack([MP.mla_projection_lowp(a, weights[index],
                                                   scales[index] if fp8 else None)[:head]
                            for index in range(chain)], 0)

    chunk_one = MP._lowp_col_chunk(idim, odim, lowp[0][0].element_size(), fp8)

    def after_one_program(a, *operands):
        """The seam's kernel on one program, to show what the second core buys."""
        weights, scales = operands[:chain], operands[chain:]
        call = wrap_nki(MP.mla_projection_lowp_kernel)
        return torch.stack([call(a, weights[index], scales[index] if fp8 else None,
                                 chunk_one)[:head] for index in range(chain)], 0)

    xd = x.to(DEVICE)
    before_inputs = (xd, *(w.to(DEVICE) for w in dense))
    after_inputs = (xd, *(w.to(DEVICE) for w, _ in lowp),
                    *(s.to(DEVICE) for _, s in lowp if s is not None))
    one_inputs = (xd, *(w.to(DEVICE) for w, _ in lowp),
                  *(MP.lowp_scale_layout(MP.lowp_scale_grid(s), 1).to(DEVICE)
                    for _, s in lowp if s is not None))
    variants = [("before", before, before_inputs), ("after", after, after_inputs)]
    if MP.lowp_programs(odim, fp8) == 2:
        variants.append(("after_one_program", after_one_program, one_inputs))
    timing, outputs = {}, {}
    for tree, fn, inputs in variants:
        graph = compiled(fn)
        outputs[tree], seconds = first_call(graph, inputs)
        timing[tree] = measure(graph, inputs, chain, args.warmup, args.iterations)
        timing[tree]["first_call_s"] = seconds
    exact = torch.stack([x[:head].double() @ w.double() for w in dense], 0)
    checks = {tree: rel_l2(out, exact) for tree, out in outputs.items()}
    if max(checks.values()) > KERNEL_REL_L2:
        raise AssertionError(f"projection {site} M={rows}: vs fp64 {checks}")
    return {
        "table": "projection",
        "case": site,
        "in": idim,
        "out": odim,
        "weight": "fp8-e4m3 + 128x128 fp32 scale" if fp8 else "bf16",
        "rows": rows,
        "calls_chained": chain,
        "programs_after": MP.lowp_programs(odim, fp8),
        "unit": "one projection of `rows` rows",
        "rows_copied_to_cpu": head,
        "vs_fp64_relative_l2": checks,
        **compare(timing["before"], timing["after"]),
        **_one_program(timing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables", nargs="+", default=["layer"],
                        choices=["layer", "attention", "projection"],
                        help="default: the layer table alone, so one leased run stays "
                             "under ten minutes; the kernel tables run as a second job")
    parser.add_argument("--layer-cases", nargs="+",
                        default=[label for label, *_ in LAYER_CASES],
                        choices=[label for label, *_ in LAYER_CASES])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prefill-rows", type=int, nargs="*", default=[1024],
                        help="extra projection row counts: the prefill chunk, which "
                             "takes the same low-precision route")
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--kernel-chain", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    if args.layers < 1 or args.kernel_chain < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use positive chain lengths and iteration counts")
    baseline_dir = args.baseline_module.resolve()
    output = args.output.resolve()
    base = load_baseline(baseline_dir)
    # The compiler leaves per-graph debug files in its working directory; keep them out
    # of the worktree.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
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
        "args": {"layers": args.layers, "kernel_chain": args.kernel_chain,
                 "warmup": args.warmup, "iterations": args.iterations,
                 "batch": args.batch, "prefill_rows": args.prefill_rows,
                 "tables": args.tables, "layer_cases": args.layer_cases},
        "synchronization": "graph output copied to CPU on every timed call",
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        brief = {k: row[k] for k in ("table", "case", "batch", "rows") if k in row}
        print(json.dumps({**brief, "before_median_us": row["before"]["median_us"],
                          "after_median_us": row["after"]["median_us"],
                          "after_over_before": row["after_over_before_median"]}),
              flush=True)

    if "layer" in args.tables:
        for label, context, before_pages, after_pages, single in LAYER_CASES:
            if label not in args.layer_cases:
                continue
            for batch in ([1] if single else args.batch):
                record(layer_case(label, context, before_pages, after_pages, batch, base,
                                  args))
    if "attention" in args.tables:
        for mode, context, pages in (("dense", 1024, 16), ("selected", 4096, 32)):
            for batch in args.batch:
                record(attention_case(mode, context, pages, batch, base, args))
    if "projection" in args.tables:
        for rows in [*args.batch, *args.prefill_rows]:
            for site, idim, odim, fp8 in SITES:
                record(projection_case(site, idim, odim, fp8, rows, base, args))


if __name__ == "__main__":
    main()
