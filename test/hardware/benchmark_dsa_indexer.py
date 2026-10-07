# SPDX-License-Identifier: Apache-2.0
"""Time the DSA decode indexer chain for ``B`` requests: this tree against 0a08ff4.

Two subcommands.

``run`` (device; through the lease, which pins the cores and the LNC -- this script
selects none). For each ``B:ctx`` case (default ``1:8192 4:8192 64:8192 64:2051``) it
builds ``--layers`` (default 11, the model's DSA layer count) layers of operands at the
per-rank TP=64 indexer shape (32 heads of 128, ``index_topk`` 2048, ``index_kpool`` 4)
and times, for ``before`` (0a08ff4, ``test/hardware/baselines/dsa_0a08ff4``) and
``after`` (this tree):

* ``chain`` -- the decode indexer chain as ``Glm5NextMLAAttention._forward_requests``
  runs it from the projections on: the query rotation (``dsa_hadamard128``) and
  ``Glm5NextDSAIndexer.forward_requests`` in the bank form (ring step, scores, the
  selection stage, the two bank writes). At a context inside the bypass
  (``ctx <= 2051``) the call site sends no query side and wants no indices, so the
  chain is the ring step and the writes, as in the model.
* one graph per kernel of the chain, ``--layers`` calls each: ``hadamard``, ``ring``,
  ``scores``, ``select`` (``before``: top-k, causal sentinel, sentinel order and expand;
  ``after``: whatever this tree's ``forward_requests`` runs there) and, ``before`` only,
  the four kernels of the select stage one at a time (``topk``, ``causal_sentinel``,
  ``sentinel_order``, ``index_expand``).

Every graph returns the last 8 elements of each layer's outputs, so every kernel stays
live and the copy to CPU that synchronises each timed call is a few hundred bytes. The
reported unit is the graph's wall time divided by the layer count; ``empty`` times a
graph that only slices its inputs the same way, the fixed per-call cost the other rows
carry. Before timing, ``before`` and ``after`` chains run once on private copies of the
same operands: the selected index sets must agree per row (order-insensitive, ``-1``
dropped) and the written rings and pools must agree bit for bit.

With ``--profile-dir`` both chains then run ``--profile-iterations`` times under the
runtime profiler (device and system profile), after the timing.

``cores`` (CPU only; any python with ``duckdb``, e.g. ``/tmp/pqvenv/bin/python``). Reads
an explorer profile ingested with ``neuron-explorer view -d <profile dir> --ingest-only
--data-path <D> --display-name <name>`` and reports, per NKI source file and execution,
the instruction count, the busy time (summed instruction time), the active time (the union
of the instruction intervals) and the first-to-last span on each physical core. With
``--functions`` the label is ``file.py:function``, the top-level function enclosing the
instruction's source line (read from the file at the profiled path), which tells apart two
kernels of one file (the ring step and the scores) and a kernel from its helpers.
"""

from __future__ import annotations

import argparse
import glob
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

DEVICE = "neuron:0"
DEFAULT_CASES = ("1:8192", "4:8192", "64:8192", "64:2051")
TAIL = 8


# ------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------


def _torch():
    import torch
    return torch


def load_baseline(directory: Path):
    """The baseline package's ``load()``: 0a08ff4's modules, read by ``git show``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dsa_indexer_baseline_loader", directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module.load()


def compiled(fn):
    torch = _torch()
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def lengths_for(ctx: int, batch: int) -> list[int]:
    """Request ``b`` is at length ``ctx - b % 4``: closed and open pools mixed."""
    return [ctx - b % 4 for b in range(batch)]


def layer_operands(cfg, ctx: int, batch: int, seed: int) -> dict:
    """One layer's decode-step operands on CPU, the bank form ``forward_requests`` takes."""
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    lengths = lengths_for(ctx, batch)
    slots_total = batch + 3
    rows = -(-(ctx // pool + 1) // 128) * 128
    return {
        "query_rows": (torch.randn(batch * heads, dim, generator=gen)).to(torch.bfloat16),
        "key": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(batch, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        "gate": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots_total, rows, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots_total, 2, pool, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "slots": torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
        "position": torch.tensor([n - 1 for n in lengths], dtype=torch.int64),
    }


def build_indexer(model_module, cfg, ape):
    torch = _torch()
    indexer = model_module.Glm5NextDSAIndexer(cfg)
    indexer.index_kpool_compress_ape = torch.nn.Parameter(ape.clone(), requires_grad=False)
    return indexer.to(DEVICE)


class Tree:
    """One tree's chain and kernels, as the graphs call them."""

    def __init__(self, name, indexer, hadamard, decode_batch, select, select_parts):
        self.name = name
        self.indexer = indexer
        self.hadamard = hadamard
        self.db = decode_batch
        self.select = select
        self.select_parts = select_parts

    def chain(self, ops, ctx, dense):
        batch = int(ops["key"].shape[0])
        ix = self.indexer
        query = None
        if not dense:
            query = self.hadamard(ops["query_rows"]).reshape(
                batch, ix.index_n_heads, ix.index_head_dim)
        return ix.forward_requests(
            ops["key"].new_zeros((batch, 1)), None, ops["pool_bank"], ops["tail_bank"],
            ops["slots"], ops["seq_lens"], ops["position"], max_seq_len=int(ctx),
            indices_wanted=not dense,
            projected=(query, ops["key"], None if dense else ops["weights"], ops["gate"]))


def live_tree(cfg, ape):
    """This tree: its kernels as ``forward_requests`` and ``project_stage`` call them."""
    from vllm_neuron.functional.dsa import decode_batch, kpool_hadamard
    from vllm_neuron.model.glm5_next import model_fp8
    indexer = build_indexer(model_fp8, cfg, ape)
    try:
        from vllm_neuron.functional.dsa import decode_select
    except ImportError:
        decode_select = None
    if decode_select is not None:
        def select(bounded, seq_lens):
            return decode_select.dsa_decode_select(
                bounded, seq_lens, select_k=indexer.select_k(),
                pool_size=indexer.index_kpool)
    else:
        def select(bounded, seq_lens):
            return indexer.expand_indices(indexer._select_bounded(bounded), seq_lens)
    return Tree("after", indexer, kpool_hadamard.dsa_hadamard128, decode_batch, select, None)


def base_tree(base, cfg, ape):
    """0a08ff4: the snapshot's kernels, composed as its ``forward_requests`` composes them."""
    indexer = build_indexer(base.model_fp8, cfg, ape)
    k = indexer.select_k()

    def select(bounded, seq_lens):
        return indexer.expand_indices(indexer._select_bounded(bounded), seq_lens)

    def topk(bounded):
        values, indices = base.topk_select.dsa_topk_select(bounded, k)
        return values, indices.to(_torch().int32)

    def sentinel(values, pool_ids, width):
        return base.causal_bound.dsa_causal_sentinel(values, pool_ids, width)

    parts = {"topk": topk, "causal_sentinel": sentinel,
             "sentinel_order": base.sentinel_order.dsa_sentinel_order,
             "index_expand": lambda ids, lens: base.index_expand.dsa_index_expand(
                 ids, lens, indexer.index_kpool)}
    return Tree("before", indexer, base.kpool_hadamard.dsa_hadamard128, base.decode_batch,
                select, parts)


def tails(*tensors):
    """The last ``TAIL`` elements of every output, fp32, stacked: keeps every kernel live."""
    torch = _torch()
    return torch.stack([t.reshape(-1)[-TAIL:].to(torch.float32) for t in tensors])


def to_device(ops: dict) -> dict:
    return {k: v.to(DEVICE, copy=True) for k, v in ops.items()}


def measure(fn, inputs, per: int, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0 / per)
    samples.sort()
    return {"iterations": iterations, "units_per_sample": per,
            "median_us": statistics.median(samples),
            "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
            "min_us": samples[0], "max_us": samples[-1]}


def selected_sets(indices) -> list[set]:
    return [set(int(v) for v in row.tolist() if v >= 0) for row in indices]


def run_case(batch: int, ctx: int, base, args) -> dict:
    torch = _torch()
    from test.vllm_neuron.functional.dsa import dsa_decode_case as case
    from vllm_neuron.functional.dsa.decode_bypass import selection_is_a_no_op

    torch._dynamo.reset()
    cfg = case.decode_config()
    pool, k = int(cfg.index_kpool), int(cfg.index_topk) // int(cfg.index_kpool)
    dense = selection_is_a_no_op(int(ctx), int(cfg.index_topk), pool)
    candidates = int(ctx) // pool
    ape = (torch.randn(pool, int(cfg.index_head_dim),
                       generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    trees = {"before": base_tree(base, cfg, ape), "after": live_tree(cfg, ape)}
    layers = [layer_operands(cfg, ctx, batch, seed=1000 * batch + ctx + 7 * i)
              for i in range(args.layers)]
    names = sorted(layers[0])

    # ---- agreement: both chains once, on private copies of layer 0 -------------------
    agreement = {}
    outs = {}
    for name, tree in trees.items():
        torch._dynamo.reset()
        ops = to_device(layers[0])
        fn = compiled(lambda *flat, _t=tree: _single_chain(_t, dict(zip(names, flat)),
                                                         ctx, dense))
        got = fn(*[ops[n] for n in names])
        outs[name] = {"indices": None if dense else got[0].to("cpu"),
                      "pool_bank": got[1].to("cpu"), "tail_bank": got[2].to("cpu")}
    agreement["banks_bit_identical"] = bool(
        torch.equal(outs["before"]["pool_bank"], outs["after"]["pool_bank"])
        and torch.equal(outs["before"]["tail_bank"], outs["after"]["tail_bank"]))
    if not dense:
        sb = selected_sets(outs["before"]["indices"])
        sa = selected_sets(outs["after"]["indices"])
        agreement["rows"] = len(sb)
        agreement["rows_with_equal_sets"] = sum(int(a == b) for a, b in zip(sa, sb))
        agreement["selected_per_row"] = sorted({len(s) for s in sb})
    if not agreement["banks_bit_identical"] or agreement.get(
            "rows_with_equal_sets", 0) != agreement.get("rows", 0):
        raise AssertionError(f"B={batch} ctx={ctx}: before and after disagree: {agreement}")

    # ---- timed graphs ----------------------------------------------------------------
    timing = {}
    graphs = _graphs(trees, layers, names, ctx, dense, candidates, pool, k, ape)
    keep = set(args.graphs) if args.graphs else None
    for gname, (fn, inputs) in graphs.items():
        if keep is not None and gname not in keep and not gname.startswith("chain"):
            continue
        torch._dynamo.reset()
        cfn = compiled(fn)
        started = time.perf_counter()
        cfn(*inputs).to("cpu")
        first = time.perf_counter() - started
        per = 1 if gname == "empty" else args.layers
        timing[gname] = measure(cfn, inputs, per, args.warmup, args.iterations)
        timing[gname]["first_call_s"] = first
        print(json.dumps({"B": batch, "ctx": ctx, "graph": gname,
                          "median_us": round(timing[gname]["median_us"], 1)}), flush=True)
        if args.profile_dir is not None and gname.startswith("chain_"):
            _profile(cfn, inputs, args, f"b{batch}_c{ctx}_{gname}")

    row = {"batch": batch, "ctx": ctx, "max_seq_len": ctx, "candidates": candidates,
           "select_k": k, "dense": dense, "lengths": lengths_for(ctx, batch)[:8],
           "layers": args.layers, "agreement": agreement, "timing": timing}
    row["speedup"] = {}
    for stage in ("chain", "hadamard", "ring", "scores", "select"):
        b, a = timing.get(f"{stage}_before"), timing.get(f"{stage}_after")
        if b and a:
            row["speedup"][stage] = {"before_median_us": b["median_us"],
                                     "after_median_us": a["median_us"],
                                     "before_p90_us": b["p90_us"], "after_p90_us": a["p90_us"],
                                     "median_speedup": b["median_us"] / a["median_us"],
                                     "after_over_before": a["median_us"] / b["median_us"]}
    return row


def _single_chain(tree, ops, ctx, dense):
    """One layer's chain, returning the indices and both banks (the agreement check)."""
    out = tree.chain(ops, ctx, dense)
    pool_bank, tail_bank = ops["pool_bank"], ops["tail_bank"]
    if out is None:
        return (pool_bank, pool_bank * 1, tail_bank * 1)
    return (out, pool_bank * 1, tail_bank * 1)


def _graphs(trees, layers, names, ctx, dense, candidates, pool, k, ape_cpu):
    """``name -> (fn, device inputs)`` for every timed graph of one case."""
    torch = _torch()
    from vllm_neuron.functional.dsa.decode_batch import dsa_decode_scores_torch_oracle
    from vllm_neuron.functional.dsa import decode_batch as live_db

    out = {}
    n_layers = len(layers)

    def flat_inputs(keys):
        dev = [to_device({kk: ops[kk] for kk in keys}) for ops in layers]
        return tuple(d[kk] for d in dev for kk in keys)

    def unflat(flat, keys):
        return [dict(zip(keys, flat[i * len(keys):(i + 1) * len(keys)]))
                for i in range(n_layers)]

    # empty: the fixed per-call cost, same output shape as every other graph.
    def empty(*flat):
        return tails(*flat)
    out["empty"] = (empty, flat_inputs(["key"]))

    for tname, tree in trees.items():
        def chain(*flat, _t=tree):
            res = []
            for ops in unflat(flat, names):
                got = _t.chain(ops, ctx, dense)
                res.append(ops["tail_bank"] if got is None else got)
            return tails(*res)
        out[f"chain_{tname}"] = (chain, flat_inputs(names))

        ring_keys = ["tail_bank", "slots", "key", "gate", "position"]
        def ring(*flat, _t=tree):
            res = []
            for ops in unflat(flat, ring_keys):
                pooled, rings = _t.db.dsa_decode_ring_step(
                    ops["tail_bank"], ops["slots"], ops["key"], ops["gate"],
                    _t.indexer.index_kpool_compress_ape.to(torch.float32), ops["position"])
                res.extend([pooled, rings])
            return tails(*res)
        out[f"ring_{tname}"] = (ring, flat_inputs(ring_keys))

        if dense:
            continue

        def had(*flat, _t=tree):
            return tails(*[_t.hadamard(x) for x in flat])
        out[f"hadamard_{tname}"] = (had, flat_inputs(["query_rows"]))

    if dense:
        return out

    # Operands the later stages read, computed once on CPU: the rotated query (the torch
    # rotation equals the kernel's to bf16 rounding; the stage timings do not depend on
    # it), this step's pooled row, and the bounded scores.
    from vllm_neuron.functional.dsa.kpool_hadamard import _dsa_hadamard128_torch
    staged = []
    for ops in layers:
        batch = int(ops["key"].shape[0])
        query = _dsa_hadamard128_torch(ops["query_rows"]).reshape(batch, -1, 128)
        pooled, _ = live_db.dsa_decode_ring_step_torch_oracle(
            ops["tail_bank"], ops["slots"], ops["key"], ops["gate"],
            ape_cpu.float(), ops["position"])
        bounded = dsa_decode_scores_torch_oracle(
            query, ops["weights"], ops["pool_bank"], ops["slots"], ops["seq_lens"],
            ops["position"], pooled, candidates=candidates, pool_size=pool)
        staged.append({"query": query, "pooled": pooled.to(torch.bfloat16),
                       "bounded": bounded.contiguous(), **ops})
    for ops in staged:
        ops["seq_lens32"] = ops["seq_lens"].to(torch.int32)

    def staged_inputs(keys):
        dev = [to_device({kk: ops[kk] for kk in keys}) for ops in staged]
        return tuple(d[kk] for d in dev for kk in keys)

    score_keys = ["query", "weights", "pool_bank", "slots", "seq_lens", "position", "pooled"]
    select_keys = ["bounded", "seq_lens32"]
    for tname, tree in trees.items():
        def scores(*flat, _t=tree):
            res = []
            for ops in unflat(flat, score_keys):
                res.append(_t.db.dsa_decode_scores(
                    ops["query"], ops["weights"], ops["pool_bank"], ops["slots"],
                    ops["seq_lens"], ops["position"], ops["pooled"],
                    candidates=candidates, pool_size=pool))
            return tails(*res)
        out[f"scores_{tname}"] = (scores, staged_inputs(score_keys))

        def select(*flat, _t=tree):
            return tails(*[_t.select(o["bounded"], o["seq_lens32"])
                           for o in unflat(flat, select_keys)])
        out[f"select_{tname}"] = (select, staged_inputs(select_keys))

    parts = trees["before"].select_parts
    width = candidates

    def topk(*flat):
        res = []
        for o in unflat(flat, ["bounded"]):
            res.extend(parts["topk"](o["bounded"]))
        return tails(*res)
    out["topk_before"] = (topk, staged_inputs(["bounded"]))

    # The sentinel, order and expand kernels on the selector's real output (CPU top-k).
    for ops in staged:
        values, idx = torch.topk(ops["bounded"], k, dim=-1)
        ops["values"], ops["pool_ids"] = values.contiguous(), idx.to(torch.int32).contiguous()

    def sentinel(*flat):
        return tails(*[parts["causal_sentinel"](o["values"], o["pool_ids"], width)
                       for o in unflat(flat, ["values", "pool_ids"])])
    out["causal_sentinel_before"] = (sentinel, staged_inputs(["values", "pool_ids"]))

    def order(*flat):
        return tails(*[parts["sentinel_order"](o["pool_ids"])
                       for o in unflat(flat, ["pool_ids"])])
    out["sentinel_order_before"] = (order, staged_inputs(["pool_ids"]))

    def expand(*flat):
        return tails(*[parts["index_expand"](o["pool_ids"], o["seq_lens32"])
                       for o in unflat(flat, ["pool_ids", "seq_lens32"])])
    out["index_expand_before"] = (expand, staged_inputs(["pool_ids", "seq_lens32"]))
    return out


def _profile(cfn, inputs, args, tag):
    torch = _torch()
    import libtorch_neuronx_lite.envs as libtorch_envs
    where = args.profile_dir / tag
    where.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(where), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            cfn(*inputs).to("cpu")
    finally:
        runtime.stop_profiling()


def cmd_run(args) -> None:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    _loader, base = load_baseline(args.baseline_module.resolve())
    output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {}
    if args.merge and output.exists():
        report = json.loads(output.read_text())
    report.setdefault("cases", [])
    report.update({
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
        "tree": str(ROOT),
        "baseline": {"commit": base.commit, "directory": base.directory},
        "args": {"layers": args.layers, "warmup": args.warmup,
                 "iterations": args.iterations, "cases": args.cases,
                 "profile_dir": None if args.profile_dir is None else str(args.profile_dir),
                 "profile_iterations": args.profile_iterations},
        "unit": "graph wall time / layers, microseconds; 'empty' is per call",
        "synchronization": "graph output (last 8 elements of each output) copied to CPU "
                           "on every timed call",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    for spec in args.cases:
        batch, ctx = (int(v) for v in spec.split(":"))
        row = run_case(batch, ctx, base, args)
        report["cases"] = [r for r in report["cases"]
                           if (r["batch"], r["ctx"]) != (batch, ctx)] + [row]
        report["cases"].sort(key=lambda r: (r["ctx"], r["batch"]))
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"B": batch, "ctx": ctx, "speedup": row["speedup"],
                          "agreement": row["agreement"]}), flush=True)


# ------------------------------------------------------------------------------------
# cores
# ------------------------------------------------------------------------------------


def _union_us(spans) -> float:
    """Time inside at least one of these ``(start, end)`` intervals (ns), microseconds."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1000.0


def _function_table(path: str) -> list[tuple[int, int, str]]:
    """``(first line, last line, name)`` of every top-level function in ``path``."""
    import ast
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return []
    return [(node.lineno, node.end_lineno or node.lineno, node.name) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _labeler(functions: bool):
    """``nki_source_location -> label``: the file name, or ``file.py:function``."""
    tables: dict[str, list] = {}
    cache: dict[str, str] = {}

    def label(location: str) -> str:
        if location in cache:
            return cache[location]
        path, _, rest = location.partition(":")
        name = path.rsplit("/", 1)[-1]
        if functions:
            line = int(rest.split(":")[0]) if rest.split(":")[0].isdigit() else -1
            table = tables.setdefault(path, _function_table(path))
            owner = next((f for a, b, f in table if a <= line <= b), "?")
            name = f"{name}:{owner}"
        cache[location] = name
        return name
    return label


def cmd_cores(args) -> None:
    import duckdb
    found = sorted(glob.glob(f"{args.data}/profiles/global/{args.name}_*_session_*@latest"))
    if not found:
        raise ValueError(f"no ingested profile {args.name} under {args.data}")
    path = found[0]
    con = duckdb.connect()
    q = lambda t: f"'{path}/{t}.parquet'"  # noqa: E731
    execs = con.sql(f"SELECT execution_index, execution_start_ts, execution_end_ts "
                    f"FROM {q('ExecutionInfo')} ORDER BY 2").fetchall()
    label = _labeler(args.functions)
    per_exec = []
    for idx, s0, e0 in execs:
        rows = con.sql(f"""
            SELECT nki_source_location, pcore_idx, start_ts, end_ts FROM {q('Instruction')}
            WHERE start_ts >= {s0} AND end_ts <= {e0}
              AND nki_source_location IS NOT NULL AND nki_source_location <> ''
            """).fetchall()
        spans: dict[tuple[str, int], list] = {}
        for location, core, start, end in rows:
            spans.setdefault((label(location), int(core)), []).append((start, end))
        kernels = {}
        for (src, core), items in sorted(spans.items()):
            kernels.setdefault(src, {})[f"pcore{core}"] = {
                "instructions": len(items),
                "busy_us": sum(e - s for s, e in items) / 1000.0,
                "span_us": (max(e for _, e in items) - min(s for s, _ in items)) / 1000.0,
                "active_us": _union_us(items)}
        per_exec.append({"execution": int(idx), "wall_us": (e0 - s0) / 1000.0,
                         "kernels": kernels})
    summary = {}
    for ex in per_exec:
        for src, cores in ex["kernels"].items():
            entry = summary.setdefault(src, {"executions": 0, "pcores_active": set(),
                                             "active_us_per_execution": {}})
            entry["executions"] += 1
            entry["pcores_active"].update(c for c, v in cores.items() if v["instructions"])
            for c, v in cores.items():
                entry["active_us_per_execution"].setdefault(c, []).append(v["active_us"])
    for entry in summary.values():
        entry["pcores_active"] = sorted(entry["pcores_active"])
        entry["active_us_per_execution"] = {
            c: round(statistics.median(v), 1)
            for c, v in sorted(entry["active_us_per_execution"].items())}
    print(json.dumps({"profile": path, "functions": bool(args.functions),
                      "per_kernel": summary, "executions": per_exec}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--baseline-module", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--profile-dir", type=Path)
    run.add_argument("--profile-iterations", type=int, default=3)
    run.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    run.add_argument("--graphs", nargs="*", help="time only these graphs (chains always)")
    run.add_argument("--layers", type=int, default=11)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--iterations", type=int, default=50)
    run.add_argument("--merge", action="store_true",
                     help="keep the output file's rows for cases this run skips")
    cores = sub.add_parser("cores")
    cores.add_argument("--data", required=True, help="explorer --data-path")
    cores.add_argument("--name", required=True, help="explorer --display-name")
    cores.add_argument("--functions", action="store_true",
                       help="label by file.py:function rather than by file")
    args = parser.parse_args()
    if args.cmd == "run":
        if args.layers < 1 or args.iterations < 1 or args.warmup < 0:
            raise ValueError("Use positive layer and iteration counts")
        cmd_run(args)
    else:
        cmd_cores(args)


if __name__ == "__main__":
    main()
