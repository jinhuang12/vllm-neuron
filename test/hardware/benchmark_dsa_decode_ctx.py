# SPDX-License-Identifier: Apache-2.0
"""Time the DSA decode indexer chain at long contexts: this tree against e3f38f8.

Three subcommands, and ``cores`` to read a profile.

``run`` (device; through the lease, which pins the cores and the LNC -- this script
selects none). For each ``B:ctx`` case (default ``B`` in {1, 64} x ``ctx`` in {2048, 4096,
8192, 16384, 32768}) it builds ``--layers`` (default 11, the model's DSA layer count)
layers of operands at the call site's real shapes -- the per-rank TP=64 indexer (32 heads
of 128, ``index_topk`` 2048, ``index_kpool`` 4), a pooled-key bank of ``ctx // 4 + 1``
rows per slot (``_glm5next_side_caches`` at ``max_model_len = ctx``) and
``max_seq_len = ctx`` -- and times, for ``before`` (e3f38f8,
``test/hardware/baselines/dsa_capped_select``) and ``after`` (this tree):

* ``chain`` -- the decode indexer chain as ``Glm5NextMLAAttention._forward_requests``
  runs it from the projections on: the query rotation and
  ``Glm5NextDSAIndexer.forward_requests`` in the bank form (ring step, scores,
  selection, the two bank writes). Inside the bypass (``ctx <= 2051``) the call site
  sends no query side and wants no indices, as in the model.
* ``ring``, ``scores``, ``select`` -- one graph per kernel of the chain, ``--layers``
  calls each, on operands computed once on CPU.

Every graph returns the last 8 elements of each layer's outputs; the reported unit is
the graph's wall time divided by the layer count, and ``empty`` is the fixed per-call
cost. A tree that refuses a shape (e3f38f8 past its 16384-candidate ceiling) is recorded
with its error and not timed.

A case ``B:ctx:max_model_len`` (``max_model_len > ctx``) is a decode bucket ``ctx`` below
``max_model_len``: the bank has ``max_model_len // 4 + 1`` rows, the rows' lengths are at
most ``ctx``, and ``chain_before`` / ``chain_after`` take ``max_seq_len = max_model_len``,
e3f38f8's call site, so their difference is the kernels' alone. ``chain_after_bound`` is
this tree's chain at this tree's call site, ``max_seq_len = ctx`` (the window bound): its
difference from ``chain_after`` is the call-site change's alone.

Checks, per case, before any timing:

* **kernels in the graph** -- each tree's chain graph is captured as dynamo hands it to
  the Neuron backend, and every ``nki_kernel_wrapper`` node is listed by kernel name and
  launch grid (``kernels_in_graph``). ``after`` must hold the ring step, the scores and
  the selection kernels and no torch fallback (the module's dispatch counters).
* **sets** -- one layer through ``after``'s ring step and scores, then its chain, on the
  device: the chain's indices, sorted per row, must equal (``torch.equal``) the torch
  selection oracle's on the very scores the device computed; and ``before``'s chain,
  where it serves the shape, must select the same sets and write the same banks.

With ``--profile-dir`` both chains then run ``--profile-iterations`` times under the
runtime profiler (device and system profile). ``cores`` reads the ingested profile and
reports instructions and active time per kernel per physical core
(``benchmark_dsa_indexer.py cores``, the same reader).

``select`` (device; through the lease): the selection kernel alone at each ``B:C`` case
(candidates, not tokens; default :data:`SELECT_CASES`), ``--layers`` calls a graph on
random scores (odd rows shorter, ``BOUND_FILL`` past them), compiled once and run
``--runs`` times; every row of every layer of every run must equal the torch oracle's
indices bit for bit, or the command fails.

``compile`` (CPU only, no device; it still runs only under the lease or where no
``/dev/neuron*`` node exists, :func:`require_no_unleased_device`): for each
``tree:kernel:B:C`` entry, one child process (``compile-one``) on an empty cache, timed,
with the child's peak RSS. By default the NKI compile (front end to BIR, what
``libtorch_neuronx_lite.nki.nki_compile.compile_nki`` runs when a graph first traces a
kernel call) of one kernel of the chain -- ring step, scores, selection -- and the size of
its compiled kernel config (a proxy for its instruction stream). With ``--full`` the whole
compile (dynamo -> HLO -> neuronx-cc -> NEFF) of one kernel call or of the whole chain
(kernel ``chain``, ``ctx = 4 C``, built on the meta device), and the NEFF size. A kernel
unrolled over the candidate axis shows compile time growing with ``C``.
``compile-one --unroll-blocks N`` overrides the score kernel's ``unroll_blocks``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

DEFAULT_CASES = tuple(f"{b}:{c}" for c in (2048, 4096, 8192, 16384, 32768) for b in (1, 64))


def _torch():
    import torch
    return torch


def _bench():
    from test.hardware import benchmark_dsa_indexer as bench
    return bench


def layer_operands(cfg, ctx: int, batch: int, seed: int, max_model_len: int) -> dict:
    """One layer's decode operands on CPU, the bank form, at the runner's bank shape."""
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    lengths = _bench().lengths_for(ctx, batch)
    slots_total = batch + 3
    rows = int(max_model_len) // pool + 1
    return {
        "query_rows": torch.randn(batch * heads, dim, generator=gen).to(torch.bfloat16),
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


class Tree:
    def __init__(self, name, indexer, hadamard, decode_batch, decode_select):
        self.name = name
        self.indexer = indexer
        self.hadamard = hadamard
        self.db = decode_batch
        self.ds = decode_select

    def query(self, ops):
        batch = int(ops["key"].shape[0])
        return self.hadamard(ops["query_rows"]).reshape(
            batch, self.indexer.index_n_heads, self.indexer.index_head_dim)

    #: ``bank``: the whole banks and a slot vector (hostpath's call site). ``views``: one
    #: view per request and no slots (e3f38f8's call site, ``_forward_requests``), which
    #: ``forward_requests`` stacks into a bank of its own before the kernels.
    form = "bank"
    #: The slots of the case's requests, python ints (the views form slices by them).
    view_slots: tuple = ()

    def chain(self, ops, max_seq_len, dense):
        batch = int(ops["key"].shape[0])
        query = None if dense else self.query(ops)
        pool_bank, tail_bank, slots = ops["pool_bank"], ops["tail_bank"], ops["slots"]
        if self.form == "views":
            pool_bank = tuple(pool_bank[s] for s in self.view_slots)
            tail_bank = tuple(tail_bank[s] for s in self.view_slots)
            slots = None
        return self.indexer.forward_requests(
            ops["key"].new_zeros((batch, 1)), None, pool_bank, tail_bank,
            slots, ops["seq_lens"], ops["position"], max_seq_len=int(max_seq_len),
            indices_wanted=not dense,
            projected=(query, ops["key"], None if dense else ops["weights"], ops["gate"]))

    def select(self, bounded, seq_lens):
        return self.ds.dsa_decode_select(bounded, seq_lens,
                                         select_k=self.indexer.select_k(),
                                         pool_size=self.indexer.index_kpool)


def _build_indexer(model_module, cfg, ape, device):
    """``benchmark_dsa_indexer.build_indexer`` on ``device``. The compile path passes
    ``meta``: moving a module to the Neuron device initialises the runtime (NRT), which
    a compile must never do."""
    torch = _torch()
    indexer = model_module.Glm5NextDSAIndexer(cfg)
    indexer.index_kpool_compress_ape = torch.nn.Parameter(ape.clone(), requires_grad=False)
    return indexer.to(device)


def trees(base, cfg, ape, device=None):
    from vllm_neuron.functional.dsa import decode_batch, decode_select, kpool_hadamard
    from vllm_neuron.model.glm5_next import model_fp8
    device = _bench().DEVICE if device is None else device
    return {
        "before": Tree("before", _build_indexer(base.model_fp8, cfg, ape, device),
                       base.kpool_hadamard.dsa_hadamard128, base.decode_batch,
                       base.decode_select),
        "after": Tree("after", _build_indexer(model_fp8, cfg, ape, device),
                      kpool_hadamard.dsa_hadamard128, decode_batch, decode_select),
    }


def _kernel_name(func) -> str:
    """``module.function`` of an ``nki.jit`` kernel (the module tells the trees apart)."""
    for candidate in (func, getattr(func, "func", None), getattr(func, "fn", None)):
        name = getattr(candidate, "__name__", None)
        if isinstance(name, str):
            module = getattr(candidate, "__module__", None) or ""
            return f"{module}.{name}" if module else name
    return repr(func)


def recording_compile(fn, record: list):
    """``torch.compile`` onto the Neuron backend, listing the graph's NKI kernel nodes."""
    torch = _torch()
    from torch._dynamo import lookup_backend
    from libtorch_neuronx_lite.nki.nki_hop import kernel_registry, nki_kernel_wrapper

    neuron = lookup_backend("neuron_libtorch")

    def backend(gm, example_inputs, **kwargs):
        for node in gm.graph.nodes:
            if node.op == "call_function" and (
                    node.target is nki_kernel_wrapper
                    or "nki_kernel_wrapper" in str(node.target)):
                idx = node.kwargs.get("kernel_idx")
                grid = node.kwargs.get("grid")
                record.append({"kernel": _kernel_name(kernel_registry.get_func(idx)),
                               "grid": list(grid) if isinstance(grid, (list, tuple))
                               else grid})
        return neuron(gm, example_inputs, **kwargs)

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def _summarise_kernels(record: list) -> dict:
    out: dict = {}
    for entry in record:
        key = f"{entry['kernel']} grid={entry['grid']}"
        out[key] = out.get(key, 0) + 1
    return out


def _sorted_rows(indices):
    torch = _torch()
    return torch.sort(indices.to(torch.int64), dim=-1).values


def agreement(trees_, layer0, ctx, dense, candidates, pool, k, bound) -> dict:
    """Sets against the torch oracle on the device's own scores; before against after;
    and, when ``bound < ctx``, the chain at ``max_seq_len = bound`` against ``ctx``."""
    torch = _torch()
    bench = _bench()
    names = sorted(layer0)
    result: dict = {}
    after = trees_["after"]

    def after_graph(*flat):
        ops = dict(zip(names, flat))
        if dense:
            got = after.chain(ops, ctx, dense)
            return (ops["pool_bank"] * 1, ops["tail_bank"] * 1)
        query = after.query(ops)
        pooled, _ = after.db.dsa_decode_ring_step(
            ops["tail_bank"], ops["slots"], ops["key"], ops["gate"],
            after.indexer.index_kpool_compress_ape.to(torch.float32), ops["position"])
        bounded = after.db.dsa_decode_scores(
            query, ops["weights"], ops["pool_bank"], ops["slots"], ops["seq_lens"],
            ops["position"], pooled, candidates=candidates, pool_size=pool)
        narrow = {n: (ops[n] * 1 if n in ("pool_bank", "tail_bank") else ops[n])
                  for n in names}
        got = after.chain(ops, ctx, dense)
        if bound == ctx:
            return (got, bounded, ops["pool_bank"] * 1, ops["tail_bank"] * 1)
        got_bound = after.chain(narrow, bound, dense)
        return (got, bounded, got_bound, narrow["pool_bank"] * 1, narrow["tail_bank"] * 1,
                ops["pool_bank"] * 1, ops["tail_bank"] * 1)

    torch._dynamo.reset()
    after.db.reset_decode_batch_dispatch_counters()
    record: list = []
    ops = bench.to_device(layer0)
    outs = [t.to("cpu") for t in recording_compile(after_graph, record)(
        *[ops[n] for n in names])]
    result["after_dispatch"] = {
        "nki_torch": list(after.db.decode_batch_dispatch_counters()),
        "ring_scores_two_program_select": list(after.db.decode_batch_route_counts())}
    if not dense:
        indices, bounded = outs[0], outs[1]
        oracle = after.ds.dsa_decode_select_torch_oracle(
            bounded, layer0["seq_lens"].to(torch.int32), select_k=k, pool_size=pool)
        result["after_equals_oracle_sorted"] = bool(
            torch.equal(_sorted_rows(indices), _sorted_rows(oracle)))
        result["after_equals_oracle_exact"] = bool(torch.equal(indices, oracle))
        result["rows"] = int(indices.shape[0])
        result["selected_per_row"] = sorted({int((r >= 0).sum()) for r in indices})
        after_sets = bench.selected_sets(indices)
        if bound != ctx:
            result["bound_equals_max_model_len_exact"] = bool(torch.equal(outs[2], indices))
            result["bound_banks_bit_identical"] = bool(
                torch.equal(outs[3], outs[5]) and torch.equal(outs[4], outs[6]))
    banks_after = outs[-2:]

    before = trees_["before"]
    torch._dynamo.reset()
    ops = bench.to_device(layer0)

    def before_graph(*flat):
        o = dict(zip(names, flat))
        got = before.chain(o, ctx, dense)
        if got is None:
            return (o["pool_bank"] * 1, o["tail_bank"] * 1)
        return (got, o["pool_bank"] * 1, o["tail_bank"] * 1)

    try:
        got = [t.to("cpu") for t in bench.compiled(before_graph)(*[ops[n] for n in names])]
    except Exception as err:  # noqa: BLE001 -- recorded: e3f38f8 refuses past its ceiling
        result["before"] = f"refused: {type(err).__name__}: {str(err)[:300]}"
        return result
    result["before"] = "served"
    result["banks_bit_identical"] = bool(torch.equal(got[-2], banks_after[0])
                                         and torch.equal(got[-1], banks_after[1]))
    if not dense:
        before_sets = bench.selected_sets(got[0])
        result["rows_with_equal_sets_before_after"] = sum(
            int(a == b) for a, b in zip(after_sets, before_sets))
        # The index tensors themselves (the layout is a function of the set).
        result["before_after_exact"] = bool(torch.equal(got[0], indices))
    return result


def graphs(trees_, layers, names, ctx, dense, candidates, pool, ape_cpu, served, bound):
    """``name -> (fn, device inputs)`` for every timed graph of one case."""
    torch = _torch()
    bench = _bench()
    from vllm_neuron.functional.dsa import decode_batch as live_db
    from vllm_neuron.functional.dsa.kpool_hadamard import _dsa_hadamard128_torch
    n_layers = len(layers)

    def flat_inputs(rows, keys):
        dev = [bench.to_device({kk: ops[kk] for kk in keys}) for ops in rows]
        return tuple(d[kk] for d in dev for kk in keys)

    def unflat(flat, keys):
        return [dict(zip(keys, flat[i * len(keys):(i + 1) * len(keys)]))
                for i in range(n_layers)]

    out = {"empty": (lambda *flat: bench.tails(*flat), flat_inputs(layers, ["key"]))}
    staged = []
    if not dense:
        for ops in layers:
            batch = int(ops["key"].shape[0])
            query = _dsa_hadamard128_torch(ops["query_rows"]).reshape(batch, -1, 128)
            pooled, _ = live_db.dsa_decode_ring_step_torch_oracle(
                ops["tail_bank"], ops["slots"], ops["key"], ops["gate"], ape_cpu.float(),
                ops["position"])
            bounded = live_db.dsa_decode_scores_torch_oracle(
                query, ops["weights"], ops["pool_bank"], ops["slots"], ops["seq_lens"],
                ops["position"], pooled, candidates=candidates, pool_size=pool)
            staged.append({"query": query, "pooled": pooled.to(torch.bfloat16),
                           "bounded": bounded.contiguous(),
                           "seq_lens32": ops["seq_lens"].to(torch.int32), **ops})
    for tname, tree in trees_.items():
        if tname not in served:
            continue

        def chain(*flat, _t=tree):
            res = []
            for ops in unflat(flat, names):
                got = _t.chain(ops, ctx, dense)
                res.append(ops["tail_bank"] if got is None else got)
            return bench.tails(*res)
        out[f"chain_{tname}"] = (chain, flat_inputs(layers, names))
        if tname == "after" and bound != ctx:
            def chain_bound(*flat, _t=tree):
                res = []
                for ops in unflat(flat, names):
                    res.append(_t.chain(ops, bound, dense))
                return bench.tails(*res)
            out["chain_after_bound"] = (chain_bound, flat_inputs(layers, names))

        ring_keys = ["tail_bank", "slots", "key", "gate", "position"]

        def ring(*flat, _t=tree):
            res = []
            for ops in unflat(flat, ring_keys):
                res.extend(_t.db.dsa_decode_ring_step(
                    ops["tail_bank"], ops["slots"], ops["key"], ops["gate"],
                    _t.indexer.index_kpool_compress_ape.to(torch.float32), ops["position"]))
            return bench.tails(*res)
        out[f"ring_{tname}"] = (ring, flat_inputs(layers, ring_keys))
        if dense:
            continue

        score_keys = ["query", "weights", "pool_bank", "slots", "seq_lens", "position",
                      "pooled"]

        def scores(*flat, _t=tree):
            return bench.tails(*[_t.db.dsa_decode_scores(
                o["query"], o["weights"], o["pool_bank"], o["slots"], o["seq_lens"],
                o["position"], o["pooled"], candidates=candidates, pool_size=pool)
                for o in unflat(flat, score_keys)])
        out[f"scores_{tname}"] = (scores, flat_inputs(staged, score_keys))

        def select(*flat, _t=tree):
            return bench.tails(*[_t.select(o["bounded"], o["seq_lens32"])
                                 for o in unflat(flat, ["bounded", "seq_lens32"])])
        out[f"select_{tname}"] = (select, flat_inputs(staged, ["bounded", "seq_lens32"]))
    return out


def _samples(fn, inputs, per: int, warmup: int, iterations: int) -> list:
    """Per-unit wall times (us) of ``iterations`` synchronised calls after ``warmup``."""
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    got = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        got.append((time.perf_counter_ns() - started) / 1000.0 / per)
    return got


def run_case(batch: int, ctx: int, max_model_len: int, base, args) -> dict:
    torch = _torch()
    bench = _bench()
    from test.vllm_neuron.functional.dsa import dsa_decode_case as case
    from vllm_neuron.functional.dsa.decode_bypass import selection_is_a_no_op

    torch._dynamo.reset()
    cfg = case.decode_config()
    pool, k = int(cfg.index_kpool), int(cfg.index_topk) // int(cfg.index_kpool)
    # ``ctx`` is the window (the decode bucket); e3f38f8's call site passes the window
    # when the selection is a no-op there and max_model_len otherwise.
    dense = selection_is_a_no_op(int(ctx), int(cfg.index_topk), pool)
    axis = int(ctx) if dense else int(max_model_len)
    candidates = axis // pool
    ape = (torch.randn(pool, int(cfg.index_head_dim),
                       generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    trees_ = trees(base, cfg, ape)
    layers = [layer_operands(cfg, ctx, batch, seed=1000 * batch + ctx + 7 * i,
                             max_model_len=max_model_len)
              for i in range(args.layers)]
    if args.form == "views":
        # One slot vector for every layer, so one python tuple of views serves them all.
        for ops in layers[1:]:
            ops["slots"] = layers[0]["slots"].clone()
        for tree in trees_.values():
            tree.form = "views"
            tree.view_slots = tuple(int(s) for s in layers[0]["slots"].tolist())
    names = sorted(layers[0])

    agree = agreement(trees_, layers[0], axis, dense, candidates, pool, k, int(ctx))
    print(json.dumps({"B": batch, "ctx": ctx, "agreement": agree}), flush=True)
    if not dense and not agree.get("after_equals_oracle_sorted"):
        raise AssertionError(f"B={batch} ctx={ctx}: after's sets differ from the oracle")
    if axis != int(ctx) and not (agree.get("bound_equals_max_model_len_exact")
                                 and agree.get("bound_banks_bit_identical")):
        raise AssertionError(f"B={batch} ctx={ctx}: the bound chain differs: {agree}")
    if agree.get("before") == "served" and (
            not agree.get("banks_bit_identical")
            or (not dense and (agree.get("rows_with_equal_sets_before_after") != batch
                               or not agree.get("before_after_exact")))):
        raise AssertionError(f"B={batch} ctx={ctx}: before and after disagree: {agree}")
    served = {"after"} | ({"before"} if agree.get("before") == "served" else set())

    timing, kernels, ready = {}, {}, {}
    torch._dynamo.reset()
    for gname, (fn, inputs) in graphs(trees_, layers, names, axis, dense, candidates, pool,
                                      ape, served, int(ctx)).items():
        if args.graphs and gname not in args.graphs and not gname.startswith("chain"):
            continue
        record: list = []
        cfn = recording_compile(fn, record)
        started = time.perf_counter()
        cfn(*inputs).to("cpu")
        first = time.perf_counter() - started
        kernels[gname] = _summarise_kernels(record)
        ready[gname] = (cfn, inputs, first)
    # Every graph is compiled first, then timed in rounds that visit each graph in turn,
    # so a drift in the device's speed (other slices' work on the chip) spreads over all
    # graphs alike instead of landing on whichever graph ran during it.
    samples = {gname: [] for gname in ready}
    per_round = {gname: [] for gname in ready}
    rounds = max(1, int(args.rounds))
    for r in range(rounds):
        for gname, (cfn, inputs, _first) in ready.items():
            per = 1 if gname == "empty" else args.layers
            got = _samples(cfn, inputs, per, args.warmup if r == 0 else 2,
                           max(1, args.iterations // rounds))
            samples[gname].extend(got)
            per_round[gname].append(statistics.median(got))
    for gname, (cfn, inputs, first) in ready.items():
        got = sorted(samples[gname])
        timing[gname] = {
            "iterations": len(got), "rounds": rounds,
            "units_per_sample": 1 if gname == "empty" else args.layers,
            "median_us": statistics.median(got),
            "p90_us": got[min(len(got) - 1, int(0.9 * len(got)))],
            "min_us": got[0], "max_us": got[-1],
            "round_medians_us": [round(v, 2) for v in per_round[gname]],
            "first_call_s": first}
        print(json.dumps({"B": batch, "ctx": ctx, "graph": gname,
                          "median_us": round(timing[gname]["median_us"], 1),
                          "round_medians_us": timing[gname]["round_medians_us"],
                          "kernels": kernels[gname]}), flush=True)
        if args.profile_dir is not None and gname.startswith("chain_"):
            bench._profile(cfn, inputs, args, f"b{batch}_c{ctx}_m{max_model_len}_{gname}")

    row = {"batch": batch, "ctx": ctx, "form": args.form, "max_model_len": max_model_len,
           "max_seq_len_before_call_site": axis, "max_seq_len_after_call_site": int(ctx),
           "candidates": candidates,
           "bank_rows": int(max_model_len) // pool + 1, "select_k": k, "dense": dense,
           "lengths": bench.lengths_for(ctx, batch)[:8], "layers": args.layers,
           "agreement": agree, "kernels_in_graph": kernels, "timing": timing,
           "speedup": {}}
    for stage in ("chain", "ring", "scores", "select"):
        b, a = timing.get(f"{stage}_before"), timing.get(f"{stage}_after")
        if b and a:
            row["speedup"][stage] = {
                "before_median_us": b["median_us"], "after_median_us": a["median_us"],
                "before_p90_us": b["p90_us"], "after_p90_us": a["p90_us"],
                "median_speedup": b["median_us"] / a["median_us"],
                "after_over_before": a["median_us"] / b["median_us"]}
    a, c = timing.get("chain_after"), timing.get("chain_after_bound")
    if a and c:
        row["call_site_bound"] = {
            "max_seq_len_max_model_len_median_us": a["median_us"],
            "max_seq_len_bound_median_us": c["median_us"],
            "saving_per_layer_us": a["median_us"] - c["median_us"],
            "saving_per_step_11_layers_us": 11 * (a["median_us"] - c["median_us"])}
    return row


def cmd_run(args) -> None:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    _loader, base = _bench().load_baseline(args.baseline_module.resolve())
    from vllm_neuron.functional.dsa import decode_batch as live_db
    if args.unroll_blocks is not None:
        # A kernel argument, so the compiled kernel (and its cache key) follows it.
        live_db.UNROLL_BLOCKS = int(args.unroll_blocks)
    output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = json.loads(output.read_text()) if args.merge and output.exists() else {}
    report.setdefault("cases", [])
    report.update({
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
        "tree": str(ROOT),
        "baseline": {"commit": base.commit, "directory": base.directory},
        "args": {"layers": args.layers, "warmup": args.warmup, "form": args.form,
                 "iterations": args.iterations, "cases": args.cases,
                 "profile_dir": None if args.profile_dir is None else str(args.profile_dir),
                 "profile_iterations": args.profile_iterations,
                 "unroll_blocks": live_db.UNROLL_BLOCKS, "rounds": args.rounds},
        "unit": "graph wall time / layers, microseconds; 'empty' is per call",
        "rounds": "every graph of a case is compiled first, then timed in --rounds rounds "
                  "that visit each graph in turn (iterations / rounds calls a visit)",
        "synchronization": "graph output (last 8 elements of each output) copied to CPU "
                           "on every timed call",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    for spec in args.cases:
        parts = [int(v) for v in spec.split(":")]
        batch, ctx = parts[0], parts[1]
        mml = parts[2] if len(parts) > 2 else ctx
        if mml < ctx:
            raise ValueError(f"case {spec}: max_model_len below the bucket")
        row = run_case(batch, ctx, mml, base, args)
        key = (batch, ctx, mml, args.form)
        report["cases"] = [r for r in report["cases"]
                           if (r["batch"], r["ctx"], r.get("max_model_len", r["ctx"]),
                               r.get("form", "bank")) != key] + [row]
        report["cases"].sort(key=lambda r: (r.get("form", "bank"),
                                            r.get("max_model_len", r["ctx"]) != r["ctx"],
                                            r["ctx"], r["batch"]))
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"B": batch, "ctx": ctx, "speedup": row["speedup"]}), flush=True)


#: ``select`` cases, ``B:C``: the selection's request-major tiles of 2 to 8 requests, the
#: segment path (C > 16384) with one to eight requests a GpSimd call, nine segments, and
#: the widest axis. 16:32768 is where worker-47's D3 found the segment lists read back
#: as zeros on the device.
SELECT_CASES = ("1:2048", "4:2048", "64:2048", "4:8192", "16:16384", "64:16384",
                "4:20480", "8:32768", "16:32768", "64:32768", "4:65536", "8:65536",
                "64:65536", "4:65537", "2:131072", "1:262144", "1:524288")


def cmd_select(args) -> None:
    """The selection kernel alone on the device, bit for bit against the torch oracle."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware check cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    torch = _torch()
    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    from vllm_neuron.functional.dsa import decode_select as DS
    from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
    k, pool = 2048 // 4, 4
    report = {"what": cmd_select.__doc__, "tree": str(ROOT), "layers": args.layers,
              "runs": args.runs, "select_k": k, "pool_size": pool,
              "environment": {key: os.environ.get(key) for key in (
                  "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                  "NEURON_LIBTORCH_CACHE_ROOT")}, "cases": []}
    for spec in args.cases:
        batch, width = (int(v) for v in spec.split(":"))
        gen = torch.Generator().manual_seed(width + batch)
        inputs, want = [], []
        for _ in range(args.layers):
            scores = torch.randn(batch, width, generator=gen)
            lens = torch.full((batch,), pool * width, dtype=torch.int32)
            for r in range(1, batch, 2):  # odd rows: a shorter request, BOUND_FILL past it
                complete = width // 2 + 7 * r
                lens[r] = pool * complete + 2
                scores[r, complete:] = BOUND_FILL
            inputs += [scores, lens]
            want.append(DS.dsa_decode_select_torch_oracle(scores, lens, select_k=k,
                                                          pool_size=pool))
        torch._dynamo.reset()

        def graph(*xs):
            return torch.stack([DS.dsa_decode_select(xs[2 * i], xs[2 * i + 1], select_k=k,
                                                     pool_size=pool)
                                for i in range(len(xs) // 2)])

        compiled = torch.compile(graph, backend="neuron_libtorch", fullgraph=True,
                                 dynamic=False)
        dev = [t.to("neuron:0") for t in inputs]
        bad = []
        for _ in range(args.runs):
            got = compiled(*dev).to("cpu")
            bad.append(sum(not torch.equal(got[i, r], want[i][r])
                           for i in range(args.layers) for r in range(batch)))
        row = {"case": spec, "batch": batch, "candidates": width,
               "programs": DS.decode_select_programs(batch),
               "fold_cap": DS.select_fold_cap(width), "segments": DS.segments(width)[0],
               "mismatched_rows_per_run": bad, "exact": not any(bad)}
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if not all(row["exact"] for row in report["cases"]):
        raise SystemExit("the selection differs from the oracle on the device")


def _kernel_call(tree: str, kernel: str, batch: int, cands: int, programs: int,
                 unroll_blocks: int | None = None):
    """``(nki function, {argument: fake tensor or int}, grid)`` of one chain kernel;
    ``unroll_blocks`` overrides ``decode_batch.UNROLL_BLOCKS`` for the score kernel."""
    torch = _torch()
    if tree == "after":
        from vllm_neuron.functional.dsa import decode_batch as db, decode_select as ds
    else:
        base = _bench().load_baseline(ROOT / "test/hardware/baselines/dsa_capped_select")[1]
        db, ds = base.decode_batch, base.decode_select
    from vllm_neuron.functional.dsa.index_expand import index_expand_width
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    heads, dim, pool, k = 32, 128, 4, 512
    slots = batch + 3

    def fake(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    if kernel == "ring":
        return db.dsa_decode_ring_step_kernel, {
            "tail_hbm": fake(slots, 2 * pool * dim, dtype=bf),
            "slots_hbm": fake(batch, 1, dtype=i32), "key_hbm": fake(batch, dim, dtype=bf),
            "score_hbm": fake(batch, dim, dtype=bf), "ape_hbm": fake(pool, dim, dtype=f32),
            "pos_hbm": fake(batch, 1, dtype=i32), "pool_size": pool,
            "source_digest": db.SOURCE_DIGEST}, (programs,)
    if kernel == "scores":
        args = {
            "q_hbm": fake(batch, heads, dim, dtype=bf), "w_hbm": fake(batch, heads, dtype=f32),
            "bank_hbm": fake(slots, cands + 1, dim, dtype=bf),
            "slots_hbm": fake(batch, 1, dtype=i32), "lens_hbm": fake(batch, 1, dtype=i32),
            "pos_hbm": fake(batch, 1, dtype=i32), "pooled_hbm": fake(batch, dim, dtype=bf),
            "candidates": cands, "pool_size": pool}
        if hasattr(db, "score_blocks"):
            args["block_tiles"] = db.score_blocks(batch, cands, programs)
        if hasattr(db, "UNROLL_BLOCKS"):
            args["unroll_blocks"] = (db.UNROLL_BLOCKS if unroll_blocks is None
                                     else int(unroll_blocks))
        args["source_digest"] = db.SOURCE_DIGEST
        return db.dsa_decode_scores_kernel, args, (programs,)
    if kernel == "select":
        return ds.dsa_decode_select_kernel, {
            "bounded_hbm": fake(batch, cands, dtype=f32), "lens_hbm": fake(batch, 1, dtype=i32),
            "select_k": k, "pool_size": pool, "out_cols": int(index_expand_width(k, pool)),
            **({"fold_cap": int(ds.select_fold_cap(cands))}
               if hasattr(ds, "select_fold_cap") else {}),
            "source_digest": ds.SOURCE_DIGEST}, (programs,)
    raise ValueError(f"unknown kernel {kernel}")


def _chain_call(tree: str, batch: int, cands: int):
    """``(fn, meta inputs)``: the decode chain as the call site runs it, at ``ctx = 4 C``
    and ``max_seq_len = ctx`` (``Tree.chain``), operands on the meta device."""
    torch = _torch()
    from test.vllm_neuron.functional.dsa import dsa_decode_case as case
    base = _bench().load_baseline(ROOT / "test/hardware/baselines/dsa_capped_select")[1]
    cfg = case.decode_config()
    pool = int(cfg.index_kpool)
    ctx = int(cands) * pool
    ape = torch.zeros(pool, int(cfg.index_head_dim), dtype=torch.bfloat16)
    # The compile takes meta parameters and inputs only, and never opens the runtime.
    chain_tree = trees(base, cfg, ape, device="meta")[tree]
    # layer_operands' shapes and dtypes, on the meta device (no host memory at 1M).
    shapes = {name: (tuple(t.shape), t.dtype) for name, t in layer_operands(
        cfg, 2 * pool, batch, seed=1, max_model_len=2 * pool).items()}
    rows = ctx // pool + 1
    shapes["pool_bank"] = ((batch + 3, rows, int(cfg.index_head_dim)), torch.bfloat16)
    names = sorted(shapes)

    def fn(*flat):
        return chain_tree.chain(dict(zip(names, flat)), ctx, False)

    return fn, [torch.empty(shapes[n][0], dtype=shapes[n][1], device="meta") for n in names]


def _full_compile(fn, inputs) -> dict:
    """Dynamo -> HLO -> neuronx-cc -> NEFF on the CPU; the executor is never built."""
    torch = _torch()
    import torch._dynamo.backends.registry as registry
    import libtorch_neuronx_lite  # noqa: F401
    import libtorch_neuronx_lite.compile.backend as backend

    # The package registers its backend only where /dev/neuron* exists; a compile run in
    # a namespace without device nodes (w47/nodev.sh, the proof that it opens none)
    # registers it here.
    if "neuron_libtorch" not in registry.list_backends():
        registry.register_backend(compiler_fn=backend.compile, name="neuron_libtorch")

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    try:
        torch.compile(fn, backend="neuron_libtorch", fullgraph=True)(*inputs)
    except BaseException as caught:  # noqa: BLE001 -- the compiler's outcome is the result
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
            return {"neff_bytes": os.path.getsize(neff) if os.path.isfile(neff) else None}
        return {"error": text[:1500] or type(caught).__name__}
    return {"error": "the compiled graph ran; build_executable was not replaced"}


def require_no_unleased_device() -> None:
    """Refuse a compile that could reach a device it does not hold.

    A compile needs no device, and the compile paths here never move a tensor or module
    to one (strace in a namespace without device nodes: no ``/dev/neuron*`` access). But a
    runtime opened without ``NEURON_RT_VISIBLE_CORES`` claims and resets every chip of the
    host (2026-10-07 18:22Z), so this runs only where that cannot happen: under the device
    lease (``devlease.py slice <name> --``, which pins ``NEURON_RT_VISIBLE_CORES`` to its
    chip, kept for the children) or where no ``/dev/neuron*`` node exists at all.
    """
    import glob
    if glob.glob("/dev/neuron*") and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise SystemExit(
            "compile: /dev/neuron* is visible and NEURON_RT_VISIBLE_CORES is not set; run "
            "under `devlease.py slice <name> --` or in a namespace without device nodes")


def cmd_compile_one(args) -> None:
    import resource
    require_no_unleased_device()
    if args.full:
        torch = _torch()
        if args.kernel == "chain":
            fn, inputs = _chain_call(args.tree, args.batch, args.cands)
        else:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
            func, kwargs, grid = _kernel_call(args.tree, args.kernel, args.batch,
                                              args.cands, args.programs, args.unroll_blocks)
            inputs = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
            scalars = [v for v in kwargs.values() if not isinstance(v, torch.Tensor)]
            call = wrap_nki(func)
            if grid[0] == 2:
                call = call[2]

            def fn(*tensors):
                return call(*tensors, *scalars)

        started = time.perf_counter()
        result = _full_compile(fn, inputs)
        elapsed = time.perf_counter() - started
        print("COMPILE_ROW " + json.dumps({
            "tree": args.tree, "kernel": args.kernel, "batch": args.batch,
            "candidates": args.cands, "programs": args.programs, "full": True,
            "unroll_blocks": args.unroll_blocks,
            "compile_s": round(elapsed, 2),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "peak_child_rss_mib": round(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
            **result}), flush=True)
        return
    from libtorch_neuronx_lite.nki.nki_compile import compile_nki
    func, kwargs, grid = _kernel_call(args.tree, args.kernel, args.batch, args.cands,
                                      args.programs, args.unroll_blocks)
    # The raw function, as ``wrap_nki`` registers it.
    func = func.func if hasattr(func, "func") else func
    started = time.perf_counter()
    result = compile_nki(func, kwargs, grid)
    elapsed = time.perf_counter() - started
    print("COMPILE_ROW " + json.dumps({
        "tree": args.tree, "kernel": args.kernel, "batch": args.batch,
        "candidates": args.cands, "programs": args.programs, "compile_s": round(elapsed, 2),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "config_b64_bytes": len(result.dumped_config)}), flush=True)


def cmd_compile(args) -> None:
    import subprocess
    require_no_unleased_device()
    # NEURON_RT_VISIBLE_CORES stays: under the lease it confines the children to its chip.
    drop = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
            "NEURON_LIBTORCH_REMOTE_CACHE")
    rows = []
    output = args.output.resolve()
    for spec in args.entries:
        tree, kernel, batch, cands = spec.split(":")
        batch, cands = int(batch), int(cands)
        programs = 2 if (batch >= 2 or kernel != "select") and args.lnc == 2 else 1
        if tree == "before" and kernel != "select" and batch < 2:
            programs = 1                      # e3f38f8 ran B = 1 on one program
        with tempfile.TemporaryDirectory(prefix="dsa_ctx_compile_") as scratch:
            env = {k: v for k, v in os.environ.items() if k not in drop}
            env.update(VLLM_NEURON_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
                       NEURON_LOGICAL_NC_CONFIG=str(args.lnc), PYTHONDONTWRITEBYTECODE="1",
                       NEURON_LIBTORCH_CACHE_ROOT=scratch, PYTHONPATH=str(ROOT),
                       NKI_COMPILE_CACHE_URL=os.path.join(scratch, "nki"))
            if args.full:
                env.pop("VLLM_NEURON_CPU_COMPILE")
                env.update(NEURON_LIBTORCH_CPU_COMPILE="1",
                           NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
                           PATH=f"{Path(sys.executable).parent}:{env.get('PATH', '')}")
            cmd = [sys.executable, str(Path(__file__).resolve()), "compile-one",
                   "--tree", tree, "--kernel", kernel, "--batch", str(batch),
                   "--cands", str(cands), "--programs", str(programs)]
            if args.full:
                cmd.append("--full")
            started = time.perf_counter()
            try:
                done = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True,
                                      text=True, timeout=args.timeout)
                text = done.stdout + done.stderr
                found = [json.loads(line.split("COMPILE_ROW ", 1)[1])
                         for line in text.splitlines() if line.startswith("COMPILE_ROW ")]
                row = found[0] if found else {
                    "tree": tree, "kernel": kernel, "batch": batch, "candidates": cands,
                    "programs": programs, "error": text.strip().splitlines()[-3:]}
            except subprocess.TimeoutExpired:
                row = {"tree": tree, "kernel": kernel, "batch": batch, "candidates": cands,
                       "programs": programs, "error": f"timeout after {args.timeout} s"}
            row["child_wall_s"] = round(time.perf_counter() - started, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        what = ("whole compile (dynamo -> HLO -> neuronx-cc -> NEFF) of one kernel call "
                "or of the chain graph, CPU, no device, empty caches per entry" if args.full
                else "NKI compile (front end to BIR) per chain kernel, CPU, empty kernel "
                "cache per entry")
        output.write_text(json.dumps({"what": what,
                                      "tree": str(ROOT), "rows": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--baseline-module", type=Path,
                     default=ROOT / "test/hardware/baselines/dsa_capped_select")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--profile-dir", type=Path)
    run.add_argument("--profile-iterations", type=int, default=3)
    run.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    run.add_argument("--graphs", nargs="*", help="time only these graphs (chains always)")
    run.add_argument("--form", choices=("bank", "views"), default="bank",
                     help="the call site's form: whole banks + slots, or per-request views")
    run.add_argument("--layers", type=int, default=11)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--iterations", type=int, default=50)
    run.add_argument("--rounds", type=int, default=5,
                     help="timing rounds; each visits every graph for iterations / rounds "
                          "calls")
    run.add_argument("--unroll-blocks", type=int,
                     help="decode_batch.UNROLL_BLOCKS for this run (0: every uniform score "
                          "block in the device loop)")
    run.add_argument("--merge", action="store_true",
                     help="keep the output file's rows for cases this run skips")
    sel = sub.add_parser("select")
    sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--cases", nargs="+", default=list(SELECT_CASES))
    sel.add_argument("--layers", type=int, default=11)
    sel.add_argument("--runs", type=int, default=5)
    cores = sub.add_parser("cores")
    cores.add_argument("--data", required=True, help="explorer --data-path")
    cores.add_argument("--name", required=True, help="explorer --display-name")
    cores.add_argument("--functions", action="store_true",
                       help="label by file.py:function rather than by file")
    comp = sub.add_parser("compile")
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--entries", nargs="+", required=True,
                      help="tree:kernel:B:C, tree in {before, after}, kernel in "
                           "{ring, scores, select}")
    comp.add_argument("--lnc", type=int, default=2)
    comp.add_argument("--timeout", type=int, default=1500)
    comp.add_argument("--full", action="store_true",
                      help="the whole compile to a NEFF (kernel in {ring, scores, select, "
                           "chain}; chain: ctx = 4 C) rather than the NKI front end alone")
    one = sub.add_parser("compile-one")
    one.add_argument("--tree", required=True)
    one.add_argument("--kernel", required=True)
    one.add_argument("--batch", type=int, required=True)
    one.add_argument("--cands", type=int, required=True)
    one.add_argument("--programs", type=int, required=True)
    one.add_argument("--full", action="store_true")
    one.add_argument("--unroll-blocks", type=int,
                     help="the score kernel's unroll_blocks (default decode_batch.UNROLL_BLOCKS)")
    args = parser.parse_args()
    if args.cmd == "compile":
        cmd_compile(args)
        return
    if args.cmd == "compile-one":
        cmd_compile_one(args)
        return
    if args.cmd == "select":
        cmd_select(args)
        return
    if args.cmd == "run":
        if args.layers < 1 or args.iterations < 1 or args.warmup < 0:
            raise ValueError("Use positive layer and iteration counts")
        cmd_run(args)
    else:
        _bench().cmd_cores(args)


if __name__ == "__main__":
    main()
