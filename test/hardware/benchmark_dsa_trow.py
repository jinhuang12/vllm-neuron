# SPDX-License-Identifier: Apache-2.0
"""Time the DSA decode legs with ``T = 4`` tokens per request on Neuron against ``T = 1``.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores. A speculative verify step hands the legs ``T = 1 + k`` tokens per request
in one call; what it must beat is ``T`` one-token steps, each its own graph launch, and
what it must approach is the bytes it moves at the rank's HBM bandwidth. Three tables:

* ``attention`` -- ``mla_decode_attention`` at ``T = 4`` against four ``T = 1`` launches,
  dense at ctx 1024 in a 2048-row window and selected at ctx 8192 in an 8192-row window,
  at B in {1, 4}; and ``T = 1`` against commit ea04c81's kernel (the snapshot under
  ``test/hardware/baselines/mla_decode_ea04c81``), which it must match within noise.
* ``indexer`` -- the ring step (``dsa_decode_ring_rows`` at depth 8, ``T = 4`` against
  four ``T = 1`` launches and against four launches of ``decode_batch``'s one-token step
  at depth 4) and the scores (``dsa_decode_scores_rows`` at 2048 candidates, ``T = 4``
  against four ``decode_batch`` launches), at B in {1, 4}.
* ``layer`` -- one layer's ``Glm5NextMLAAttention._forward_requests`` (indexer,
  projections, attention, ``o_proj``) at the per-rank TP=64 shape, ``T = 4`` against four
  ``T = 1`` launches: ctx 1024 in a 2048-row window with ``max_seq_len`` 2048 (the
  bypass) and ctx 8192 in an 8192-row window with ``max_seq_len`` 8192 (selection), at
  B in {1, 4}. ``--layers`` distinct layers are chained in one graph, each layer's
  output feeding the next one's input; the figure is the graph's time over the count.

Every row carries its bytes/bandwidth entitlement: the bytes the step must move (bank
rows read through the tables, weights, operands and results; formulas in
:func:`attention_bytes`, :func:`indexer_bytes`, :func:`layer_bytes`) over
``HBM_BYTES_PER_S``, and the measured time over it. Compilation and warmup are excluded.
Each timed call copies the graph's output to CPU to synchronise, so a sample includes one
dispatch; chaining amortises it. Outputs are checked finite, and the attention against
the CPU oracle, before any timing.
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

import vllm_neuron  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402,F401
from vllm_neuron.functional.attention import mla_decode as MD  # noqa: E402
from vllm_neuron.functional.dsa import decode_batch as DB  # noqa: E402
from vllm_neuron.functional.dsa import decode_tail_update as TU  # noqa: E402
from vllm_neuron.functional.dsa import decode_trow as TR  # noqa: E402
from vllm_neuron.model.glm5_next import model_fp8  # noqa: E402

from test.vllm_neuron.functional.dsa import dsa_decode_case as case  # noqa: E402

DEVICE = "neuron:0"
ROWS = 4
"""Tokens per request of the verify step timed here: ``1 + k`` at ``k = 3``."""
HBM_BYTES_PER_S = 716.0e9
"""The per-rank HBM bandwidth every entitlement in the campaign's reports uses
(``reports/mtp.md``, section 3: a 64 MiB GEMV in ~94 us)."""
ENTITLEMENT_RATIO_LIMIT = 1.5
"""The NKI mandate's ceiling on measured time over the bytes/bandwidth entitlement."""
LATENT, INDEX_DIM, INDEX_HEADS, POOL, TOPK = 512, 128, 32, 4, 2048
DEEP = TU.ring_depth_for(POOL, 6)
"""The ring depth a config with up to five draft tokens allocates (8)."""
#: (label, ctx, window pages): the bypass window and the selecting one.
ATTENTION_CASES = (("dense", 1024, 16), ("selected", 8192, 64))
#: (label, ctx, window pages, max_seq_len).
LAYER_CASES = (("bypass", 1024, 16, 2048), ("selecting", 8192, 64, 8192))
KERNEL_REL_L2 = 4e-3
BASELINE_DIR = ROOT / "test" / "hardware" / "baselines" / "mla_decode_ea04c81"


def load_baseline(directory: Path):
    """The baseline package's ``load()``: ea04c81's ``mla_decode``, read by ``git show``."""
    spec = importlib.util.spec_from_file_location(
        "mla_decode_baseline_loader", directory / "__init__.py",
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
    out = fn(*inputs)
    out = tuple(o.to("cpu") for o in out) if isinstance(out, tuple) else out.to("cpu")
    return out, time.perf_counter() - started


def measure(fn, inputs, per: int, warmup: int, iterations: int) -> dict:
    """Median and p90 of one unit (graph time / ``per``), in microseconds."""
    def sync(out):
        if isinstance(out, tuple):
            for o in out:
                o.to("cpu")
        else:
            out.to("cpu")

    for _ in range(warmup):
        sync(fn(*inputs))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        sync(fn(*inputs))
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


def entitlement(step_bytes: int, measured_us: float) -> dict:
    floor_us = step_bytes / HBM_BYTES_PER_S * 1.0e6
    return {"bytes": int(step_bytes), "entitlement_us": floor_us,
            "measured_over_entitlement": measured_us / floor_us,
            "within_limit": measured_us / floor_us <= ENTITLEMENT_RATIO_LIMIT}


def verdict(rows_timing: dict, one_timing: dict, step_bytes: int, rows: int = ROWS) -> dict:
    """The T-row step against ``rows`` one-token launches and its entitlement."""
    rows_us = rows_timing["median_us"]
    one_us = one_timing["median_us"]
    return {
        "rows": rows_timing,
        "one": one_timing,
        "one_times_rows_us": one_us * rows,
        "rows_over_one_times_rows": rows_us / (one_us * rows),
        "beats_sequential": rows_us < one_us * rows,
        "entitlement": entitlement(step_bytes, rows_us),
        "one_entitlement": entitlement(step_bytes, one_us),
    }


# --- attention -----------------------------------------------------------------------

def attention_operands(batch, rows, pages, context, selected, seed):
    """One step of ``batch`` requests x ``rows`` rows ending at window row ``context - 1``,
    each request on its own shuffled pages."""
    gen = torch.Generator().manual_seed(seed)
    bank_pages = pages * batch + 3
    bank = (torch.randn(bank_pages * case.PAGE, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    order = torch.randperm(bank_pages, generator=gen)
    table = order[:pages * batch].reshape(batch, pages).to(torch.int32)
    used = -(-context // case.PAGE)
    table[:, used:] = -1
    q = torch.randn(batch * rows, 1, LATENT, generator=gen).to(torch.bfloat16)
    written = (torch.randn(batch * rows, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    start = context - rows
    pos = torch.full((batch,), start, dtype=torch.int32)
    indices = None
    if selected:
        picks = []
        for b in range(batch):
            for t in range(rows):
                picks.append(torch.randperm(start + t + 1, generator=gen)[:TOPK])
        indices = torch.stack(picks).to(torch.int32)
    return q, bank, table, pos, written, indices


def attention_bytes(batch, rows, pages, selected) -> int:
    """Bytes one attention step moves: the window rows the kernel reads (dense: every
    table entry's page; selected: ``TOPK`` gathered rows per query row), the queries,
    the step's own rows, the table, the indices and the fp32 result."""
    row_bytes = LATENT * 2
    if selected:
        window = batch * rows * TOPK * row_bytes + batch * rows * TOPK * 4
    else:
        window = batch * pages * case.PAGE * row_bytes
    operands = batch * rows * row_bytes * 2 + batch * pages * 4 + batch * 4
    return window + operands + batch * rows * LATENT * 4


def attention_case(label, context, pages, batch, base, args) -> dict:
    torch._dynamo.reset()
    selected = label == "selected"
    chain = args.kernel_chain
    scale = 256.0 ** -0.5

    def graph_for(module, rows):
        def fn(*flat):
            outs = []
            for index in range(chain):
                q, bank, table, pos, written, indices = flat[6 * index:6 * index + 6]
                outs.append(module.mla_decode_attention(
                    q, bank, table, pos, written, scale, case.PAGE,
                    indices if selected else None))
            return torch.stack(outs, 0)
        return fn

    timing, checks = {}, {}
    for variant, module, rows in (("rows", MD, ROWS), ("one", MD, 1),
                                  ("one_ea04c81", base.mla_decode, 1)):
        sets = [attention_operands(batch, rows, pages, context, selected, seed=5 + index)
                for index in range(chain)]
        inputs = tuple(t.to(DEVICE) for one in sets for t in one
                       if t is not None)
        if not selected:
            # No indices operand: six slots per set become five.
            def fn(*flat, rows=rows, module=module):
                outs = []
                for index in range(chain):
                    q, bank, table, pos, written = flat[5 * index:5 * index + 5]
                    outs.append(module.mla_decode_attention(
                        q, bank, table, pos, written, scale, case.PAGE, None))
                return torch.stack(outs, 0)
        else:
            fn = graph_for(module, rows)
        graph = compiled(fn)
        out, seconds = first_call(graph, inputs)
        oracle = torch.stack([MD.mla_decode_attention_torch_oracle(
            one[0], one[1], one[2], one[3], one[4], scale, case.PAGE,
            one[5] if selected else None) for one in sets], 0)
        checks[variant] = rel_l2(out, oracle)
        timing[variant] = measure(graph, inputs, chain, args.warmup, args.iterations)
        timing[variant]["first_call_s"] = seconds
        del graph
    if max(checks.values()) > KERNEL_REL_L2:
        raise AssertionError(f"attention {label} B={batch}: vs CPU oracle {checks}")
    return {
        "table": "attention",
        "case": label,
        "ctx": context,
        "window_rows": pages * case.PAGE,
        "batch": batch,
        "rows": ROWS,
        "calls_chained": chain,
        "programs": MD._programs(batch),
        "unit": "one decode attention launch for all `batch` requests",
        "vs_cpu_oracle_relative_l2": checks,
        **verdict(timing["rows"], timing["one"],
                  attention_bytes(batch, ROWS, pages, selected)),
        "one_ea04c81": timing["one_ea04c81"],
        "one_over_one_ea04c81": timing["one"]["median_us"] / timing["one_ea04c81"]["median_us"],
    }


# --- indexer -------------------------------------------------------------------------

def indexer_operands(batch, rows, depth, candidates, seed):
    gen = torch.Generator().manual_seed(seed)
    slots_total = batch + 3
    tail = (torch.randn(slots_total, 2, depth, INDEX_DIM, generator=gen) * 0.5).to(torch.bfloat16)
    pool_bank = (torch.randn(slots_total, candidates + 1, INDEX_DIM, generator=gen) * 0.5
                 ).to(torch.bfloat16)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    key = torch.randn(batch * rows, INDEX_DIM, generator=gen).to(torch.bfloat16)
    score = torch.randn(batch * rows, INDEX_DIM, generator=gen).to(torch.bfloat16)
    ape = torch.randn(POOL, INDEX_DIM, generator=gen) * 0.1
    position = torch.full((batch,), candidates * POOL - rows, dtype=torch.int32)
    query = torch.randn(batch * rows, INDEX_HEADS, INDEX_DIM, generator=gen).to(torch.bfloat16)
    weights = torch.randn(batch * rows, INDEX_HEADS, generator=gen) * INDEX_HEADS ** -0.5
    pooled = torch.randn(batch * rows, INDEX_DIM, generator=gen).to(torch.bfloat16)
    return tail, pool_bank, slots, key, score, ape, position, query, weights, pooled


def indexer_bytes(batch, rows, depth, candidates, stage) -> int:
    """Bytes one indexer stage moves. Ring: each request's ring read and written, its
    tokens and ``ape`` read, its pooled rows written. Scores: each request's candidate
    rows and its rows' queries, weights and pooled read, the fp32 scores written."""
    if stage == "ring":
        ring = batch * 2 * depth * INDEX_DIM * 2 * 2
        return ring + batch * rows * INDEX_DIM * 2 * 3 + POOL * INDEX_DIM * 4
    keys = batch * candidates * INDEX_DIM * 2
    rows_in = batch * rows * (INDEX_HEADS * INDEX_DIM * 2 + INDEX_HEADS * 4 + INDEX_DIM * 2)
    return keys + rows_in + batch * rows * candidates * 4


def indexer_case(batch, args) -> dict:
    torch._dynamo.reset()
    chain = args.kernel_chain
    candidates = 2048
    out = {"table": "indexer", "batch": batch, "rows": ROWS, "calls_chained": chain,
           "candidates": candidates, "ring_depth": DEEP, "stages": {}}
    # ---- ring -----------------------------------------------------------------------
    timing = {}
    for variant, rows, depth in (("rows", ROWS, DEEP), ("one", 1, DEEP), ("one_batch", 1, POOL)):
        sets = [indexer_operands(batch, rows, depth, candidates, seed=7 + i) for i in range(chain)]
        inputs = tuple(t.to(DEVICE) for one in sets for t in one[:7])
        step = DB.dsa_decode_ring_step if variant == "one_batch" else TU.dsa_decode_ring_rows

        def fn(*flat, step=step):
            outs = []
            for index in range(chain):
                tail, _bank, slots, key, score, ape, position = flat[7 * index:7 * index + 7]
                pooled, rings = step(tail, slots, key, score, ape, position)
                outs.append(pooled)
                outs.append(rings.reshape(-1, INDEX_DIM))
            return torch.cat(outs, 0)
        graph = compiled(fn)
        got, seconds = first_call(graph, inputs)
        if not torch.isfinite(got.float()).all():
            raise AssertionError(f"ring {variant} B={batch}: nonfinite")
        timing[variant] = measure(graph, inputs, chain, args.warmup, args.iterations)
        timing[variant]["first_call_s"] = seconds
        del graph
    out["stages"]["ring"] = {
        "unit": "one ring launch for all `batch` requests",
        **verdict(timing["rows"], timing["one"], indexer_bytes(batch, ROWS, DEEP, candidates, "ring")),
        "one_batch_depth4": timing["one_batch"],
        "rows_over_one_batch_depth4_times_rows": timing["rows"]["median_us"] / (
            timing["one_batch"]["median_us"] * ROWS),
    }
    # ---- scores ---------------------------------------------------------------------
    timing = {}
    for variant, rows in (("rows", ROWS), ("one", 1)):
        sets = [indexer_operands(batch, rows, DEEP, candidates, seed=17 + i) for i in range(chain)]
        picked = [(one[1], one[2], one[6], one[7], one[8], one[9]) for one in sets]
        inputs = tuple(t.to(DEVICE) for one in picked for t in one)

        def fn(*flat, rows=rows):
            outs = []
            for index in range(chain):
                bank, slots, position, query, weights, pooled = flat[6 * index:6 * index + 6]
                if rows == 1:
                    outs.append(DB.dsa_decode_scores(
                        query, weights, bank, slots, position + 1, position, pooled,
                        candidates=candidates, pool_size=POOL))
                else:
                    outs.append(TR.dsa_decode_scores_rows(
                        query, weights, bank, slots, position, pooled,
                        candidates=candidates, pool_size=POOL))
            return torch.cat(outs, 0)
        graph = compiled(fn)
        got, seconds = first_call(graph, inputs)
        if not torch.isfinite(got.float()).all():
            raise AssertionError(f"scores {variant} B={batch}: nonfinite")
        timing[variant] = measure(graph, inputs, chain, args.warmup, args.iterations)
        timing[variant]["first_call_s"] = seconds
        del graph
    out["stages"]["scores"] = {
        "unit": "one score launch for all `batch` requests",
        "programs": TR._programs(batch),
        **verdict(timing["rows"], timing["one"],
                  indexer_bytes(batch, ROWS, DEEP, candidates, "scores")),
    }
    return out


# --- layer ---------------------------------------------------------------------------

def layer_operands(cfg, batch, rows, pages, context, max_seq_len, depth, seed):
    """One step of ``batch`` requests x ``rows`` rows ending at ``context - 1``, on the
    runner's layout: a paged latent bank, a pooled store and a ring per state slot."""
    gen = torch.Generator().manual_seed(seed)
    pool, head_dim = int(cfg.index_kpool), int(cfg.index_head_dim)
    latent = int(cfg.kv_lora_rank)
    start = context - rows
    bank_pages = pages * batch + 3
    order = torch.randperm(bank_pages, generator=gen).to(torch.int32)
    table = order[:pages * batch].reshape(batch, pages)
    used = -(-context // case.PAGE)
    table[:, used:] = -1
    positions = (torch.full((batch, 1), start, dtype=torch.int64)
                 + torch.arange(rows)[None, :]).reshape(-1)
    request = torch.arange(batch).repeat_interleave(rows)
    latent_slots = (table[request, positions // case.PAGE].to(torch.int64) * case.PAGE
                    + positions % case.PAGE)
    slots_total = batch + 3
    candidates_max = int(max_seq_len) // pool
    pool_rows = -(-(candidates_max + 1) // case.PAGE) * case.PAGE
    return {
        "hidden": (torch.randn(batch * rows, int(cfg.hidden_size), generator=gen) * 0.5
                   ).to(torch.bfloat16),
        "latent_cache": (torch.randn(bank_pages * case.PAGE, 1, latent, generator=gen) * 0.5
                         ).to(torch.bfloat16),
        "pool_cache": (torch.randn(slots_total, pool_rows, head_dim, generator=gen) * 0.5
                       ).to(torch.bfloat16),
        "tail": (torch.randn(slots_total, 2, depth, head_dim, generator=gen) * 0.5
                 ).to(torch.bfloat16),
        "state_slots": torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32),
        # One length per row of the step: ``position[b] + t + 1`` (the translator's layout).
        "seq_lens": (positions + 1).to(torch.int32),
        "start_position": torch.full((batch,), start, dtype=torch.int64),
        "position": torch.full((batch,), start, dtype=torch.int64),
        "block_table_row": table.t().contiguous(),
        "latent_slots": latent_slots,
    }


LAYER_NAMES = ("latent_cache", "pool_cache", "seq_lens", "start_position", "block_table_row",
               "latent_slots", "tail", "position", "state_slots")


def layer_bytes(module, batch, rows, pages, max_seq_len, depth) -> int:
    """Bytes one layer step moves: every prepared weight once, each request's latent
    window (the dense form reads every table entry's page; the selected form gathers
    ``TOPK`` rows per query row), each request's pooled store and ring, and the rows'
    activations in and out."""
    weights = sum(p.numel() * p.element_size() for p in module.parameters())
    weights += sum(b.numel() * b.element_size() for b in module.buffers())
    hidden = int(module.hidden_size)
    selects = int(max_seq_len) // POOL > module.indexer.select_k()
    if selects:
        window = batch * rows * TOPK * LATENT * 2
    else:
        window = batch * pages * case.PAGE * LATENT * 2
    store = batch * ((int(max_seq_len) // POOL + 1) * INDEX_DIM * 2 + 2 * depth * INDEX_DIM * 2 * 2)
    return weights + window + store + batch * rows * hidden * 2 * 2


def layer_case(label, context, pages, max_seq_len, batch, args) -> dict:
    torch._dynamo.reset()
    cfg = case.decode_config()
    statics = {"softmax_scale": float(int(cfg.qk_nope_head_dim)
                                      + int(cfg.qk_rope_head_dim)) ** -0.5,
               "max_seq_len": max_seq_len, "page_size": case.PAGE, "collector": None}
    layers = [case.build_attention(model_fp8, cfg, seed=4242 + 31 * index, device=DEVICE)
              for index in range(args.layers)]

    def step_graph(rows):
        def step(*flat):
            hidden = flat[0]
            rest = flat[1:]
            at = 0
            for layer in layers:
                ops = dict(zip(LAYER_NAMES, rest[at:at + len(LAYER_NAMES)]))
                at += len(LAYER_NAMES)
                hidden = layer._forward_requests(hidden, **ops, **statics)
            return hidden
        return step

    timing, outputs = {}, {}
    for variant, rows in (("rows", ROWS), ("one", 1)):
        rest, hidden = [], None
        for index in range(args.layers):
            ops = layer_operands(cfg, batch, rows, pages, context, max_seq_len, DEEP,
                                 seed=99 + 7 * index)
            if index == 0:
                hidden = ops["hidden"].to(DEVICE)
            rest.extend(ops[name].to(DEVICE) for name in LAYER_NAMES)
        inputs = (hidden, *rest)
        graph = compiled(step_graph(rows))
        outputs[variant], seconds = first_call(graph, inputs)
        if not torch.isfinite(outputs[variant].float()).all():
            raise AssertionError(f"{label} B={batch} {variant}: nonfinite output")
        timing[variant] = measure(graph, inputs, args.layers, args.warmup, args.iterations)
        timing[variant]["first_call_s"] = seconds
        del graph
    step_bytes = layer_bytes(layers[0], batch, ROWS, pages, max_seq_len, DEEP)
    del layers
    return {
        "table": "layer",
        "case": label,
        "ctx": context,
        "window_rows": pages * case.PAGE,
        "max_seq_len": max_seq_len,
        "batch": batch,
        "rows": ROWS,
        "ring_depth": DEEP,
        "layers_chained": args.layers,
        "unit": "one layer's decode step for all `batch` requests",
        **verdict(timing["rows"], timing["one"], step_bytes),
    }


# --- main ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables", nargs="+", default=["attention"],
                        choices=["attention", "indexer", "layer"],
                        help="default: the attention table alone; each table is one "
                             "leased job")
    parser.add_argument("--layer-cases", nargs="+",
                        default=[label for label, *_ in LAYER_CASES],
                        choices=[label for label, *_ in LAYER_CASES])
    parser.add_argument("--attention-cases", nargs="+",
                        default=[label for label, *_ in ATTENTION_CASES],
                        choices=[label for label, *_ in ATTENTION_CASES])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kernel-chain", type=int, default=4)
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
    output = args.output.resolve()
    base = load_baseline(BASELINE_DIR) if "attention" in args.tables else None
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
        "baseline": None if base is None else {"commit": base.commit,
                                               "directory": base.directory},
        "rows": ROWS,
        "hbm_bytes_per_s": HBM_BYTES_PER_S,
        "entitlement_ratio_limit": ENTITLEMENT_RATIO_LIMIT,
        "args": vars(args) | {"output": str(output)},
        "synchronization": "graph output copied to CPU on every timed call",
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        brief = {k: row[k] for k in ("table", "case", "batch") if k in row}
        if "rows_over_one_times_rows" in row:
            brief.update(rows_us=row["rows"]["median_us"], one_us=row["one"]["median_us"],
                         rows_over_one_x4=row["rows_over_one_times_rows"],
                         over_entitlement=row["entitlement"]["measured_over_entitlement"])
        else:
            for stage, got in row["stages"].items():
                brief[stage] = {"rows_us": got["rows"]["median_us"],
                                "one_us": got["one"]["median_us"],
                                "rows_over_one_x4": got["rows_over_one_times_rows"],
                                "over_entitlement": got["entitlement"]["measured_over_entitlement"]}
        print(json.dumps(brief), flush=True)

    for batch in args.batch:
        if "attention" in args.tables:
            for label, context, pages in ATTENTION_CASES:
                if label in args.attention_cases:
                    record(attention_case(label, context, pages, batch, base, args))
        if "indexer" in args.tables:
            record(indexer_case(batch, args))
        if "layer" in args.tables:
            for label, context, pages, max_seq_len in LAYER_CASES:
                if label in args.layer_cases:
                    record(layer_case(label, context, pages, max_seq_len, batch, args))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
