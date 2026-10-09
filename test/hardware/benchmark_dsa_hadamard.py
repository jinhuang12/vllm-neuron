# SPDX-License-Identifier: Apache-2.0
"""The DSA Hadamard kernels on Neuron: commit f3a833f's against this tree's, and any revision's.

Set the Neuron cores before launching (``devlease.py slice`` does, with
``NEURON_LOGICAL_NC_CONFIG=2``). This script does not select cores. References are
computed on the CPU; compilation and warmup are excluded from the timings.

Five variants per kernel and shape, timed interleaved in one process:

* ``before``: commit f3a833f's kernel, read by ``git show`` against a pinned blob id and
  launched as f3a833f's wrapper launched it (one program, flattened operands).
* ``after``: this tree's public entry point, ``dsa_hadamard128`` or ``dsa_kpool_hadamard``.
* ``after_aa``: the same call compiled into separate graphs, an A/A pair whose
  difference from ``after`` is the noise floor.
* ``floor_copy``: the kernel's own loads and stores at its own tiling, with no compute.
* ``floor_launch``: one row in and out per program, the fixed cost of an NKI kernel
  inside a graph (for the pooling, plus the chain's link; see below).
* ``NAME`` for each ``--variant NAME=REV``: that revision's public entry point, from its own
  ``kpool_hadamard.py`` read by ``git show`` (:func:`load_revision`), launched as that
  revision launches it. Each revision's entry must reach its NKI kernel, which the run
  checks on its dispatch counters.

The rotation runs at the served prefill row counts (``--rows``: ``tokens * 32`` heads at
1k and 2k tokens) and the decode row counts (``--decode-rows``: ``batch * 32`` at batch
1, 4, 16 and 64); the fused pooling at the served pool counts (``--pools``, one pool per
token). Per-call device time is the slope ``(T_L - T_1) / (L - 1)`` between an
``L``-call graph and a one-call graph (``--links``), paired per iteration, from the
runtime system trace (LNC2 physical-core intervals merged). Both ``L``-call graphs are
dependency chains, so no call can overlap the next and the slope is a call's latency, as
in the model, where each call waits for the op before it. The rotation chains each call
on the previous output. The pooling's output is not its input's shape, so its ``L``
calls take distinct keys and scores and chain through ``ape``: the next call's bias is
the previous output's first ``pool_size`` rows, cast to fp32 (one small slice-and-cast
op per link, which the pooling's ``floor_launch`` chain carries as well). Each graph
runs ``--iterations`` timed executions, rotated across graphs, in each of ``--reps``
repetitions; a variant's figure is the median of its per-call samples over every
repetition, reported with the range of its per-repetition medians.

The decode kernels that call this module's butterfly, ``dsa_decode_tail_update_at``
(one request) and ``dsa_decode_ring_step`` (``--batches`` requests), run from f3a833f's
snapshot, which imports f3a833f's butterfly, from this tree, and from each ``--variant``
revision (its own three modules). Each revision's ring step also runs on this tree's
``kpool_hadamard`` and ``decode_tail_update`` (``NAME_on_after``: the revision's
``decode_batch.py`` alone), the pairing a stack of the two would ship. Their one-call
graphs are compared bit for bit against f3a833f's and timed.

Numerics run first, on the one-call graphs at ``--seeds``: ``after`` against ``before``
(``torch.equal``, the mismatch count and the largest distance in ulps) and both against
the CPU torch reference and an fp64 evaluation of the same function (the largest error,
in absolute terms and in ulps of the output dtype, and the count of elements more than
half an ulp off), and both against the error bound derived in
``test/vllm_neuron/functional/dsa/test_kpool_hadamard_error_bound.py``, the pooling's with
the Scalar Engine errors this run measures; every ``--variant`` is read the same way
(``NAME_*`` keys) and against ``after``. The rotation's fp32 route (``--fp32-rows``) is
read for ``before`` and ``after``, plus its involution. The tensors are saved under ``--pt-dir``. Before
all of it, the engine facts the kernels rely on are read on the device: Tensor Engine
transposes return their input bit for bit, and the Scalar Engine's reciprocal and ``exp``
are read at every fp32 of the ranges the error bound takes them over.

Every run compiles from scratch: ``--cache-root`` must name a new or empty directory,
which becomes ``NEURON_LIBTORCH_CACHE_ROOT``, and the compile cache is disabled on top.
The NKI kernel cache keys on a kernel's own source text, so a warm cache would serve an
old binary for a kernel whose helpers changed, and one binary for both sides of the
decode kernels, whose own source this change does not touch.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

#: This file's repository. Run as a script, Python puts ``test/hardware`` on
#: ``sys.path`` and the venv resolves ``vllm_neuron`` to another checkout; the
#: "after" side must be this tree.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

from vllm_neuron.functional.dsa import decode_batch as decode_batch_after  # noqa: E402
from vllm_neuron.functional.dsa import decode_tail_update as decode_tail_after  # noqa: E402
from vllm_neuron.functional.dsa import kpool_hadamard as kh  # noqa: E402
from test.vllm_neuron.functional.dsa import test_kpool_hadamard_error_bound as bound  # noqa: E402

DEVICE = "neuron:0"
HEAD_DIM = kh.INDEX_HEAD_DIM
POOL_SIZE = kh.DEFAULT_POOL_SIZE
#: One device chunk of a Scalar Engine sweep: ``SWEEP_ROWS`` partitions' worth of rows,
#: ``SWEEP_WIDTH`` fp32 each.
SWEEP_ROWS, SWEEP_WIDTH = 2048, 8192

BASELINE_COMMIT = "f3a833f"
#: module name -> (repository path, git blob id at :data:`BASELINE_COMMIT`).
BASELINE_SOURCES = {
    "kpool_hadamard": ("vllm_neuron/functional/dsa/kpool_hadamard.py",
                       "dc13e33ae7de774ec42b0a6bd401677ed7d7797e"),
    "decode_tail_update": ("vllm_neuron/functional/dsa/decode_tail_update.py",
                           "b1c07db22c9d5c61c5fdeb4d61712f5708137c97"),
    "decode_batch": ("vllm_neuron/functional/dsa/decode_batch.py",
                     "8b834c0bbd910e0783d84d261a745596c0dac49b"),
}
BASELINE_PACKAGE = "dsa_hadamard_f3a833f"

#: The package the three modules live in.
DSA_PACKAGE = "vllm_neuron.functional.dsa"

#: Variant names this script defines itself; a ``--variant`` may not take one.
FIXED_VARIANTS = ("before", "after", "after_aa", "floor_copy", "floor_launch")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def _rewrite_imports(text: str, leaves, package: str) -> str:
    """``text`` with its imports of the :data:`DSA_PACKAGE` modules in ``leaves`` pointed at
    ``package``. Refuses a source that would still import one of them from the live tree."""
    for leaf in leaves:
        text = text.replace(f"{DSA_PACKAGE}.{leaf}", f"{package}.{leaf}")
        text = re.sub(rf"^(\s*)from {re.escape(DSA_PACKAGE)} import {leaf}\b",
                      rf"\1from {package} import {leaf}", text, flags=re.MULTILINE)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module:
            named = [f"{node.module}.{alias.name}" for alias in node.names] + [node.module]
        elif isinstance(node, ast.Import):
            named = [alias.name for alias in node.names]
        else:
            continue
        live = [n for n in named for leaf in leaves
                if n == f"{DSA_PACKAGE}.{leaf}" or n.startswith(f"{DSA_PACKAGE}.{leaf}.")]
        if live:
            raise RuntimeError(f"an import of {live} was not redirected to {package}")
    return text


def load_revision(rev: str, where: Path, package: str, paths: dict[str, str],
                  pins: dict[str, str] | None = None) -> types.SimpleNamespace:
    """``rev``'s modules ``paths`` (module name -> repository path), written under
    ``where/package``, each importing the snapshot's copies of the others.

    Imports of any other module resolve to this tree. With ``pins`` (module name -> git
    blob id) every file is checked against its pinned blob, so a silently different
    baseline is impossible; without, the blob ids found are recorded.
    """
    directory = where / package
    directory.mkdir()
    (directory / "__init__.py").write_text(f'"""Revision {rev}, read by git show."""\n')
    blobs = {}
    for name, path in paths.items():
        blobs[name] = _git("rev-parse", f"{rev}:{path}").strip()
        if pins is not None and blobs[name] != pins[name]:
            raise RuntimeError(f"{rev}:{path} is blob {blobs[name]}, pinned {pins[name]}")
        text = _rewrite_imports(_git("show", f"{rev}:{path}"), paths, package)
        (directory / f"{name}.py").write_text(text)
    if str(where) not in sys.path:
        sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{package}.{name}") for name in paths}
    return types.SimpleNamespace(rev=rev, commit=_git("rev-parse", f"{rev}^{{commit}}").strip(),
                                 directory=str(directory), blobs=blobs, **modules)


def load_baseline(where: Path) -> types.SimpleNamespace:
    """f3a833f's three modules, each importing the snapshot's own copies of the others,
    checked against their pinned blob ids. The decode modules' kernels therefore trace
    f3a833f's butterfly."""
    return load_revision(BASELINE_COMMIT, where, BASELINE_PACKAGE,
                         {name: path for name, (path, _) in BASELINE_SOURCES.items()},
                         {name: blob for name, (_, blob) in BASELINE_SOURCES.items()})


def load_variants(specs: list[str], where: Path) -> tuple[dict, dict]:
    """Each ``NAME=REV`` of ``--variant``: ``REV``'s three modules (``variants[NAME]``), and
    ``REV``'s ``decode_batch`` alone on this tree's other two (``on_after[NAME_on_after]``)."""
    variants, on_after = {}, {}
    paths = {name: path for name, (path, _) in BASELINE_SOURCES.items()}
    for spec in specs:
        name, sep, rev = spec.partition("=")
        if not sep or not name.isidentifier() or not rev:
            raise ValueError(f"--variant takes NAME=REV; got {spec!r}")
        if name in FIXED_VARIANTS or name in variants:
            raise ValueError(f"--variant name {name!r} is taken")
        variants[name] = load_revision(rev, where, f"dsa_hadamard_{name}", paths)
        on_after[f"{name}_on_after"] = load_revision(
            rev, where, f"dsa_hadamard_{name}_on_after", {"decode_batch": paths["decode_batch"]})
    return variants, on_after


def check_routes(modules: dict, entry_kernel: str) -> dict:
    """Each ``kpool_hadamard`` module of ``modules`` (variant name -> module) dispatched its
    ``entry_kernel`` and never its torch path since its counters were last reset."""
    routes = {}
    for name, module in modules.items():
        nki_count, fallback = module.kpool_hadamard_dispatch_counters()
        identity = module.kpool_hadamard_kernel_identity()
        routes[name] = {"nki_dispatch": nki_count, "torch_fallback": fallback,
                        "kernel": list(identity) if identity else None}
        if nki_count == 0 or fallback or identity != (module.__name__, entry_kernel):
            raise RuntimeError(f"{name} did not take {module.__name__}.{entry_kernel}: "
                               f"{routes[name]}")
    return routes


# ---------------------------------------------------------------------------------------------
# Floor kernels
# ---------------------------------------------------------------------------------------------


@nki.jit
def rotation_copy_floor(x_hbm, n_rows):
    """The rotation's loads and stores at its own blocks, no compute: ``out = x``."""
    out = nl.ndarray((n_rows, x_hbm.shape[1]), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    first, owned = kh._program_share(n_rows)
    for block in kh._row_blocks(first, owned, nl.tile_size.pmax, kh._CHUNK_TILES):
        rows_in, _ = kh._load_block(x_hbm, block)
        nisa.dma_copy(dst=kh._block_rows(out, block, (x_hbm.shape[1],)), src=rows_in)
    return out


@nki.jit
def pool_copy_floor(slot_k_hbm, slot_score_hbm, ape_hbm, n_pools, pool_size):
    """The pooling's loads and stores at its own blocks, no compute: ``out = slot 0's key``.

    ``ape`` is loaded as the kernel loads it, so the chain through it is a real dependency.
    """
    head_dim = slot_k_hbm.shape[1]
    out = nl.ndarray((n_pools, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.shared_hbm)
    first, owned = kh._program_share(n_pools)
    for slot in range(pool_size):
        kh._broadcast_row(ape_hbm, nl.tile_size.pmax, head_dim, slot)
    unit = (pool_size, head_dim)
    for block in kh._row_blocks(first, owned, nl.tile_size.pmax, kh._POOL_TILE_COLUMNS):
        shape = (block[1], block[2], pool_size, head_dim)
        score = nl.ndarray(shape, dtype=slot_score_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=score, src=kh._block_rows(slot_score_hbm, block, unit))
        key = nl.ndarray(shape, dtype=slot_k_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=key, src=kh._block_rows(slot_k_hbm, block, unit))
        nisa.dma_copy(dst=kh._block_rows(out, block, (head_dim,)), src=key[:, :, 0, :])
    return out


@nki.jit
def pool_launch_floor(slot_k_hbm, slot_score_hbm, ape_hbm, n_pools, pool_size):
    """The pooling's fixed cost: one pool of each program's run in and out. ``ape`` is an
    operand the kernel does not read; as an operand it still orders the call in the chain."""
    head_dim = slot_k_hbm.shape[1]
    out = nl.ndarray((n_pools, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.shared_hbm)
    first, _ = kh._program_share(n_pools)
    row = nl.ndarray((1, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=slot_k_hbm[first * pool_size:first * pool_size + 1, :])
    nisa.dma_copy(dst=out[first:first + 1, :], src=row)
    return out


@nki.jit
def launch_floor(x_hbm, n_rows):
    """An NKI kernel's fixed cost: one row of each program's run in and out."""
    out = nl.ndarray((n_rows, x_hbm.shape[1]), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    first, _ = kh._program_share(n_rows)
    row = nl.ndarray((1, x_hbm.shape[1]), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=x_hbm[first:first + 1, :])
    nisa.dma_copy(dst=out[first:first + 1, :], src=row)
    return out


@nki.jit
def transpose_check(x_hbm, scalar_copy):
    """``x_hbm[128 * tiles, head_dim]`` through the kernels' load and ``_transpose_group``.

    Returns the ``[head_dim, tiles, 128]`` SBUF tile as it leaves PSUM, copied out by the
    Scalar Engine (the pooling's choice) when ``scalar_copy`` is set, else by the Vector
    Engine (the rotation's).
    """
    pmax = nl.tile_size.pmax
    tiles = x_hbm.shape[0] // pmax
    head_dim = x_hbm.shape[1]
    block = (0, pmax, tiles)
    group = kh._tile_groups([block], tiles)[0]
    rows_in, _ = kh._load_block(x_hbm, block)
    engine = nisa.vector_engine
    if scalar_copy:
        engine = nisa.scalar_engine
    transposed = kh._transpose_group(rows_in, block, group, engine)
    out = nl.ndarray((head_dim, tiles, pmax), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=transposed)
    return out


@nki.jit
def scalar_engine_check(x_hbm, use_exp):
    """``exp(x)`` when ``use_exp`` is set, else ``1 / x``, for every element of
    ``x_hbm[rows, width]`` fp32, on the Scalar Engine as the pooling computes them."""
    pmax = nl.tile_size.pmax
    out = nl.ndarray(x_hbm.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    for tile in range(x_hbm.shape[0] // pmax):
        x = nl.ndarray((pmax, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_hbm[tile * pmax:(tile + 1) * pmax, :])
        y = nl.ndarray((pmax, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
        if use_exp:
            nisa.activation(dst=y, op=nl.exp, data=x)
        else:
            nisa.activation(dst=y, op=nl.reciprocal, data=x)
        nisa.dma_copy(dst=out[tile * pmax:(tile + 1) * pmax, :], src=y)
    return out


# ---------------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------------


def rotation_variants(base, revisions):
    """The rotation's variants; ``revisions`` maps a ``--variant`` name to its modules."""
    def before(x):
        return wrap_nki(base.kpool_hadamard._hadamard128_nki)(x, int(x.shape[0]))

    def after(x):
        return kh.dsa_hadamard128(x)

    def after_aa(x):
        return kh.dsa_hadamard128(x)

    def floor_copy(x):
        return wrap_nki(rotation_copy_floor)[kh._programs(int(x.shape[0]))](x, int(x.shape[0]))

    def floor_launch(x):
        return wrap_nki(launch_floor)[kh._programs(int(x.shape[0]))](x, int(x.shape[0]))

    variants = {"before": before, "after": after, "after_aa": after_aa,
                "floor_copy": floor_copy, "floor_launch": floor_launch}
    for name, revision in revisions.items():
        variants[name] = revision.kpool_hadamard.dsa_hadamard128
    return variants


def pool_variants(base, revisions):
    """The pooling's variants; ``revisions`` maps a ``--variant`` name to its modules."""
    def before(slot_k, slot_score, ape):
        n_pools, pool_size, head_dim = (int(d) for d in slot_k.shape)
        return wrap_nki(base.kpool_hadamard._kpool_hadamard_nki)(
            slot_k.reshape(n_pools * pool_size, head_dim), slot_score.reshape(
                n_pools * pool_size, head_dim), ape, n_pools, pool_size)

    def after(slot_k, slot_score, ape):
        return kh.dsa_kpool_hadamard(slot_k, slot_score, ape)

    def after_aa(slot_k, slot_score, ape):
        return kh.dsa_kpool_hadamard(slot_k, slot_score, ape)

    def floor_copy(slot_k, slot_score, ape):
        n_pools, pool_size, head_dim = (int(d) for d in slot_k.shape)
        return wrap_nki(pool_copy_floor)[kh._programs(n_pools)](
            slot_k.reshape(n_pools * pool_size, head_dim),
            slot_score.reshape(n_pools * pool_size, head_dim), ape, n_pools, pool_size)

    def floor_launch(slot_k, slot_score, ape):
        n_pools, pool_size, head_dim = (int(d) for d in slot_k.shape)
        return wrap_nki(pool_launch_floor)[kh._programs(n_pools)](
            slot_k.reshape(n_pools * pool_size, head_dim),
            slot_score.reshape(n_pools * pool_size, head_dim), ape, n_pools, pool_size)

    variants = {"before": before, "after": after, "after_aa": after_aa,
                "floor_copy": floor_copy, "floor_launch": floor_launch}
    for name, revision in revisions.items():
        variants[name] = revision.kpool_hadamard.dsa_kpool_hadamard
    return variants


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)


def chain_graph(call, links: int):
    def chain(x):
        for _ in range(links):
            x = call(x)
        return x
    return compile_fn(chain)


def pool_chain_graph(call, links: int):
    """``links`` pooling calls on distinct keys and scores, each call's bias the previous
    output's first ``pool_size`` rows: a dependency chain, as :func:`chain_graph` is."""
    def chain(ape, *operands):
        for i in range(links):
            out = call(operands[2 * i], operands[2 * i + 1], ape)
            ape = out[:POOL_SIZE].float()
        return out
    return compile_fn(chain)


def stats_us(samples: list[float]) -> dict:
    ordered = sorted(samples)

    def quantile(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"n": len(ordered), "median_us": statistics.median(ordered), "min_us": ordered[0],
            "p10_us": quantile(0.1), "p90_us": quantile(0.9), "max_us": ordered[-1]}


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, in execution order, LNC2 core intervals merged."""
    starts = {}
    intervals: dict[int, list[tuple[int, int]]] = {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            intervals.setdefault(start["data"]["exec_id"], []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in intervals[e]) - min(b for b, _ in intervals[e])) / 1000.0
            for e in sorted(intervals)]


def _first(result):
    while isinstance(result, tuple):
        result = result[0]
    return result


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Every graph's device samples in us, the calls rotated across graphs per iteration:
    one repetition."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(warmup):
        for name in names:
            _first(graphs[name](*inputs[name])).to("cpu")
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                _first(graphs[name](*inputs[name])).to("cpu")
                order.append(name)
        events_json = trace.fetch_events_json()
    device_all = device_intervals(events_json)
    if len(device_all) != len(order):
        raise AssertionError(f"system trace has {len(device_all)} executions for {len(order)} calls")
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return device


def time_reps(graphs: dict, inputs: dict, args) -> list[dict]:
    """``args.reps`` repetitions of :func:`time_graphs`."""
    return [time_graphs(graphs, inputs, args.warmup, args.iterations) for _ in range(args.reps)]


def rep_stats(reps: list[list[float]]) -> dict:
    """:func:`stats_us` of every repetition's samples together, plus each repetition's
    median and their range."""
    medians = [statistics.median(samples) for samples in reps]
    return {**stats_us([v for samples in reps for v in samples]), "rep_medians_us": medians,
            "rep_range_us": [min(medians), max(medians)]}


def slopes(reps: list[dict], variants, links: int) -> tuple[dict, dict]:
    """Per variant, the one-call graph's and the per-call slope's statistics over every
    repetition, and the noise floor: ``|after - after_aa|`` per paired sample."""
    out = {}
    per_call = {}
    for name in variants:
        one = [device[f"{name}|1"] for device in reps]
        many = [device[f"{name}|L"] for device in reps]
        per_call[name] = [[(m - o) / (links - 1) for m, o in zip(m_rep, o_rep)]
                          for m_rep, o_rep in zip(many, one)]
        out[name] = {"one_call_graph": rep_stats(one), "per_call": rep_stats(per_call[name]),
                     "samples": {"one": [v for r in one for v in r],
                                 "many": [v for r in many for v in r]}}
    noise = [abs(a - b) for a_rep, b_rep in zip(per_call["after"], per_call["after_aa"])
             for a, b in zip(a_rep, b_rep)]
    return out, stats_us(noise)


# ---------------------------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------------------------


def ulps(out: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """``|out - exact|`` in units in the last place of ``out``'s dtype at ``exact``.

    A result rounded once from a value within half an ulp is within one ulp of the exact
    value, wherever that value falls between two representable numbers, so ``< 1`` is the
    bound every faithful kernel meets.
    """
    return (out.double() - exact).abs() / bound.ulp(exact, out.dtype)


def compare(after: torch.Tensor, before: torch.Tensor, reference: torch.Tensor,
            exact: torch.Tensor, error_bound: torch.Tensor, side: str = "after") -> dict:
    """``after`` against ``before``, and both against the CPU ``reference``, the fp64 ``exact``
    value and the derived ``error_bound`` of ``test_kpool_hadamard_error_bound``. ``side``
    names ``after`` in the keys (a ``--variant`` name for a variant's output)."""
    after_ulps, before_ulps = ulps(after, exact), ulps(before, exact)
    after_ratio = bound.error_over_bound(after, exact, error_bound)
    before_ratio = bound.error_over_bound(before, exact, error_bound)
    return {
        f"{side}_equals_before": bool(torch.equal(after, before)),
        f"{side}_before_mismatches": int((after != before).sum()),
        "elements": after.numel(),
        f"{side}_vs_reference_max_abs": float((after.float() - reference.float()).abs().max()),
        "before_vs_reference_max_abs": float((before.float() - reference.float()).abs().max()),
        f"{side}_vs_exact_max_abs": float((after.double() - exact).abs().max()),
        "before_vs_exact_max_abs": float((before.double() - exact).abs().max()),
        f"{side}_before_max_ulps": float(ulps(after, before.double()).max()),
        f"{side}_vs_exact_max_ulps": float(after_ulps.max()),
        "before_vs_exact_max_ulps": float(before_ulps.max()),
        f"{side}_elements_over_half_ulp": int((after_ulps > 0.5).sum()),
        "before_elements_over_half_ulp": int((before_ulps > 0.5).sum()),
        f"{side}_max_error_over_bound": float(after_ratio.max()),
        "before_max_error_over_bound": float(before_ratio.max()),
        f"{side}_within_bound": bool(after_ratio.max() <= 1.0),
        "before_within_bound": bool(before_ratio.max() <= 1.0),
    }


def compare_variants(outputs: dict, before: torch.Tensor, after: torch.Tensor, reference,
                     exact, error_bound) -> dict:
    """:func:`compare` of each variant's output (name -> tensor), and its mismatches with
    ``after``."""
    out = {}
    for name, tensor in outputs.items():
        out.update(compare(tensor, before, reference, exact, error_bound, side=name))
        out[f"{name}_equals_after"] = bool(torch.equal(tensor, after))
        out[f"{name}_after_mismatches"] = int((tensor != after).sum())
    return out


def rows_input(n_rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n_rows, HEAD_DIM, generator=gen).to(torch.bfloat16)


def pool_input(n_pools: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    slot_k = torch.randn(n_pools, POOL_SIZE, HEAD_DIM, generator=gen).to(torch.bfloat16)
    slot_score = torch.randn(n_pools, POOL_SIZE, HEAD_DIM, generator=gen).to(torch.bfloat16)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=gen)
    return slot_k, slot_score, ape


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def reset_routes(revisions: dict) -> None:
    """Zero the dispatch counters of this tree's ``kpool_hadamard`` and every variant's."""
    kh.reset_kpool_hadamard_dispatch_counters()
    for revision in revisions.values():
        revision.kpool_hadamard.reset_kpool_hadamard_dispatch_counters()


def entry_modules(revisions: dict) -> dict:
    """Variant name -> the ``kpool_hadamard`` module its public entry point lives in."""
    return {"after": kh, **{name: r.kpool_hadamard for name, r in revisions.items()}}


def rotation_case(n_rows: int, args, base, revisions: dict, fixed_variants) -> dict:
    variants = rotation_variants(base, revisions)
    timed_variants = (*fixed_variants, *revisions)
    x = rows_input(n_rows, seed=n_rows).to(DEVICE)
    graphs, inputs = {}, {}
    for name in timed_variants:
        graphs[f"{name}|1"] = chain_graph(variants[name], 1)
        graphs[f"{name}|L"] = chain_graph(variants[name], args.links)
        inputs[f"{name}|1"] = inputs[f"{name}|L"] = (x,)
    reset_routes(revisions)
    for key, graph in graphs.items():
        started = time.time()
        graph(*inputs[key]).to("cpu")
        print(f"compiled rotation {key} rows={n_rows} in {time.time() - started:.1f}s", flush=True)
    routes = check_routes(entry_modules(revisions), "_hadamard128_nki")
    numerics = []
    for seed in args.seeds:
        x_cpu = rows_input(n_rows, seed)
        x_dev = x_cpu.to(DEVICE)
        after = graphs["after|1"](x_dev).to("cpu")
        before = graphs["before|1"](x_dev).to("cpu")
        others = {name: graphs[f"{name}|1"](x_dev).to("cpu") for name in revisions}
        reference = kh._dsa_hadamard128_torch(x_cpu)
        exact, error_bound = bound.exact_rotation(x_cpu), bound.rotation_error_bound(x_cpu)
        path = args.pt_dir / f"rotation_rows{n_rows}_seed{seed}.pt"
        torch.save({"seed": seed, "x": x_cpu, "after": after, "before": before, **others}, path)
        numerics.append({"seed": seed, "pt": str(path),
                         **compare(after, before, reference, exact, error_bound),
                         **compare_variants(others, before, after, reference, exact,
                                            error_bound)})
        print(json.dumps({"rotation_rows": n_rows, **numerics[-1]}), flush=True)
    timing, noise = slopes(time_reps(graphs, inputs, args), timed_variants, args.links)
    return {"kernel": "_hadamard128_nki", "rows": n_rows, "shape": [n_rows, HEAD_DIM],
            "dtype": "bfloat16", "links": args.links, "graph": "chain", "variants": timing,
            "noise_floor_aa_abs_diff": noise, "routes": routes, "numerics": numerics}


def pool_case(n_pools: int, args, base, revisions: dict, engine: dict) -> dict:
    """The pooling at ``n_pools``: numerics against the derived bound, with the engine errors
    ``engine_checks`` measured in this run, then the timings."""
    errors = {"exp_error": engine["scalar_engine_exp"]["max_relative"],
              "reciprocal_error": engine["scalar_engine_reciprocal"]["max_relative"]}
    variants = pool_variants(base, revisions)
    operands = [pool_input(n_pools, seed=n_pools + link) for link in range(args.links)]
    ape = operands[0][2].to(DEVICE)
    flat = [t.to(DEVICE) for slot_k, slot_score, _ in operands for t in (slot_k, slot_score)]
    graphs, inputs = {}, {}
    for name in variants:
        graphs[f"{name}|1"] = pool_chain_graph(variants[name], 1)
        graphs[f"{name}|L"] = pool_chain_graph(variants[name], args.links)
        inputs[f"{name}|1"] = (ape, *flat[:2])
        inputs[f"{name}|L"] = (ape, *flat)
    reset_routes(revisions)
    for key, graph in graphs.items():
        started = time.time()
        _first(graph(*inputs[key])).to("cpu")
        print(f"compiled pooling {key} pools={n_pools} in {time.time() - started:.1f}s", flush=True)
    routes = check_routes(entry_modules(revisions), "_kpool_hadamard_nki")
    numerics = []
    for seed in args.seeds:
        slot_k, slot_score, ape_cpu = pool_input(n_pools, seed)
        dev = (ape_cpu.to(DEVICE), slot_k.to(DEVICE), slot_score.to(DEVICE))
        after = _first(graphs["after|1"](*dev)).to("cpu")
        before = _first(graphs["before|1"](*dev)).to("cpu")
        others = {name: _first(graphs[f"{name}|1"](*dev)).to("cpu") for name in revisions}
        reference = kh._dsa_kpool_hadamard_torch(slot_k, slot_score, ape_cpu)
        path = args.pt_dir / f"pooling_pools{n_pools}_seed{seed}.pt"
        torch.save({"seed": seed, "slot_k": slot_k, "slot_score": slot_score, "ape": ape_cpu,
                    "after": after, "before": before, **others}, path)
        exact = bound.exact_pooling(slot_k, slot_score, ape_cpu)
        error_bound = bound.pooling_error_bound(slot_k, slot_score, ape_cpu, **errors)
        numerics.append({"seed": seed, "pt": str(path),
                         **compare(after, before, reference, exact, error_bound),
                         **compare_variants(others, before, after, reference, exact,
                                            error_bound)})
        print(json.dumps({"pooling_pools": n_pools, **numerics[-1]}), flush=True)
    timing, noise = slopes(time_reps(graphs, inputs, args), variants, args.links)
    return {"kernel": "_kpool_hadamard_nki", "pools": n_pools,
            "shape": [n_pools, POOL_SIZE, HEAD_DIM], "dtype": "bfloat16 slot_k and slot_score",
            "links": args.links, "graph": "chain through ape", "variants": timing,
            "noise_floor_aa_abs_diff": noise, "bound_engine_errors": errors, "routes": routes,
            "numerics": numerics}


def fp32_rotation_case(n_rows: int, args, base) -> dict:
    """The rotation's fp32 route: after against before and the reference, and the involution."""
    variants = rotation_variants(base, {})
    once = {name: chain_graph(variants[name], 1) for name in ("before", "after")}
    twice = chain_graph(variants["after"], 2)
    numerics = []
    for seed in args.seeds:
        gen = torch.Generator().manual_seed(seed)
        x_cpu = torch.randn(n_rows, HEAD_DIM, generator=gen)
        x_dev = x_cpu.to(DEVICE)
        after = once["after"](x_dev).to("cpu")
        before = once["before"](x_dev).to("cpu")
        involution = twice(x_dev).to("cpu")
        reference = kh._dsa_hadamard128_torch(x_cpu)
        path = args.pt_dir / f"rotation_fp32_rows{n_rows}_seed{seed}.pt"
        torch.save({"seed": seed, "x": x_cpu, "after": after, "before": before,
                    "involution": involution}, path)
        numerics.append({"seed": seed, "pt": str(path),
                         **compare(after, before, reference, bound.exact_rotation(x_cpu),
                                   bound.rotation_error_bound(x_cpu)),
                         "after_before_max_abs": float((after - before).abs().max()),
                         "involution_max_abs": float((involution - x_cpu).abs().max())})
        print(json.dumps({"rotation_fp32_rows": n_rows, **numerics[-1]}), flush=True)
    return {"kernel": "_hadamard128_nki", "rows": n_rows, "dtype": "float32",
            "numerics": numerics}


def engine_checks(args) -> dict:
    """The engine facts the kernels' numerics rest on, read on the device.

    Tensor Engine transposes, as ``_transpose_group`` runs them with either copy engine, must
    return their input bit for bit, in bf16 and fp32, over a wide exponent range. The Scalar
    Engine's two functions in the pooling are read at every fp32 they can meet there, against
    the exact value: the reciprocal from 1 to ``POOL_SIZE``, the whole range of a softmax sum,
    and ``exp`` from the error bound's ``EXP_LOW`` to 0, where the max-shifted scores of every
    input the bound admits lie.
    """
    tiles = 4
    gen = torch.Generator().manual_seed(args.seeds[0])
    mantissa = torch.randn(tiles * 128, HEAD_DIM, generator=gen)
    exponent = torch.randint(-100, 100, (tiles * 128, HEAD_DIM), generator=gen)
    wide = torch.ldexp(mantissa, exponent)
    transposes = []
    for dtype in (torch.bfloat16, torch.float32):
        x_cpu = wide.to(dtype)
        expected = x_cpu.reshape(128, tiles, HEAD_DIM).permute(2, 1, 0)
        for scalar_copy in (0, 1):
            graph = compile_fn(
                lambda x, scalar_copy=scalar_copy: wrap_nki(transpose_check)(x, scalar_copy))
            got = _first(graph(x_cpu.to(DEVICE))).to("cpu")
            transposes.append({"dtype": str(dtype), "copy_engine": "scalar" if scalar_copy else "vector",
                               "elements": got.numel(), "equal": bool(torch.equal(got, expected))})
            print(json.dumps({"transpose_check": transposes[-1]}), flush=True)
    reciprocal_graph = compile_fn(lambda x: wrap_nki(scalar_engine_check)(x, 0))
    reciprocal = scalar_engine_sweep(reciprocal_graph, 1.0, float(POOL_SIZE), lambda x: 1.0 / x)
    print(json.dumps({"reciprocal_check": reciprocal}), flush=True)
    exp_graph = compile_fn(lambda x: wrap_nki(scalar_engine_check)(x, 1))
    exp = scalar_engine_sweep(exp_graph, -0.0, bound.EXP_LOW, torch.exp)
    # The max slot's shift is +0.0, which the sweep's -0.0 does not stand for.
    at_zero = _first(exp_graph(torch.zeros(SWEEP_ROWS, SWEEP_WIDTH).to(DEVICE))).to("cpu")
    exp["results_at_positive_zero"] = [float(v) for v in at_zero.unique()]
    print(json.dumps({"exp_check": exp}), flush=True)
    # The CPU tests take these errors as constants; a device that exceeds them needs new ones.
    reciprocal["within_test_constant"] = (
        reciprocal["max_relative"] <= bound.SCALAR_ENGINE_RECIPROCAL_ERROR)
    exp["within_test_constant"] = exp["max_relative"] <= bound.SCALAR_ENGINE_EXP_ERROR
    return {"transposes": transposes, "scalar_engine_reciprocal": reciprocal,
            "scalar_engine_exp": exp}


def scalar_engine_sweep(graph, first: float, last: float, exact) -> dict:
    """``graph``, a compiled ``scalar_engine_check``, at every fp32 from ``first`` to ``last``.

    ``first`` and ``last`` share a sign. The values go through the device
    ``SWEEP_ROWS * SWEEP_WIDTH`` at a time, in bit-pattern order; the last chunk is padded
    with ``last``, and the padding is not counted. ``exact`` maps the fp64 inputs to the exact
    results.
    """
    low = int(torch.tensor(first).view(torch.int32))
    high = int(torch.tensor(last).view(torch.int32))
    chunk = SWEEP_ROWS * SWEEP_WIDTH
    worst_relative, worst_ulps, over_half, largest = 0.0, 0.0, 0, -float("inf")
    worst_input = None
    for start in range(low, high + 1, chunk):
        raw = torch.arange(start, start + chunk, dtype=torch.int64)
        counted = (raw <= high).reshape(SWEEP_ROWS, SWEEP_WIDTH)
        bits = raw.clamp(max=high)
        x_cpu = bits.to(torch.int32).view(torch.float32).reshape(SWEEP_ROWS, SWEEP_WIDTH)
        got = _first(graph(x_cpu.to(DEVICE))).to("cpu")
        want = exact(x_cpu.double())
        relative = ((got.double() - want) / want).abs()
        distance = ulps(got, want)
        if float(relative.max()) > worst_relative:
            worst_relative = float(relative.max())
            worst_input = float(x_cpu.flatten()[int(relative.argmax())])
        worst_ulps = max(worst_ulps, float(distance.max()))
        over_half += int((distance[counted] > 0.5).sum())
        largest = max(largest, float(got.max()))
    return {"values": high - low + 1, "range": [first, last], "max_relative": worst_relative,
            "max_relative_at": worst_input, "max_ulps": worst_ulps,
            "elements_over_half_ulp": over_half, "largest_result": largest}


def tail_input(seed: int):
    gen = torch.Generator().manual_seed(seed)
    tail = torch.randn(2, POOL_SIZE, HEAD_DIM, generator=gen).to(torch.bfloat16)
    key = torch.randn(1, HEAD_DIM, generator=gen).to(torch.bfloat16)
    score = torch.randn(1, HEAD_DIM, generator=gen).to(torch.bfloat16)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=gen)
    # The last slot of a pool: the step that runs the butterfly and whose output is used.
    position = torch.tensor(4 * POOL_SIZE * 1000 + POOL_SIZE - 1, dtype=torch.int64)
    return tail, key, score, ape, position


def ring_input(batch: int, seed: int, slots_total: int):
    gen = torch.Generator().manual_seed(seed)
    bank = torch.randn(slots_total, 2, POOL_SIZE, HEAD_DIM, generator=gen).to(torch.bfloat16)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    key = torch.randn(batch, HEAD_DIM, generator=gen).to(torch.bfloat16)
    score = torch.randn(batch, HEAD_DIM, generator=gen).to(torch.bfloat16)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=gen)
    # Every request ends a pool, so every row of ``pooled`` is the butterfly's output.
    position = torch.full((batch,), POOL_SIZE * 1000 + POOL_SIZE - 1, dtype=torch.int32)
    return bank, slots, key, score, ape, position


def decode_cases(args, base, revisions: dict, on_after: dict) -> list[dict]:
    """f3a833f's decode kernels (with f3a833f's butterfly) against this tree's and each
    variant's, bit for bit; the ring step also from each variant's ``decode_batch`` on this
    tree's other modules (``on_after``)."""
    tails = {"before": base.decode_tail_update.dsa_decode_tail_update_at,
             "after": decode_tail_after.dsa_decode_tail_update_at,
             **{name: r.decode_tail_update.dsa_decode_tail_update_at
                for name, r in revisions.items()}}
    rings = {"before": base.decode_batch.dsa_decode_ring_step,
             "after": decode_batch_after.dsa_decode_ring_step,
             **{name: r.decode_batch.dsa_decode_ring_step
                for name, r in {**revisions, **on_after}.items()}}
    entries = {
        "tail_update": (tails, [("batch1", lambda seed: tail_input(seed))]),
        "ring_step": (rings,
                      [(f"batch{b}", (lambda b: lambda seed: ring_input(b, seed, 2 * b + 3))(b))
                       for b in args.batches]),
    }
    cases = []
    for entry, (calls, shapes) in entries.items():
        for label, make in shapes:
            graphs = {side: compile_fn(fn) for side, fn in calls.items()}
            first = [t.to(DEVICE) for t in make(args.seeds[0])]
            inputs = {side: first for side in calls}
            numerics = []
            for seed in args.seeds:
                cpu = make(seed)
                dev = [t.to(DEVICE) for t in cpu]
                outputs = {side: [t.to("cpu") for t in graph(*dev)]
                           for side, graph in graphs.items()}
                before, after = outputs["before"], outputs["after"]
                path = args.pt_dir / f"decode_{entry}_{label}_seed{seed}.pt"
                torch.save({"seed": seed, "inputs": cpu, **outputs}, path)
                record = {
                    "seed": seed, "pt": str(path),
                    "pooled_equal": bool(torch.equal(after[0], before[0])),
                    "ring_equal": bool(torch.equal(after[1], before[1])),
                    "pooled_mismatches": int((after[0] != before[0]).sum()),
                }
                for side, out in outputs.items():
                    if side in ("before", "after"):
                        continue
                    record[f"{side}_pooled_equal"] = bool(torch.equal(out[0], before[0]))
                    record[f"{side}_ring_equal"] = bool(torch.equal(out[1], before[1]))
                    record[f"{side}_pooled_mismatches"] = int((out[0] != before[0]).sum())
                numerics.append(record)
                print(json.dumps({"decode": entry, "shape": label, **numerics[-1]}), flush=True)
            reps = time_reps(graphs, inputs, args)
            cases.append({"entry": entry, "shape": label, "numerics": numerics,
                          "one_call_graph": {side: rep_stats([device[side] for device in reps])
                                             for side in graphs},
                          "samples": {side: [v for device in reps for v in device[side]]
                                      for side in graphs}})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-dir", type=Path, help="numerics tensors (default: --out's directory)")
    parser.add_argument("--rows", type=int, nargs="*", default=[32768, 65536])
    parser.add_argument("--decode-rows", type=int, nargs="*", default=[32, 128, 512, 2048])
    parser.add_argument("--pools", type=int, nargs="*", default=[1024, 2048])
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 4, 16, 64])
    parser.add_argument("--fp32-rows", type=int, nargs="*", default=[37, 4874, 32768],
                        help="fp32 rotation numerics (4874 reaches every tiling branch on 2 programs)")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5,
                        help="repetitions of the interleaved timing, each of --iterations")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=REV",
                        help="also time and check REV's kernels as variant NAME")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="a new or empty directory: NEURON_LIBTORCH_CACHE_ROOT for this run")
    parser.add_argument("--time-limit", type=int, default=7200,
                        help="seconds before the run aborts")
    args = parser.parse_args()
    signal.alarm(args.time_limit)
    args.cache_root = args.cache_root.resolve()
    if args.cache_root.exists() and any(args.cache_root.iterdir()):
        raise ValueError(f"--cache-root {args.cache_root} is not empty; name a fresh directory")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["NEURON_LIBTORCH_CACHE_ROOT"] = str(args.cache_root)
    os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise ValueError("The served configuration is NEURON_LOGICAL_NC_CONFIG=2")
    if args.links < 2 or args.iterations < 5 or args.reps < 1 or len(args.seeds) < 1:
        raise ValueError("Use --links >= 2, --iterations >= 5, --reps >= 1 and at least one seed")
    if not Path(kh.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{kh.__name__} imported from {kh.__file__}")
    args.out = args.out.resolve()
    args.pt_dir = (args.pt_dir or args.out.parent).resolve()
    args.pt_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 256
    base = load_baseline(args.cache_root)
    revisions, on_after = load_variants(args.variant, args.cache_root)
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = args.cache_root / "bench-cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "device": "neuron:0 = one logical core (LNC2: 2 physical cores) of the slice",
        "after_module": kh.__file__,
        "baseline": {"commit": BASELINE_COMMIT, "directory": base.directory,
                     "blobs": {name: blob for name, (_, blob) in BASELINE_SOURCES.items()}},
        "variants": {name: {"rev": r.rev, "commit": r.commit, "directory": r.directory,
                            "blobs": r.blobs}
                     for name, r in {**revisions, **on_after}.items()},
        "method": ("per-call = (L-call graph - 1-call graph) / (L - 1), paired per iteration; "
                   "device time from the runtime system trace (LNC2 physical-core intervals "
                   "merged); rotation L-call graphs chain each call on the previous output, "
                   "pooling L-call graphs take L distinct keys and scores and chain through ape "
                   "(next ape = previous output[:pool_size].float())"),
        "iterations": args.iterations, "warmup": args.warmup, "reps": args.reps,
        "cache_root": {"path": str(args.cache_root), "empty_at_start": True},
        "engine_checks": {}, "rotation": [], "pooling": [], "rotation_fp32": [], "decode": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write():
        args.out.write_text(json.dumps(report, indent=1) + "\n")

    report["engine_checks"] = engine_checks(args)
    write()
    for n_rows in args.rows:
        report["rotation"].append(rotation_case(n_rows, args, base, revisions, FIXED_VARIANTS))
        write()
    for n_rows in args.decode_rows:
        report["rotation"].append(rotation_case(n_rows, args, base, revisions,
                                                ("before", "after", "after_aa")))
        write()
    for n_pools in args.pools:
        report["pooling"].append(pool_case(n_pools, args, base, revisions,
                                           report["engine_checks"]))
        write()
    for n_rows in args.fp32_rows:
        report["rotation_fp32"].append(fp32_rotation_case(n_rows, args, base))
        write()
    report["decode"] = decode_cases(args, base, revisions, on_after)
    write()
    for case in report["rotation"] + report["pooling"]:
        print(json.dumps({
            "kernel": case["kernel"], "rows": case.get("rows", case.get("pools")),
            **{name: round(v["per_call"]["median_us"], 2) for name, v in case["variants"].items()},
            "noise_floor_us": round(case["noise_floor_aa_abs_diff"]["median_us"], 2),
            "equal": all(n["after_equals_before"] for n in case["numerics"]),
        }), flush=True)


if __name__ == "__main__":
    main()
