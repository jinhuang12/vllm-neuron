# SPDX-License-Identifier: Apache-2.0
"""Does neuronx-cc overlap the row-parallel all-reduce with independent compute?

No async collective handle reaches the device: the FX graph waits on each all-reduce
right after it, the Neuron lowering emits a plain HLO ``all-reduce`` (never a
``-start`` / ``-done`` pair, which neuronx-cc also refuses: the ``async_start_done``
probe below), and NKI collectives exist only inside a kernel. So any overlap of a site's
all-reduce with compute must come from the compiler's scheduler, given work that does not
depend on the reduced sum. This script measures whether it does, from HLO graphs built
here, compiled with ``neuronx-cc``, timed with ``neuron-bench`` and profiled with
``neuron-explorer`` on one chip.

Run the device stage ONLY through the device lease, which pins the cores and sets LNC2::

    python3 <devlease.py> slice spare1 -- timeout 5400 \\
        /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python \\
        test/hardware/benchmark_allreduce_overlap.py --output-dir <records dir>

``--stage compile`` builds and compiles every graph on the CPU, without the lease;
``--stage run`` then times and profiles those NEFFs under the lease (``--stage all`` does
both). ``--stage summarize`` re-derives ``summary.json`` from the records of an earlier run
(``neuron-explorer view`` reads the kept profiles again; it needs no device).

Geometry. One site's partial is ``[tokens, hidden]`` at the served prefill chunk; the
producer is a matmul whose contraction is the KDA output projection's per-rank width at
:data:`SLICE_RANKS` ranks (``num_heads * head_dim / ranks``), all read from the
checkpoint's config. The all-reduce spans the slice's :data:`SLICE_RANKS` logical cores.

Forms (the ``form`` of an arm):

* ``base``: ``x @ w`` -> all-reduce -> the carrier cast and the residual add.
* ``indep_mm``: ``base`` plus an independent matmul ``z @ wz`` sized to the mHC combine's
  device time (:func:`independent_contraction`), added to the reduced sum by the consumer.
* ``indep_mhc``: the mHC post combine as the consumer, in elementwise HLO:
  ``out[:, j] = post[:, j] * sum + sum_i comb[:, i, j] * streams[:, i]``. Its stream-mix
  term does not read the sum, so it is independent work on the vector engines.
* ``split``: the token split: :data:`SPLIT_PARTS` token blocks, each with its own matmul,
  all-reduce and consumer; block ``k + 1``'s matmul does not read block ``k``'s sum.

Two variants of a form with independent work differ only in one dependency. In the free
variant the independent work reads a scalar from the partial (row 0, summed); in ``dep`` it
reads the same scalar from the reduced sum, so it must wait for the collective. The two
graphs have the same ops and the same cost, so ``dep`` wall minus free wall is the time the
scheduler saved by overlapping (``hidden``). In ``split``, block ``k + 1``'s input reads
that scalar from block ``k``. Each form also runs without the collective (``nocc``, the
identity in its place) as the reference for the exposed time. Every arm runs on the fp32
wire (as built) and on the bf16 wire (the partial cast to bf16 before the collective,
``VLLM_NEURON_TP_ALLREDUCE_DTYPE=bf16``).

Flag sets (``flagset``): ``served`` is ``NeuronModelRunner.load_model``'s neuronx-cc
arguments at ``-O1`` (its fp8 cast option concerns fp8 kernels, which these graphs do not
have). Its ``--modular-flow-mac-threshold=10`` lets the compiler cut any graph into
modules, which it schedules one after another; the module sequence of each arm is in the
manifest (``modules``: the ``split`` form's two equal token blocks become one module run
twice, for one). ``single`` raises that threshold above every graph here
(:data:`ONE_MODULE_ARG`), so the graph is one module and the all-reduce sits inside it, as
most of the served prefill graph's collectives do (its compile log cuts at 14 of its 91
collectives). ``_fuse`` adds the policy's ``FUSE_COMPILER_ARG``. Those four are the core
table (``--table core``). The option sweep (``--table options``) adds the
options that claim to tune whole-graph or collective scheduling (:data:`OPTIONS`) to a
base flag set (:data:`OPTION_SETS`), for the :data:`OPTION_FORMS` on the fp32 wire, with
the collective only.

What is read (``summarize``):

* wall: neuron-bench's ``NCLatency`` per logical core, median over the iterations; the
  core with the smallest median is the one that waited least for the others. Per arm, the
  median over the repeats.
* collective: per rank, the sum of one execution's collective latencies (neuron-bench's
  ``cc_latency_data.json``), median over the executions; the rank with the smallest.
* exposed (wall): a ``cc`` arm's wall minus its ``nocc`` partner's. The compiler may
  schedule the graph without the collective differently, so this is a reference, not the
  verdict. Its noise floor is the sum of the two arms' ranges over the repeats.
* hidden: the ``dep`` wall minus the free wall (same form, wire and flags). ``overlap`` is
  a hidden time above its noise floor, the sum of the two arms' ranges over the repeats.
* exposed (profile): rank 0's device profile of one warm execution; the time a
  collective is in flight while no compute engine (Tensor, Vector, Scalar, GpSimd) of
  either physical core is active. A busy engine is not proof of a hidden collective: the
  engines can slow down while the collective shares the DMA rings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from test.hardware.benchmark_allreduce_policy import (
    CHECKPOINT_CONFIG,
    LEASE_MARKER,
    NEURON_BENCH,
    NEURONX_CC,
    ROOT_COMM_ID,
    RUN_TIMEOUT_S,
    _add,
    _now,
    _per_core_medians,
    _per_rank_collectives,
    _results,
    collective_count,
)

#: The served prefill chunk: the rows of one site's partial.
SITE_TOKENS = 1024
#: Token blocks of the ``split`` form: two 512-token halves of the chunk.
SPLIT_PARTS = 2
#: One chip: the lease slice's four logical cores, one rank each.
SLICE_RANKS = 4
#: The mHC combine's MEASURED device time per half layer at the served 1k chunk (the p1
#: prefill profile: ``hyper_connection_combine``, about 0.55 ms), the size the
#: ``indep_mm`` form's independent matmul is given.
MHC_COMBINE_S = 0.55e-3
#: The MEASURED LNC2 bf16 matmul peak, FLOP/s, that turns that time into a matmul size.
PEAK_FLOPS = 153.0e12
#: The matmul's contraction is a whole number of partition tiles.
PARTITION = 128
#: The runner's neuronx-cc arguments at ``-O1`` (``NeuronModelRunner.load_model``),
#: without its fp8 cast option.
SERVED_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
)
#: The served option that cuts a graph into modules at its collectives.
MODULAR_ARG = SERVED_ARGS[3]
#: The served options of the backend (``walrus_driver``).
BACKEND_ARG = SERVED_ARGS[4]
#: A modular-flow MAC threshold above every graph here (the largest, ``indep_mm``, has
#: about 5.1e10 MACs), so each compiles as one module; :func:`compile_graph` checks it.
ONE_MODULE_MACS = 10**15
ONE_MODULE_ARG = (
    f"--internal-hlo2tensorizer-options=--modular-flow-mac-threshold={ONE_MODULE_MACS}"
)
#: The flag sets of the core table.
CORE_FLAGSETS = ("served", "served_fuse", "single", "single_fuse")
#: Options that claim to tune the scheduling of a whole graph or of its collectives, each
#: as (name, arguments to drop, arguments to add). The first four are documented by
#: ``neuronx-cc compile --help``. ``spmd`` is not: the backend's own help lists
#: ``--enable-SPMD-opt``, "Enable reordering of collectives"; it is measured here and
#: never shipped.
OPTIONS = (
    ("O2", ("-O1",), ("-O2",)),
    ("O3", ("-O1",), ("-O3",)),
    ("transformer", (), ("--model-type=transformer",)),
    ("llmtraining", (), ("--distribution-strategy=llm-training",)),
    ("spmd", (BACKEND_ARG,), (f"{BACKEND_ARG} --enable-SPMD-opt",)),
)
#: (base flag set, option) of the option sweep. ``-O2`` and ``-O3`` compile every graph
#: here as one module from either base, so they run on ``served`` only.
OPTION_SETS = (
    ("served", "O2"),
    ("served", "O3"),
    ("served", "transformer"),
    ("single", "transformer"),
    ("served", "llmtraining"),
    ("single", "llmtraining"),
    ("served", "spmd"),
    ("single", "spmd"),
)
#: The forms of the option sweep: the base (the reference), one with independent work,
#: and the token split.
OPTION_FORMS = ("base", "indep_mm", "split")
FORMS = ("base", "indep_mm", "indep_mhc", "split")
WIRES = ("f32", "bf16")
#: Profile engines that count as compute; ``sync`` (queue issue) and the collective
#: trigger instructions do not.
COMPUTE_ENGINES = ("tensor", "vector", "scalar", "gpsimd")
#: Executions of one profile capture; only the last, a warm one, is profiled.
PROFILE_EXECS = 3
NEURON_EXPLORER = Path("/opt/aws/neuron/bin/neuron-explorer")


# ── geometry ──────────────────────────────────────────────────────────────────


def geometry() -> dict:
    """The site's extents, from the checkpoint's own config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    text = Glm5NextConfig.from_configs(
        json.loads(CHECKPOINT_CONFIG.read_text())
    ).text_config
    kda = text.linear_attn_config
    width = int(kda["num_heads"]) * int(kda["head_dim"])
    if width % SLICE_RANKS:
        raise ValueError(f"KDA width {width} does not shard over {SLICE_RANKS} ranks")
    if SITE_TOKENS % SPLIT_PARTS:
        raise ValueError(f"{SITE_TOKENS} tokens do not split into {SPLIT_PARTS} parts")
    hidden = int(text.hidden_size)
    return {
        "tokens": SITE_TOKENS,
        "hidden": hidden,
        "streams": int(text.hc_mult),
        "contraction": width // SLICE_RANKS,
        "independent_contraction": independent_contraction(SITE_TOKENS, hidden),
        "ranks": SLICE_RANKS,
        "split_parts": SPLIT_PARTS,
    }


def independent_contraction(tokens: int, hidden: int) -> int:
    """The ``indep_mm`` matmul's contraction: :data:`MHC_COMBINE_S` at the peak rate,
    ``2 * tokens * K * hidden`` FLOPs, rounded up to whole partition tiles."""
    flops = MHC_COMBINE_S * PEAK_FLOPS
    return math.ceil(flops / (2 * tokens * hidden) / PARTITION) * PARTITION


# ── the graphs ────────────────────────────────────────────────────────────────


#: ``dot_dimension_numbers`` of ``[m, k] @ [k, n]``.
MATMUL = {"lhs_contracting_dimensions": [1], "rhs_contracting_dimensions": [0]}


def _span(start: int, stop: int) -> dict:
    """One axis of ``slice_dimensions``: ``start:stop``."""
    return {"start": start, "limit": stop, "stride": 1}


def build_graph(form: str, wire: str, cc: bool, dep: bool, geo: dict) -> bytes:
    """The serialized ``HloModuleProto`` of one arm (see the module docstring)."""
    from libtorch_neuronx_lite.pyhlo.scribe import HloScribe

    t, h, s, k = geo["tokens"], geo["hidden"], geo["streams"], geo["contraction"]
    groups = [list(range(geo["ranks"]))]

    def graph(scribe):
        f32, bf16 = scribe.f32, scribe.bf16
        wire_type = {"f32": f32, "bf16": bf16}[wire]
        params = iter(range(16))

        def param(shape):
            return shape.Parameter(parameter_number=next(params))

        def site(x, w, rows):
            """The site's fp32 partial, and its sum: in the wire dtype, all-reduced (or
            not), back in fp32."""
            partial = f32[rows, h].Dot(x, w, dot_dimension_numbers=MATMUL)
            total = bf16[rows, h].Convert(partial) if wire == "bf16" else partial
            if cc:
                total = wire_type[rows, h].AllReduce(
                    total, replica_groups=groups, to_apply=_add(wire_type)
                )
            return partial, (total if wire == "f32" else f32[rows, h].Convert(total))

        w = param(bf16[k, h])
        zero = None if form == "base" else param(f32[()])

        def gate(partial, total, shape, dtype):
            """A scalar broadcast to ``shape``: the sum of row 0 of the reduced sum
            (``dep``) or of the partial (free), so ``dep`` alone orders the independent
            work after the collective, at the same cost."""
            source = total if dep else partial
            row = f32[1, h].Slice(source, slice_dimensions=[_span(0, 1), _span(0, h)])
            scalar = f32[()].Reduce(row, zero, dimensions=[0, 1], to_apply=_add(f32))
            return dtype[shape].Broadcast(dtype[()].Convert(scalar), dimensions=[])

        if form in ("base", "split"):
            parts = 1 if form == "base" else geo["split_parts"]
            rows = t // parts
            xs = [param(bf16[rows, k]) for _ in range(parts)]
            residuals = [param(bf16[rows, h]) for _ in range(parts)]
            outs, previous = [], None
            for x, r in zip(xs, residuals):
                if previous is not None:
                    x = bf16[rows, k].Multiply(x, gate(*previous, [rows, k], bf16))
                previous = site(x, w, rows)
                outs.append(bf16[rows, h].Add(bf16[rows, h].Convert(previous[1]), r))
            if parts == 1:
                return outs[0]
            return scribe.tuple(*[bf16[rows, h]] * parts).Tuple(*outs)
        x = param(bf16[t, k])
        partial, total = site(x, w, t)
        if form == "indep_mm":
            residual = param(bf16[t, h])
            kz = geo["independent_contraction"]
            z, wz = param(bf16[t, kz]), param(bf16[kz, h])
            z = bf16[t, kz].Multiply(z, gate(partial, total, [t, kz], bf16))
            independent = f32[t, h].Dot(z, wz, dot_dimension_numbers=MATMUL)
            out = f32[t, h].Add(total, independent)
            return bf16[t, h].Add(bf16[t, h].Convert(out), residual)
        if form == "indep_mhc":
            streams = f32[t, s, h].Multiply(
                f32[t, s, h].Convert(param(bf16[t, s, h])),
                gate(partial, total, [t, s, h], f32),
            )
            post, comb = param(f32[t, s]), param(f32[t, s, s])

            def column(tensor, *index):
                """``tensor[:, *index]``, ``[t]``, broadcast to ``[t, h]``."""
                cut = f32[[t] + [1] * len(index)].Slice(
                    tensor,
                    slice_dimensions=[_span(0, t)] + [_span(i, i + 1) for i in index],
                )
                return f32[t, h].Broadcast(f32[t].Reshape(cut), dimensions=[0])

            stream = [
                f32[t, h].Reshape(
                    f32[t, 1, h].Slice(
                        streams,
                        slice_dimensions=[_span(0, t), _span(i, i + 1), _span(0, h)],
                    )
                )
                for i in range(s)
            ]
            mixed = []
            for j in range(s):
                acc = f32[t, h].Multiply(column(post, j), total)
                for i in range(s):
                    acc = f32[t, h].Add(
                        acc, f32[t, h].Multiply(column(comb, i, j), stream[i])
                    )
                mixed.append(bf16[t, 1, h].Reshape(bf16[t, h].Convert(acc)))
            return bf16[t, s, h].Concatenate(*mixed, dimensions=[1])
        raise ValueError(f"unknown form {form!r}")

    graph.__name__ = arm_name(form, dep, wire, cc, "graph")
    return HloScribe()(graph).module_proto.SerializeToString()


def build_async_probe(geo: dict) -> bytes:
    """``base`` (fp32, ``cc``) with its all-reduce as an ``all-reduce-start`` /
    ``all-reduce-done`` pair, the async form the HLO opcode set has.

    The scribe gives the replica groups and the reduction computation to ``all-reduce``
    only, so the pair is written as ``all-reduce`` and ``copy`` and then renamed.
    """
    from libtorch_neuronx_lite.pyhlo.scribe import HloScribe

    t, h, k = geo["tokens"], geo["hidden"], geo["contraction"]

    def async_start_done(scribe):
        f32, bf16 = scribe.f32, scribe.bf16
        w = bf16[k, h].Parameter(parameter_number=0)
        x = bf16[t, k].Parameter(parameter_number=1)
        residual = bf16[t, h].Parameter(parameter_number=2)
        partial = f32[t, h].Dot(x, w, dot_dimension_numbers=MATMUL)
        start = f32[t, h].AllReduce(
            partial, replica_groups=[list(range(geo["ranks"]))], to_apply=_add(f32)
        )
        done = f32[t, h].Copy(start)
        return bf16[t, h].Add(bf16[t, h].Convert(done), residual)

    module = HloScribe()(async_start_done).module_proto
    renames = {"all-reduce": "all-reduce-start", "copy": "all-reduce-done"}
    for computation in module.computations:
        for inst in computation.instructions:
            inst.opcode = renames.get(inst.opcode, inst.opcode)
    return module.SerializeToString()


def flagsets() -> dict[str, tuple[str, ...]]:
    """Every flag set by name: :data:`CORE_FLAGSETS` and :data:`OPTION_SETS`."""
    from vllm_neuron.model.glm5_next.collective_policy import FUSE_COMPILER_ARG

    single = tuple(ONE_MODULE_ARG if a == MODULAR_ARG else a for a in SERVED_ARGS)
    out = {
        "served": SERVED_ARGS,
        "served_fuse": SERVED_ARGS + (FUSE_COMPILER_ARG,),
        "single": single,
        "single_fuse": single + (FUSE_COMPILER_ARG,),
    }
    options = {name: (drop, add) for name, drop, add in OPTIONS}
    for base, option in OPTION_SETS:
        drop, add = options[option]
        kept = tuple(a for a in out[base] if a not in drop)
        out[f"{base}_{option}"] = kept + add
    return out


def arm_name(form: str, dep: bool, wire: str, cc: bool, flagset: str) -> str:
    """``<form>[_dep]_<wire>_<cc|nocc>_<flagset>``."""
    variant = f"{form}_dep" if dep else form
    return f"{variant}_{wire}_{'cc' if cc else 'nocc'}_{flagset}"


def arms() -> list[dict]:
    """Every timed arm, the core table and then the option sweep.

    A form with independent work runs free and ``dep`` (``cc``). In the core table each
    form also runs free without the collective (``nocc``: there the two variants are the
    same graph), and a ``cc`` arm's ``nocc`` partner is that arm at its flag set without
    ``_fuse`` (no collective, nothing for the flag to change). The option sweep has no
    ``nocc`` arms: its verdict is the free/``dep`` pair alone.
    """

    def arm(form, dep, wire, cc, flagset, partnered):
        partner = None
        if cc and partnered:
            partner = arm_name(form, False, wire, False, flagset.removesuffix("_fuse"))
        return {
            "name": arm_name(form, dep, wire, cc, flagset),
            "form": form,
            "dep": dep,
            "wire": wire,
            "cc": cc,
            "flagset": flagset,
            "partner": partner,
            "table": "core" if flagset in CORE_FLAGSETS else "options",
        }

    def form_arms(form, wire, cc_flagsets, nocc_flagsets):
        variants = (False,) if form == "base" else (False, True)
        partnered = bool(nocc_flagsets)
        out = [
            arm(form, dep, wire, True, f, partnered)
            for dep in variants
            for f in cc_flagsets
        ]
        return out + [arm(form, False, wire, False, f, False) for f in nocc_flagsets]

    out = []
    for wire in WIRES:
        for form in FORMS:
            out += form_arms(form, wire, CORE_FLAGSETS, ("served", "single"))
    for base, option in OPTION_SETS:
        flagset = f"{base}_{option}"
        for form in OPTION_FORMS:
            out += form_arms(form, "f32", (flagset,), ())
    return out


# ── compile (CPU) ─────────────────────────────────────────────────────────────


def compile_graph(graph: bytes, flags: tuple[str, ...], work: Path) -> dict:
    """Compile ``graph`` in ``work``; the command, exit code and collective count."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "graph.pb").write_bytes(graph)
    neff = work / "graph.neff"
    cmd = [
        str(NEURONX_CC),
        "compile",
        "graph.pb",
        "--framework",
        "XLA",
        "--target",
        "trn2",
        "--output",
        str(neff),
        "--logfile",
        "log-neuron-cc.txt",
        *flags,
    ]
    with open(work / "compile_stdout.txt", "w") as out:
        rc = subprocess.run(
            cmd, cwd=work, stdout=out, stderr=subprocess.STDOUT, check=False
        ).returncode
    for scratch in work.glob("neuronxcc-*"):
        shutil.rmtree(scratch, ignore_errors=True)
    record = {"command": cmd, "rc": rc, "neff": str(neff) if neff.exists() else None}
    if rc == 0 and neff.exists():
        log = work / "log-neuron-cc.txt"
        record["allreduces_compiled"] = collective_count(log)
        record["modules"] = modules(log)
        if ONE_MODULE_ARG in flags and len(record["modules"]) != 1:
            raise RuntimeError(f"{work}: {ONE_MODULE_ARG} left {record['modules']}")
    return record


def modules(log: Path) -> list[int]:
    """The module sequence one execution runs, as module definition ids: the
    partitioner's last ``DefMap`` line, or ``[0]`` for a graph it did not cut. Repeated
    ids are one compiled module executed again (two equal token blocks, for one)."""
    found = re.findall(r"DefMap: ([\d ]+)", log.read_text(errors="replace"))
    return [int(v) for v in found[-1].split()] if found else [0]


# ── bench and profile (device) ────────────────────────────────────────────────


def bench(neff: Path, out: Path, cc: bool, iterations: int, warmup: int) -> dict:
    """One neuron-bench run of ``neff`` on all :data:`SLICE_RANKS` cores at once: one CC
    rank per core, or (``nocc``) one independent instance per core."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(NEURON_BENCH),
        "exec",
        "-n",
        str(iterations),
        "-w",
        str(warmup),
        "-o",
        str(out / "results"),
        "--disable-hw-profile",
        "--disable-system-perf",
        "--disable-throughput",
        f"--fixed-instance-count={SLICE_RANKS}",
    ]
    env = dict(os.environ)
    if cc:
        cmd += [
            "--run-as-cc-neff",
            "--cc-single-neff",
            f"--cc-world-size={SLICE_RANKS}",
        ]
        env["NEURON_RT_ROOT_COMM_ID"] = ROOT_COMM_ID
    cmd.append(str(neff))
    return _run(cmd, out, env)


def profile(neff: Path, out: Path) -> dict:
    """A device profile of rank 0 over :data:`SLICE_RANKS` CC ranks; only the last of
    :data:`PROFILE_EXECS` executions is captured."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(NEURON_EXPLORER),
        "capture",
        "-n",
        str(neff),
        "-s",
        str(out / "profile.ntff"),
        "-r",
        str(SLICE_RANKS),
        f"--collectives-worker-count={SLICE_RANKS}",
        "-i",
        "0",
        f"--num-exec={PROFILE_EXECS}",
        f"--profile-nth-exec={PROFILE_EXECS}",
    ]
    env = dict(os.environ, NEURON_RT_ROOT_COMM_ID=ROOT_COMM_ID)
    return _run(cmd, out, env)


def _run(cmd: list[str], out: Path, env: dict) -> dict:
    """Run ``cmd`` in ``out`` with its output in ``out/stdout.txt``."""
    start = _now()
    with open(out / "stdout.txt", "w") as log:
        rc = subprocess.run(
            cmd,
            cwd=out,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
        ).returncode
    return {"command": cmd, "rc": rc, "start": start, "end": _now()}


# ── summarize (offline, from the records) ─────────────────────────────────────


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The union of ``[start, end)`` intervals, sorted and disjoint."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _length(intervals: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in intervals)


def _intersection(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total length of the overlap of two sorted disjoint interval lists."""
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def profile_exposure(neff: Path, ntff: Path, scratch: Path) -> dict:
    """From one profile: each collective's window, and how much of the collective time
    has a compute engine active on either physical core (ns -> us)."""
    scratch.mkdir(parents=True, exist_ok=True)
    data = scratch / "profile.json"
    subprocess.run(
        [
            str(NEURON_EXPLORER),
            "view",
            "-n",
            str(neff),
            "-s",
            str(ntff),
            "--output-format",
            "json",
            "--output-file",
            str(data),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        doc = json.loads(data.read_text())
    finally:
        data.unlink(missing_ok=True)
    ops = [
        {
            "operation": op["operation"],
            "bytes": op["input_size"],
            "dtype": op["dtype"],
            "algorithm": op["algorithm"],
            "start_us": op["timestamp"] / 1e3,
            "duration_us": op["duration"] / 1e3,
        }
        for op in sorted(doc["cc_ops"], key=lambda op: op["timestamp"])
    ]
    collective = _union(
        [(op["timestamp"], op["timestamp"] + op["duration"]) for op in doc["cc_ops"]]
    )
    compute = _union(
        [
            (a["start_ts"], a["end_ts"])
            for a in doc["active_time"]
            if a.get("engine") in COMPUTE_ENGINES
        ]
    )
    busy = _intersection(collective, compute)
    ends = [a["end_ts"] for a in doc["active_time"]]
    return {
        "collectives": ops,
        "collective_us": _length(collective) / 1e3,
        "compute_during_collective_us": busy / 1e3,
        "exposed_us": (_length(collective) - busy) / 1e3,
        "execution_us": (max(ends) - min(a["start_ts"] for a in doc["active_time"]))
        / 1e3,
        "compute_us": _length(compute) / 1e3,
    }


def _rep_values(output_dir: Path, manifest: dict, name: str) -> list[dict]:
    """Per repeat of arm ``name``: the wall and, for a ``cc`` arm, the collective time."""
    record = manifest["compiled"][name]
    reps = []
    for rep in range(len(manifest["bench"][name])):
        results = _results(output_dir / "raw" / name / f"rep{rep}")
        per_core = _per_core_medians(results)
        value = {"wall_us_per_core": per_core, "wall_us": min(per_core.values())}
        if record["cc"]:
            collectives = _per_rank_collectives(results)
            counts = {len(v) for v in collectives.values()}
            if len(counts) != 1 or counts.pop() % manifest["iterations"]:
                raise RuntimeError(
                    f"{name} rep {rep}: uneven collective counts per rank"
                )
            per_execution = (
                len(next(iter(collectives.values()))) // manifest["iterations"]
            )
            # Each rank's collective time per execution: the sum of its consecutive
            # ``per_execution`` latencies, median over the executions.
            per_rank = {
                r: statistics.median(
                    sum(v[i : i + per_execution])
                    for i in range(0, len(v), per_execution)
                )
                for r, v in collectives.items()
            }
            value["collectives_per_execution"] = per_execution
            value["collective_us_per_rank"] = per_rank
            value["collective_us"] = min(per_rank.values())
        reps.append(value)
    return reps


def summarize(output_dir: Path) -> dict:
    """``summary.json`` from ``manifest.json`` and the records under ``raw/``.

    ``rows`` holds every arm's numbers; ``table`` one line per (form, wire, flag set):
    the free and ``dep`` walls, what the collective exposes in each, and ``hidden``, the
    ``dep`` wall minus the free one (the time the scheduler saves where it may overlap),
    with its noise floor (the two arms' ranges over the repeats added) and the verdict.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text())
    failed = [
        f"{name} rep {rep}"
        for name, records in manifest["bench"].items()
        for rep, record in enumerate(records)
        if record is None or record["rc"] != 0
    ] + [name for name, record in manifest["profile"].items() if record["rc"] != 0]
    if failed:
        raise RuntimeError(f"runs that did not exit 0: {failed}")
    rows = {}
    for name in manifest["bench"]:
        record = manifest["compiled"][name]
        reps = _rep_values(output_dir, manifest, name)
        walls = [r["wall_us"] for r in reps]
        row = {
            key: record[key]
            for key in ("form", "dep", "wire", "cc", "flagset", "partner", "table")
        }
        row.update(
            allreduces_compiled=record["allreduces_compiled"],
            modules=record["modules"],
            wall_us_reps=walls,
            wall_us=statistics.median(walls),
            wall_range_us=max(walls) - min(walls),
        )
        if record["cc"]:
            row["collectives_per_execution"] = reps[0]["collectives_per_execution"]
            row["collective_us_reps"] = [r["collective_us"] for r in reps]
            row["collective_us"] = statistics.median(row["collective_us_reps"])
            partner = manifest["bench"].get(row["partner"])
            if partner is not None:
                walls_nocc = [
                    r["wall_us"]
                    for r in _rep_values(output_dir, manifest, row["partner"])
                ]
                row["exposed_us"] = row["wall_us"] - statistics.median(walls_nocc)
                row["exposed_noise_us"] = row["wall_range_us"] + (
                    max(walls_nocc) - min(walls_nocc)
                )
        if name in manifest["profile"]:
            (ntff,) = sorted((output_dir / "raw" / name / "profile").glob("*.ntff"))
            exposure = profile_exposure(
                Path(record["neff"]), ntff, output_dir / "scratch"
            )
            exposure["ntff"] = str(ntff.relative_to(output_dir))
            row["profile"] = exposure
        rows[name] = row
    shutil.rmtree(output_dir / "scratch", ignore_errors=True)
    table = []
    for name, row in rows.items():
        if not row["cc"] or row["dep"]:
            continue
        line = {
            key: row[key] for key in ("table", "form", "wire", "flagset", "modules")
        }
        line.update(
            collectives=row["collectives_per_execution"],
            wall_us=row["wall_us"],
            exposed_us=row.get("exposed_us"),
            exposed_noise_us=row.get("exposed_noise_us"),
            compute_share_of_collective=_compute_share(row),
        )
        dep = rows.get(arm_name(row["form"], True, row["wire"], True, row["flagset"]))
        if dep is not None:
            line.update(
                dep_wall_us=dep["wall_us"],
                dep_exposed_us=dep.get("exposed_us"),
                dep_compute_share_of_collective=_compute_share(dep),
                hidden_us=dep["wall_us"] - row["wall_us"],
                hidden_noise_us=dep["wall_range_us"] + row["wall_range_us"],
            )
            line["overlap"] = line["hidden_us"] > line["hidden_noise_us"]
        table.append(line)
    return {
        "kind": "MEASURED",
        "script": manifest["script"],
        "script_sha256": manifest["script_sha256"],
        "environment": manifest["environment"],
        "window": [manifest.get("bench_start"), manifest.get("bench_end")],
        "geometry": manifest["geometry"],
        "flagsets": manifest["flagsets"],
        "iterations": manifest["iterations"],
        "warmup": manifest["warmup"],
        "repeats": manifest["repeats"],
        "probes": manifest["probes"],
        "table": table,
        "rows": rows,
    }


def _compute_share(row: dict) -> float | None:
    """The share of a profiled arm's collective time with a compute engine active."""
    exposure = row.get("profile")
    if exposure is None:
        return None
    return exposure["compute_during_collective_us"] / exposure["collective_us"]


# ── driver ────────────────────────────────────────────────────────────────────


def compile_all(out: Path, timed: list[dict], jobs: int) -> dict:
    """Compile every arm of ``timed`` and the async probe under ``out/raw``; the
    manifest's compile records."""
    raw = out / "raw"
    if raw.exists() and any(raw.iterdir()):
        raise SystemExit(f"{raw} is not empty; give a new --output-dir")
    raw.mkdir(parents=True, exist_ok=True)
    geo, flags = geometry(), flagsets()
    compiled = {}
    start = _now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                compile_graph,
                build_graph(a["form"], a["wire"], a["cc"], a["dep"], geo),
                flags[a["flagset"]],
                raw / a["name"] / "compile",
            ): a
            for a in timed
        }
        probe = pool.submit(
            compile_graph,
            build_async_probe(geo),
            SERVED_ARGS,
            raw / "probe_async_start_done" / "compile",
        )
        for future in concurrent.futures.as_completed(futures):
            a = futures[future]
            record = future.result()
            if record["rc"] != 0:
                raise RuntimeError(f"{a['name']}: neuronx-cc exit {record['rc']}")
            compiled[a["name"]] = {**a, **record}
            print(
                f"compiled {a['name']}: {record['allreduces_compiled']} all-reduces, "
                f"modules {record['modules']}",
                flush=True,
            )
        async_probe = probe.result()
    return {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": sys.argv,
        "geometry": geo,
        "flagsets": {name: list(v) for name, v in flags.items()},
        "compile_start": start,
        "compile_end": _now(),
        "compiled": {a["name"]: compiled[a["name"]] for a in timed},
        "probes": {"async_start_done": async_probe},
        "bench": {},
        "profile": {},
    }


def main() -> int:
    """Compile, time, profile and summarize, or one stage of that."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="records directory: raw/, manifest.json, summary.json",
    )
    parser.add_argument(
        "--stage", choices=("all", "compile", "run", "summarize"), default="all"
    )
    parser.add_argument(
        "--table",
        choices=("all", "core", "options"),
        default="all",
        help="the core table, the option sweep, or both",
    )
    parser.add_argument(
        "--wire",
        choices=("all",) + WIRES,
        default="all",
        help="the arms of one wire only (a shorter lease)",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=6, help="parallel compiles")
    args = parser.parse_args()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    out = args.output_dir.resolve()
    manifest_path = out / "manifest.json"

    def save():
        """The manifest so far: a stopped run keeps the records it made."""
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    if args.stage == "summarize":
        summary = summarize(out)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"wrote {out / 'summary.json'}")
        return 0
    if args.stage != "compile":
        if not os.environ.get(LEASE_MARKER):
            raise SystemExit(
                f"run under devlease.py slice <name>, which sets {LEASE_MARKER}"
            )
        if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
            raise SystemExit(
                "the NEFFs are LNC2; the device lease sets NEURON_LOGICAL_NC_CONFIG=2"
            )
    selected = [
        a
        for a in arms()
        if args.table in ("all", a["table"]) and args.wire in ("all", a["wire"])
    ]
    if args.stage == "run":
        manifest = json.loads(manifest_path.read_text())
        if manifest["bench"]:
            raise SystemExit(f"{out} has bench records already; compile a new one")
        timed = [a for a in selected if a["name"] in manifest["compiled"]]
    else:
        manifest = compile_all(out, selected, args.jobs)
        timed = selected
        save()
        if args.stage == "compile":
            return 0
    manifest.update(
        argv_run=sys.argv,
        iterations=args.iterations,
        warmup=args.warmup,
        repeats=args.repeats,
        environment={
            key: os.environ.get(key)
            for key in (
                LEASE_MARKER,
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
            )
        },
        bench_start=_now(),
    )
    raw = out / "raw"
    for rep in range(args.repeats):
        # Interleaved: every arm once per repeat, the order rotated each repeat.
        shift = rep * len(timed) // args.repeats
        for a in timed[shift:] + timed[:shift]:
            record = bench(
                Path(manifest["compiled"][a["name"]]["neff"]),
                raw / a["name"] / f"rep{rep}",
                a["cc"],
                args.iterations,
                args.warmup,
            )
            manifest["bench"].setdefault(a["name"], [None] * args.repeats)[rep] = record
            save()
            print(f"rep {rep} {a['name']}: rc {record['rc']}", flush=True)
    for a in timed:
        if a["cc"]:
            record = profile(
                Path(manifest["compiled"][a["name"]]["neff"]),
                raw / a["name"] / "profile",
            )
            manifest["profile"][a["name"]] = record
            save()
            print(f"profile {a['name']}: rc {record['rc']}", flush=True)
    manifest["bench_end"] = _now()
    save()
    summary = summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
