# SPDX-License-Identifier: Apache-2.0
"""Time the device ops of the row-parallel all-reduce policy, next to their entitlement.

``collective_policy.reduce_row_parallel`` puts no new op on a site: the fp32 wire is the
as-built graph, and the bf16 wire moves the site's f32 -> bf16 cast from after the
collective to before it. This script measures that cast and the collective in both orders,
from HLO graphs built here, compiled with ``neuronx-cc`` and timed with ``neuron-bench``.

Run the device stage ONLY through the device lease. The lease pins the cores
(``NEURON_RT_VISIBLE_CORES``) and sets LNC2; this script refuses to time without the
pinned cores and does not select cores itself::

    python3 <devlease.py> slice spare2 -- timeout 900 \\
        /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python \\
        test/hardware/benchmark_allreduce_policy.py --output-dir <records dir>

``--stage compile`` builds and compiles every graph on the CPU, without the lease.
``--stage summarize`` re-derives ``summary.json`` from the raw records of an earlier run.
The script sets ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` itself when it is unset.

Geometry: a site's partial is ``[tokens, hidden_size]``, with ``hidden_size`` read from the
checkpoint's config (``test/vllm_neuron/model/glm5_next/fixtures/hf-config.json``).

* ``convert_<T>``: one ``convert`` f32 -> bf16 of ``[T, hidden]`` for T in 256 .. 4096, on
  one logical core. Its ENTITLEMENT is ``T * hidden * (4 + 2)`` bytes at the per-core HBM
  bandwidth. The launch cost is the intercept of a least-squares fit of the NEFF time over
  the bytes; the op time is the NEFF time minus that intercept.
* ``site_f32`` / ``site_bf16``: a TP=4 site at 1024 tokens in each order. A ``multiply``
  stands in for the producer: ``multiply -> all-reduce f32 -> convert`` (as built) and
  ``multiply -> convert -> all-reduce bf16`` (the bf16 wire).
* ``allreduce_f32`` / ``allreduce_f32_fused`` / ``allreduce_bf16``: ``multiply ->
  all-reduce`` at TP=4 and 1024 tokens, as compiled (the compiler tiles the 16 MiB fp32
  all-reduce in two), with the policy's fusion flag (one collective), and in bf16.
* ``nccom-test all_reduce`` at TP=4, 4 / 8 / 16 MiB, fp32 and bf16: the runtime collective
  alone.
* Tiling probes (compile only): ``multiply -> all-reduce`` over 64 replicas, the served TP
  degree, at the sizes of the report's s4; each counts the collectives the compiler emits.

The NEFF device time is neuron-bench's ``NCLatency`` p50 over ``--iterations`` executions;
a CC NEFF also reports each rank's collective latencies. Every command's output and
results directory is kept under ``--output-dir`` (``raw/``), so the summary can be
re-derived from the records alone.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

#: The checkpoint's own HF config, the file the model tests read as the real one.
CHECKPOINT_CONFIG = (
    ROOT
    / "test"
    / "vllm_neuron"
    / "model"
    / "glm5_next"
    / "fixtures"
    / "hf-config.json"
)
#: Token counts of the standalone convert: a prefill chunk of 1024 and two octaves around.
CONVERT_TOKENS = (256, 512, 1024, 2048, 4096)
#: The served prefill chunk, the size of every TP=4 arm.
SITE_TOKENS = 1024
#: One chip: the lease slice's four logical cores, one rank each.
SLICE_RANKS = 4
#: The served tensor-parallel degree, for the compile-only tiling probes.
SERVED_RANKS = 64
#: (dtype, tokens, fused) of each tiling probe: the rows of the report's s4 table.
TILING_PROBES = (
    ("f32", 512, False),
    ("f32", 1024, False),
    ("f32", 2048, False),
    ("f32", 4096, False),
    ("bf16", 1024, False),
    ("bf16", 2048, False),
    ("f32", 1024, True),
)
MIB = 1 << 20
#: The size neuronx-cc's ``SimpleAllReduceTiling`` cuts a larger all-reduce into.
COMPILER_TILE_BYTES = 8 * MIB
#: nccom-test all_reduce sizes, a doubling series around the tile: half, one, two tiles.
NCCOM_STEP_FACTOR = 2
NCCOM_BYTES = (
    COMPILER_TILE_BYTES // NCCOM_STEP_FACTOR,
    COMPILER_TILE_BYTES,
    COMPILER_TILE_BYTES * NCCOM_STEP_FACTOR,
)
#: Bytes per element of each HLO dtype.
DTYPE_BYTES = {"f32": 4, "bf16": 2}
#: nccom-test's name for each HLO dtype.
NCCOM_DTYPES = {"f32": "fp32", "bf16": "bf16"}
#: Per logical core (two physical cores at LNC2): the planner's
#: ``trn2_constants.json`` ``hbm_bw_bytes_per_s``.
HBM_BYTES_PER_S = 716e9
#: Logical cores per compiled NEFF: neuronx-cc targets LNC2 on trn2, and its
#: ``[Unroll] CollectiveCompute`` line counts each collective once per physical core.
LNC = 2
#: The flags ``NeuronModelRunner.load_model`` passes at ``-O1`` that apply to these
#: graphs (no auto-cast, the verifier off). Its hlo2tensorizer MAC threshold, fp8 cast and
#: nested-loop options concern NKI kernels and fp8, which these graphs do not have. The
#: fusion flag is the policy's own ``FUSE_COMPILER_ARG``.
COMPILER_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-backend-options=--enable-verifier=false",
)
#: The local rendezvous a single-process CC NEFF needs; neuron-bench does not load one
#: without it.
ROOT_COMM_ID = "127.0.0.1:47917"
#: The pinned-core marker the device lease sets.
LEASE_MARKER = "NEURON_RT_VISIBLE_CORES"
#: Seconds one neuron-bench or nccom-test run may take before the script gives up.
RUN_TIMEOUT_S = 600
NEURONX_CC = Path(sys.executable).parent / "neuronx-cc"
NEURON_BENCH = Path("/opt/aws/neuron/bin/neuron-bench")
NCCOM_TEST = Path("/opt/aws/neuron/bin/nccom-test")


def hidden_size() -> int:
    """``hidden_size`` of the checkpoint's text config: the width of every site's partial."""
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    config = Glm5NextConfig.from_configs(json.loads(CHECKPOINT_CONFIG.read_text()))
    return int(config.text_config.hidden_size)


# ── the graphs ────────────────────────────────────────────────────────────────


def _add(dtype):
    """The ``to_apply`` computation of a sum all-reduce in ``dtype``."""

    def add(scribe):
        return dtype.Add(
            dtype.Parameter(parameter_number=0), dtype.Parameter(parameter_number=1)
        )

    return add


def build_graph(kind: str, tokens: int, hidden: int, ranks: int = 1) -> bytes:
    """The serialized ``HloModuleProto`` of one arm.

    Args:
        kind: ``convert`` (f32 in, bf16 out), ``site_f32`` (multiply, all-reduce f32,
            convert), ``site_bf16`` (multiply, convert, all-reduce bf16), or
            ``allreduce_<dtype>`` (multiply, all-reduce in ``dtype``).
        tokens, hidden: the partial's shape.
        ranks: the all-reduce's replica group, ``0 .. ranks - 1``.
    """
    from libtorch_neuronx_lite.pyhlo.scribe import HloScribe

    groups = [list(range(ranks))]

    def graph(scribe):
        f32, bf16 = scribe.f32[tokens, hidden], scribe.bf16[tokens, hidden]
        if kind == "convert":
            return bf16.Convert(f32.Parameter(parameter_number=0))
        if kind in ("site_f32", "site_bf16"):
            x = f32.Parameter(parameter_number=0)
            partial = f32.Multiply(x, x)
            if kind == "site_f32":
                total = f32.AllReduce(
                    partial, replica_groups=groups, to_apply=_add(scribe.f32)
                )
                return bf16.Convert(total)
            wire = bf16.Convert(partial)
            return bf16.AllReduce(
                wire, replica_groups=groups, to_apply=_add(scribe.bf16)
            )
        if kind.startswith("allreduce_"):
            dtype = kind.removeprefix("allreduce_")
            shape = {"f32": f32, "bf16": bf16}[dtype]
            x = shape.Parameter(parameter_number=0)
            partial = shape.Multiply(x, x)
            return shape.AllReduce(
                partial, replica_groups=groups, to_apply=_add(getattr(scribe, dtype))
            )
        raise ValueError(f"unknown graph kind {kind!r}")

    # The scribe names the module after the function.
    graph.__name__ = kind
    return HloScribe()(graph).module_proto.SerializeToString()


def arms(hidden: int) -> list[dict]:
    """Every timed arm: its graph, its compiler flags and how neuron-bench runs it."""
    from vllm_neuron.model.glm5_next.collective_policy import FUSE_COMPILER_ARG

    out = [
        {
            "name": f"convert_{t}",
            "kind": "convert",
            "tokens": t,
            "ranks": 1,
            "flags": [],
        }
        for t in CONVERT_TOKENS
    ]
    for kind, flags, name in (
        ("site_f32", [], "site_f32"),
        ("site_bf16", [], "site_bf16"),
        ("allreduce_f32", [], "allreduce_f32"),
        ("allreduce_f32", [FUSE_COMPILER_ARG], "allreduce_f32_fused"),
        ("allreduce_bf16", [], "allreduce_bf16"),
    ):
        out.append(
            {
                "name": name,
                "kind": kind,
                "tokens": SITE_TOKENS,
                "ranks": SLICE_RANKS,
                "flags": flags,
            }
        )
    for arm in out:
        arm["hidden"] = hidden
    return out


def probes(hidden: int) -> list[dict]:
    """The compile-only tiling probes over :data:`SERVED_RANKS` replicas."""
    from vllm_neuron.model.glm5_next.collective_policy import FUSE_COMPILER_ARG

    return [
        {
            "name": f"probe_{dtype}_{tokens}" + ("_fused" if fused else ""),
            "kind": f"allreduce_{dtype}",
            "tokens": tokens,
            "ranks": SERVED_RANKS,
            "hidden": hidden,
            "flags": [FUSE_COMPILER_ARG] if fused else [],
        }
        for dtype, tokens, fused in TILING_PROBES
    ]


# ── compile (CPU) ─────────────────────────────────────────────────────────────


def collective_count(log: Path) -> int:
    """All-reduces in a compiled NEFF: the ``[Unroll] CollectiveCompute`` lines, per LNC."""
    total = 0
    for line in log.read_text(errors="replace").splitlines():
        if "[Unroll]" in line and "CollectiveCompute:" in line:
            total += int(line.rsplit("CollectiveCompute:", 1)[1].split()[0])
    if total % LNC:
        raise RuntimeError(
            f"{log}: {total} CollectiveCompute is not a multiple of {LNC}"
        )
    return total // LNC


def compile_arm(arm: dict, work: Path) -> dict:
    """Build and compile ``arm`` in its own directory; the NEFF and its collective count."""
    work.mkdir(parents=True, exist_ok=True)
    hlo = work / "graph.pb"
    hlo.write_bytes(
        build_graph(arm["kind"], arm["tokens"], arm["hidden"], arm["ranks"])
    )
    neff, log = work / "graph.neff", work / "log-neuron-cc.txt"
    cmd = [
        str(NEURONX_CC),
        "compile",
        str(hlo),
        "--framework",
        "XLA",
        "--target",
        "trn2",
        "--output",
        str(neff),
        "--logfile",
        str(log),
        *COMPILER_ARGS,
        *arm["flags"],
    ]
    with open(work / "compile_stdout.txt", "w") as out:
        rc = subprocess.run(
            cmd, cwd=work, stdout=out, stderr=subprocess.STDOUT, check=False
        ).returncode
    if rc != 0 or not neff.exists():
        raise RuntimeError(f"{arm['name']}: neuronx-cc exit {rc}; see {work}")
    for scratch in work.glob("neuronxcc-*"):
        shutil.rmtree(scratch, ignore_errors=True)
    return {"command": cmd, "neff": str(neff), "allreduces": collective_count(log)}


# ── bench (device) ────────────────────────────────────────────────────────────


def bench_arm(arm: dict, neff: Path, out: Path, iterations: int, warmup: int) -> dict:
    """One neuron-bench run of ``neff``; its stdout and results directory go to ``out``."""
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
    ]
    env = dict(os.environ)
    if arm["ranks"] > 1:
        # One instance per rank, all from the one NEFF.
        cmd += [
            "--run-as-cc-neff",
            "--cc-single-neff",
            f"--cc-world-size={arm['ranks']}",
            f"--fixed-instance-count={arm['ranks']}",
        ]
        env["NEURON_RT_ROOT_COMM_ID"] = ROOT_COMM_ID
    else:
        cmd += ["--fixed-nc-count=1"]
    cmd.append(str(neff))
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


def nccom(dtype: str, out: Path, iterations: int, warmup: int) -> dict:
    """``nccom-test all_reduce`` at :data:`SLICE_RANKS` over :data:`NCCOM_BYTES`."""
    out.mkdir(parents=True, exist_ok=True)
    report = out / "report.json"
    cmd = [
        str(NCCOM_TEST),
        "all_reduce",
        "-r",
        str(SLICE_RANKS),
        "-d",
        NCCOM_DTYPES[dtype],
        "-b",
        str(min(NCCOM_BYTES)),
        "-e",
        str(max(NCCOM_BYTES)),
        "-f",
        str(NCCOM_STEP_FACTOR),
        "-n",
        str(iterations),
        "-w",
        str(warmup),
        "--non-interactive",
        "--report-to-json-file",
        str(report),
    ]
    start = _now()
    with open(out / "stdout.txt", "w") as log:
        rc = subprocess.run(
            cmd,
            cwd=out,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
        ).returncode
    return {"command": cmd, "rc": rc, "start": start, "end": _now()}


def _now() -> str:
    """The current UTC time, to the second, as the records stamp it."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── summarize (offline, from the raw records) ─────────────────────────────────


def _results(rep_dir: Path) -> Path:
    """The one results directory neuron-bench wrote for a run (``results/<model>_...``)."""
    (found,) = [d for d in (rep_dir / "results").iterdir() if d.is_dir()]
    return found


def _per_core_medians(results: Path) -> dict[str, float]:
    """``{logical core: median NEFF device time, us}``: neuron-bench's ``NCLatency``."""
    data = json.loads((results / "nc_latency_data.json").read_text())["latency_data"]
    return {core: statistics.median(values) for core, values in sorted(data.items())}


def _per_rank_collectives(results: Path) -> dict[str, list[float]]:
    """``{rank: [collective latency, us, ...]}``, every collective of every execution."""
    data = json.loads((results / "cc_latency_data.json").read_text())["latency_data"]
    (node,) = data.values()
    return dict(sorted(node.items()))


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``y = a + b x``: ``(a, b)``."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return my - b * mx, b


def _convert_rows(output_dir: Path, manifest: dict) -> dict:
    """The standalone convert: per-token-count times, entitlement, and the launch fit."""
    hidden = manifest["hidden_size"]
    rows = {}
    for tokens in CONVERT_TOKENS:
        name = f"convert_{tokens}"
        reps = []
        for rep in range(len(manifest["bench"][name])):
            (median,) = _per_core_medians(
                _results(output_dir / "raw" / name / f"rep{rep}")
            ).values()
            reps.append(median)
        nbytes = tokens * hidden * (DTYPE_BYTES["f32"] + DTYPE_BYTES["bf16"])
        rows[str(tokens)] = {
            "bytes": nbytes,
            "neff_device_us_reps": reps,
            "neff_device_us": statistics.median(reps),
            "entitlement_us": nbytes / HBM_BYTES_PER_S * 1e6,
        }
    fixed_us, us_per_byte = _fit(
        [row["bytes"] for row in rows.values()],
        [row["neff_device_us"] for row in rows.values()],
    )
    for row in rows.values():
        row["op_us"] = row["neff_device_us"] - fixed_us
        row["op_over_entitlement"] = row["op_us"] / row["entitlement_us"]
        row["neff_over_entitlement"] = row["neff_device_us"] / row["entitlement_us"]
    bytes_per_s = 1e6 / us_per_byte
    return {
        "shape": f"[T, {hidden}]",
        "by_tokens": rows,
        "fit": {
            "fixed_us": fixed_us,
            "bytes_per_s": bytes_per_s,
            "over_entitlement": HBM_BYTES_PER_S / bytes_per_s,
        },
    }


def _tp_rows(output_dir: Path, manifest: dict) -> dict:
    """Each TP arm: per repeat, the collectives per execution and the per-rank and
    per-core medians; over repeats, the site's collective time and the NEFF time.

    A site's collective time is the collectives per execution times the median latency
    of the rank that waits least (the last to arrive); the NEFF time is the fastest
    core's median.
    """
    out = {}
    for name, record in manifest["compiled"].items():
        if record["ranks"] != SLICE_RANKS or name not in manifest["bench"]:
            continue
        reps = []
        for rep in range(len(manifest["bench"][name])):
            results = _results(output_dir / "raw" / name / f"rep{rep}")
            collectives = _per_rank_collectives(results)
            per_execution = {
                len(v) / manifest["iterations"] for v in collectives.values()
            }
            if per_execution != {record["allreduces"]}:
                raise RuntimeError(
                    f"{name} rep {rep}: {per_execution} collectives per execution, "
                    f"the compile log counts {record['allreduces']}"
                )
            per_rank = {r: statistics.median(v) for r, v in collectives.items()}
            per_core = _per_core_medians(results)
            reps.append(
                {
                    "collectives_per_iter": record["allreduces"],
                    "collective_us_median_per_rank": per_rank,
                    "collective_us_median_min_rank": min(per_rank.values()),
                    "site_collective_us": record["allreduces"] * min(per_rank.values()),
                    "neff_device_us_median_per_core": per_core,
                    "neff_device_us_median_min_core": min(per_core.values()),
                }
            )
        site = statistics.median(r["site_collective_us"] for r in reps)
        neff = statistics.median(r["neff_device_us_median_min_core"] for r in reps)
        out[name] = {
            "flags": record["flags"],
            "reps": reps,
            "collectives_per_iter": record["allreduces"],
            "site_collective_us": site,
            "neff_device_us": neff,
            "outside_collective_us": neff - site,
        }
    return out


def _nccom_rows(output_dir: Path, manifest: dict) -> dict:
    """``nccom-test`` average time per size and dtype, and a fit of the fp32 sizes."""
    rows = {}
    for name, records in manifest["nccom"].items():
        dtype = name.removeprefix("nccom_")
        for rep in range(len(records)):
            report = output_dir / "raw" / name / f"rep{rep}" / "report.json"
            for result in json.loads(report.read_text())["results"]:
                key = f"{dtype}_{result['size(B)'] // MIB}MiB"
                rows.setdefault(key, {"reps": []})["reps"].append(
                    result["time:avg(us)"]
                )
    for row in rows.values():
        row["median"] = statistics.median(row["reps"])
    mib = [nbytes // MIB for nbytes in NCCOM_BYTES]
    fixed_us, us_per_mib = _fit(mib, [rows[f"f32_{m}MiB"]["median"] for m in mib])
    # The bus bandwidth of a ring or RDH all-reduce: 2 (N - 1) / N of the bytes per rank.
    share = 2 * (SLICE_RANKS - 1) / SLICE_RANKS
    return {
        "avg_us": rows,
        "fit_f32": {
            "fixed_us": fixed_us,
            "us_per_MiB": us_per_mib,
            "busbw_GBps": share * MIB / (us_per_mib * 1e-6) / 1e9,
        },
    }


def summarize(output_dir: Path) -> dict:
    """``summary.json`` from ``manifest.json`` and the raw records under ``raw/``."""
    manifest = json.loads((output_dir / "manifest.json").read_text())
    failed = [
        f"{name} rep {rep}"
        for runs in (manifest["bench"], manifest["nccom"])
        for name, records in runs.items()
        for rep, record in enumerate(records)
        if record["rc"] != 0
    ]
    if failed:
        raise RuntimeError(f"runs that did not exit 0: {failed}")
    probes_rows = [
        {
            "probe": name,
            "dtype": record["kind"].removeprefix("allreduce_"),
            "tokens": record["tokens"],
            "ranks": record["ranks"],
            "bytes": record["tokens"]
            * record["hidden"]
            * DTYPE_BYTES[record["kind"].removeprefix("allreduce_")],
            "flags": record["flags"],
            "allreduces": record["allreduces"],
        }
        for name, record in manifest["compiled"].items()
        if record["ranks"] == SERVED_RANKS
    ]
    return {
        "kind": "MEASURED",
        "script": manifest["script"],
        "script_sha256": manifest["script_sha256"],
        "environment": manifest["environment"],
        "window": [manifest.get("bench_start"), manifest.get("bench_end")],
        "hidden_size": manifest["hidden_size"],
        "iterations": manifest["iterations"],
        "warmup": manifest["warmup"],
        "repeats": manifest["repeats"],
        "hbm_bytes_per_s": HBM_BYTES_PER_S,
        "tiling_probes": probes_rows,
        "convert_f32_to_bf16": _convert_rows(output_dir, manifest),
        "tp": _tp_rows(output_dir, manifest),
        "nccom": _nccom_rows(output_dir, manifest),
    }


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    """Compile, time and summarize, or one stage of that; see the module docstring."""
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
        "--stage", choices=("all", "compile", "summarize"), default="all"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    out = args.output_dir.resolve()
    if args.stage == "summarize":
        summary = summarize(out)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"wrote {out / 'summary.json'}")
        return 0
    if args.stage == "all" and not os.environ.get(LEASE_MARKER):
        raise SystemExit(
            f"run under devlease.py slice <name>, which sets {LEASE_MARKER}"
        )
    if args.stage == "all" and os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit(
            "the timed NEFFs are LNC2; the device lease sets NEURON_LOGICAL_NC_CONFIG=2"
        )
    raw = out / "raw"
    if raw.exists() and any(raw.iterdir()):
        # neuron-bench adds a results directory per run; one per repeat is what the
        # summary reads.
        raise SystemExit(f"{raw} is not empty; give a new --output-dir")
    raw.mkdir(parents=True, exist_ok=True)
    hidden = hidden_size()
    timed, compiled = arms(hidden), probes(hidden)
    manifest = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": sys.argv,
        "hidden_size": hidden,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "environment": {
            key: os.environ.get(key)
            for key in (
                LEASE_MARKER,
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
            )
        },
        "compiled": {},
        "bench": {},
        "nccom": {},
    }
    manifest["compile_start"] = _now()
    for arm in timed + compiled:
        manifest["compiled"][arm["name"]] = {
            **arm,
            **compile_arm(arm, raw / arm["name"]),
        }
        print(
            f"compiled {arm['name']}: "
            f"{manifest['compiled'][arm['name']]['allreduces']} all-reduces",
            flush=True,
        )
    manifest["compile_end"] = _now()
    if args.stage == "all":
        manifest["bench_start"] = _now()
        for rep in range(args.repeats):
            for arm in timed:
                record = bench_arm(
                    arm,
                    Path(manifest["compiled"][arm["name"]]["neff"]),
                    raw / arm["name"] / f"rep{rep}",
                    args.iterations,
                    args.warmup,
                )
                manifest["bench"].setdefault(arm["name"], []).append(record)
                print(f"rep {rep} {arm['name']}: rc {record['rc']}", flush=True)
            for dtype in DTYPE_BYTES:
                name = f"nccom_{dtype}"
                record = nccom(
                    dtype, raw / name / f"rep{rep}", args.iterations, args.warmup
                )
                manifest["nccom"].setdefault(name, []).append(record)
                print(f"rep {rep} {name}: rc {record['rc']}", flush=True)
        manifest["bench_end"] = _now()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.stage == "all":
        summary = summarize(out)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
