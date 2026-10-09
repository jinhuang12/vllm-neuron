# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc places every gather of the vendored top-k apart from both of its sources.

The device runs ``nc_n_gather`` in pieces that each read their part of ``data`` and of the
indices from SBUF, so a destination that shares SBUF with a source lets a piece read what an
earlier piece wrote (the gather dst/src alias). The compiler's post-schedule SBUF pass
(``address_rotation_sb``) may give a tile born at a gather the address of a source that dies
there. Before the fix it put the last stage's destination of ``rotational_topk.py``'s gather on
the first bytes of ``indices[:, :offset]`` at the 64k serving width, at an 8-row decode width and
on the sampler's vocab shards. The simulator gathers in one numpy step and never shows it.

This test builds, on the CPU (``NEURON_LIBTORCH_CPU_COMPILE=1``), the graph each caller of the
kernel launches: the DSA seam, ``dsa_topk_select`` at the production ``select_k`` on float32
scores, and the sampler, ``functional.topk``'s ``_topk_nki`` at ``max_top_k`` on bfloat16 logits
of one vocab shard. It compiles the graph's HLO again with the BIR dump after ``lower_sync`` (the
last pass that places tiles has run), checks that the dump compile wrote the same NEFF program,
and asserts that every gather of the two vendored files on every core writes SBUF that neither
its data nor its indices occupy. The footprints are the operands' access patterns, not their
whole tiles, because a gather may read one part of a tile and write another. It also asserts
that no destination is a tile of its own: each rotational-core gather writes into the memory
location of its data and each sort gather into that of its indices, at columns the kernel keeps
apart, so the check does not rest on where this compile placed a fresh tile. The child stops
before the executor is built, so no device node is opened.
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

import nki.language as nl
import pytest

from vllm_neuron.functional.dsa.topk_select import _nki_config
from vllm_neuron.functional.topk import _get_rotational_topk_config
from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
    HW_PARAMS,
)
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig

_CONFIG = Glm5NextTextConfig()
#: Pools the selection keeps (``Glm5NextDSAIndexer.select_k``).
SELECT_K = _CONFIG.index_topk // _CONFIG.index_kpool
#: Query rows of one prefill chunk on the 64k serving line (its ``max_num_batched_tokens``).
PREFILL_CHUNK = 2048
#: Requests in the decode step the alias was seen at.
DECODE_BATCH = 8
#: The sampler's top-k width (``OnDeviceSamplingConfig.max_top_k``'s default).
SAMPLER_K = OnDeviceSamplingConfig().max_top_k
#: Requests in the sampling step the sampler's alias was seen at.
SAMPLER_BATCH = 64
#: The tensor-parallel degree whose vocab shard the sampler's case takes.
SAMPLER_TP = 4
#: Each caller's selection width, input dtype and config builder (the one its graph calls).
_CALLERS = {
    "dsa": (SELECT_K, "float32", _nki_config),
    "sampler": (SAMPLER_K, "bfloat16", _get_rotational_topk_config),
}
#: Geometries the alias was seen at before the fix, as (caller, rows, candidates): one prefill
#: chunk over the pools of a 64k-token context, one decode step over an 8k-token one, and one
#: sampling step over a vocab shard.
CASES = {
    "dsa_prefill_64k": ("dsa", PREFILL_CHUNK, 64 * 1024 // _CONFIG.index_kpool),
    "dsa_decode_b8_8k": ("dsa", DECODE_BATCH, 8 * 1024 // _CONFIG.index_kpool),
    "sampler_b64_tp4": ("sampler", SAMPLER_BATCH, _CONFIG.vocab_size // SAMPLER_TP),
}
#: The served logical core config: two physical cores, one program and one BIR dump each.
LNC = 2
ROW = "rotational_topk_gather_alias"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_KERNEL_DIR = (
    _ROOT / "vllm_neuron" / "functional" / "vendored_kernels" / "rotational_topk"
)
_KERNEL_FILES = {
    str(_KERNEL_DIR / "rotational_topk.py"): "core",
    str(_KERNEL_DIR / "rotational_topk_utils.py"): "sort",
}
#: The source whose memory location each kind of gather writes into: the rotational core
#: gathers into columns of the `indices` tile it reads, `sort` into the other half of the tile
#: that holds its pass indices.
_HOST_SOURCE = {"core": "data", "sort": "indices"}
_DROP = (
    "NKI_SIMULATOR",
    "NKI_PRECISE_FP",
    "VLLM_NEURON_CPU_MODE",
    "NEURON_LIBTORCH_CPU_MODE",
    "NEURON_RT_VISIBLE_CORES",
)
#: Dump the BIR after the pass that emits the semaphores, the placement the NEFF runs. The
#: debug mode keeps the work directory the dump is written to; the backend options turn off
#: its other dumps and the performance simulator, which the check does not read.
_DUMP_FLAGS = (
    "--internal-print-after=lower_sync",
    "--internal-compiler-debug-mode=all",
    "--internal-backend-options=--print-after-all=false --enable-perf-sim=false",
)
#: A NEFF is a fixed-size header followed by a gzipped tar of the per-engine programs.
_NEFF_HEADER_BYTES = 1024
_DTYPE_BYTES = {"float32": 4, "uint32": 4, "int32": 4, "bfloat16": 2, "float16": 2}


def _child(caller: str, rows: int, cands: int) -> None:
    """Compile one caller's top-k graph; print the NEFF path."""
    import libtorch_neuronx_lite  # noqa: F401  (registers the backends)
    import torch
    from libtorch_neuronx_lite.compile import backend

    from vllm_neuron.functional.dsa.topk_select import dsa_topk_select
    from vllm_neuron.functional.topk import _topk_nki

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    k, dtype, _ = _CALLERS[caller]
    graph = {
        "dsa": lambda scores: dsa_topk_select(scores, k),
        "sampler": lambda logits: _topk_nki(logits, k, -1),
    }[caller]
    scores = torch.empty(
        (rows, cands), dtype=getattr(torch, dtype), device=torch.device("meta")
    )
    neff, message = "", ""
    try:
        torch.compile(graph, backend="neuron_libtorch", fullgraph=True)(scores)
        message = "the compiled graph ran; build_executable was not replaced"
    except Exception as caught:  # noqa: BLE001 -- success, too, arrives as an exception
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:1500] or type(caught).__name__
    print(f"{ROW}|neff={neff or 'none'}|diagnostic={message or 'none'}", flush=True)


def _expected_gathers(caller: str, rows: int, cands: int) -> dict[str, int]:
    """Gathers one core runs, from the config the caller's graph builds: per row tile, one per
    stage and gather group in the rotational core, and one per eight-wide pass of the sort."""
    k, dtype, build_config = _CALLERS[caller]
    config = build_config(rows, cands, k, getattr(nl, dtype))
    groups = math.ceil(config.local_top_k_per_stage / HW_PARAMS.gather_group_size)
    passes = math.ceil(config.orig_k / HW_PARAMS.dve_max_alus)
    return {
        "core": config.n_bxs_tiles * config.n_stages * groups,
        "sort": config.n_bxs_tiles * passes,
    }


def _programs(neff: pathlib.Path) -> dict[str, bytes]:
    """The ``*.bin`` members of a NEFF: the instruction streams the device runs."""
    payload = io.BytesIO(neff.read_bytes()[_NEFF_HEADER_BYTES:])
    with tarfile.open(fileobj=payload, mode="r:gz") as archive:
        return {
            m.name: archive.extractfile(m).read()
            for m in archive.getmembers()
            if m.isfile() and m.name.endswith(".bin")
        }


def _footprint(operand: dict, memlocs: dict) -> tuple[frozenset, range]:
    """The SBUF partitions and the byte span one operand's access pattern touches.

    The partition of an element is the tile's start partition plus its offset over the
    per-partition pitch; the byte span runs from the first to one past the last element on a
    partition. A span covers any gap of a strided pattern, so two footprints this calls
    disjoint are disjoint.
    """
    assert operand.get("kind") == "physical_ap", operand
    memloc = memlocs[operand["memref"]]
    assert memloc["type"] == "SB", memloc
    size = _DTYPE_BYTES[operand["dtype"]]
    pitch = memloc["dims"][1] // size
    (step, count), *inner = operand["ap"]
    offset = operand["offset"]
    base = memloc.get("base") or 0
    partitions = frozenset(base + (offset + i * step) // pitch for i in range(count))
    first = memloc["addr"] + (offset % pitch) * size
    last = first + sum((n - 1) * s for s, n in inner) * size + size
    return partitions, range(first, last)


def _overlap(a: tuple[frozenset, range], b: tuple[frozenset, range]) -> bool:
    """Whether two footprints share a partition and a byte."""
    return bool(a[0] & b[0]) and a[1].start < b[1].stop and b[1].start < a[1].stop


def _topk_gathers(dump: dict) -> list[dict]:
    """``{name, kind, line, dst, data, indices, memrefs}`` of every gather the two vendored files
    emit: each operand's footprint, and under ``memrefs`` each operand's memory location."""
    memlocs = {
        m["name"]: m
        for a in dump["functions"][0]["allocations"]
        for m in a.get("memorylocations", [])
    }
    found = []
    for block in dump["functions"][0]["blocks"]:
        for inst in block["instructions"]:
            debug = inst.get("debug") or {}
            if inst["opcode"] != "Gather" or debug.get("filename") not in _KERNEL_FILES:
                continue
            operands = (inst["outs"][0], *inst["ins"][:2])
            spans = [_footprint(op, memlocs) for op in operands]
            found.append(
                dict(
                    zip(("dst", "data", "indices"), spans),
                    memrefs=dict(
                        zip(
                            ("dst", "data", "indices"),
                            (op["memref"] for op in operands),
                        )
                    ),
                    name=inst["name"],
                    kind=_KERNEL_FILES[debug["filename"]],
                    line=debug.get("lineno"),
                )
            )
    return found


@pytest.mark.skipif(
    shutil.which("neuronx-cc") is None
    and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists(),
    reason="neuronx-cc is not installed",
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_topk_gather_writes_sbuf_its_sources_do_not_occupy(case):
    """Every top-k gather's destination is disjoint from its data and its indices, on each core."""
    caller, rows, cands = CASES[case]
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="topk_gather_alias_") as scratch:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1",
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG=str(LNC),
            NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch,
            NKI_COMPILE_CACHE_URL=os.path.join(scratch, "nki"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}",
        )
        done = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "child",
                caller,
                str(rows),
                str(cands),
            ],
            cwd=scratch,
            env=environment,
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        printed = [
            line for line in done.stdout.splitlines() if line.startswith(ROW + "|")
        ]
        assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
        neff = pathlib.Path(printed[-1].split("|")[1].split("=", 1)[1])
        assert neff.is_file(), printed[-1]

        # The graph's own neuronx-cc command, its output moved and the dump flags added.
        dump_dir = pathlib.Path(scratch, "dump")
        dump_dir.mkdir()
        command = shlex.split((neff.parent / "command.txt").read_text())
        command[command.index("--output") + 1] = str(dump_dir / neff.name)
        command[command.index("--logfile") + 1] = str(dump_dir / "log-neuron-cc.txt")
        rebuilt = subprocess.run(
            command + list(_DUMP_FLAGS),
            cwd=dump_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        assert rebuilt.returncode == 0, rebuilt.stdout[-3000:]
        served = _programs(neff)
        assert served and _programs(dump_dir / neff.name) == served, (
            "the dump compile wrote another program; its placement is not the compiled one"
        )

        gathers = {}
        for path in sorted(
            glob.glob(
                str(
                    dump_dir
                    / "nc0*"
                    / "sg00"
                    / "bir_debug.*.after-lower_sync.before-lower_act.*.json"
                )
            )
        ):
            core = pathlib.Path(path).parts[-3]
            gathers[core] = _topk_gathers(json.loads(pathlib.Path(path).read_text()))
    # Every gather the kernel issues is in the dump of each core, so the check below cannot
    # pass on a dump without them.
    expected = _expected_gathers(caller, rows, cands)
    counted = {
        core: {kind: sum(g["kind"] == kind for g in found) for kind in expected}
        for core, found in gathers.items()
    }
    assert counted == {f"nc{c:02d}": expected for c in range(LNC)}, counted
    aliased = [
        (core, g["name"], g["kind"], g["line"], source)
        for core, found in gathers.items()
        for g in found
        for source in ("data", "indices")
        if _overlap(g["dst"], g[source])
    ]
    assert not aliased, f"gather destination placed on its own source: {aliased}"
    # A destination in a memory location of its own is placed by the allocator, which may put
    # it on a source at a geometry this test does not compile.
    renamed = [
        (core, g["name"], g["kind"], g["line"], g["memrefs"])
        for core, found in gathers.items()
        for g in found
        if g["memrefs"]["dst"] != g["memrefs"][_HOST_SOURCE[g["kind"]]]
    ]
    assert not renamed, (
        f"gather destination not in its source's memory location: {renamed}"
    )


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
