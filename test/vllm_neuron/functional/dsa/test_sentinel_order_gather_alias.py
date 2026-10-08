# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc places the sentinel order's gather destination apart from both gather sources.

The compiler's post-schedule SBUF pass (``address_rotation_sb``) may give a gather's
destination the address of a source whose last read is that gather, as if the gather
were an in-place streaming op. It is not: the device runs ``nc_n_gather`` in pieces that
each read their part of the data and the indices from SBUF, so a piece can read values an
earlier piece wrote (the gather dst/src alias, hazard class (c) of trn2-1's 2026-10-08
pipeline notice; it gave wrong rows on a decode kernel, and wrong ids here at widths of
more than one piece). The simulator gathers in one numpy step and never shows it, and a
compile of the kernel alone does not reproduce the placement at every shape: before the fix
it appeared when the order ran inside the prefill selection graph, and in the order alone at
some row counts and widths.

This test builds three graphs on the CPU (``NEURON_LIBTORCH_CPU_COMPILE=1``, ``GRAPHS``): the
prefill selection graph, the indexer's own ``score_pools`` then ``select_bounded_pools``,
which takes the production dials (``Glm5NextTextConfig``) because the placement is a
property of the whole graph as served; and the order alone at two widths of several gather
pieces, up to the gate's ceiling ``SEARCH_MAX_FREE``, the case in which the alias returned
wrong ids on the device. The rows, widths and gather counts it asserts come from the
kernel's own ``PARTITION_MAX`` and ``SEARCH_MAX_FREE``. It compiles each graph's HLO
again with the BIR dump after ``lower_sync`` (the last pass that places tiles has run),
checks that the dump compile wrote the same NEFF program, and asserts that every gather of
``sentinel_order.py`` on every core writes SBUF that neither its data nor its indices
occupy. The child stops before the executor is built, so no device node is opened.
"""

from __future__ import annotations

import glob
import io
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pytest

from vllm_neuron.functional.dsa.sentinel_order import PARTITION_MAX, SEARCH_MAX_FREE
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

_CONFIG = Glm5NextTextConfig()
#: Elements per partition one piece of ``nc_n_gather`` takes: the instruction runs as
#: ``ceil(elements / 512)`` ISA groups (``nki.isa.nc_n_gather``).
GATHER_PIECE = 512
#: ``name -> (rows, width)`` of the graphs compiled; each dump compile takes under a minute.
#: ``selection``: the prefill selection graph, ``width`` candidate pools. The smallest found to
#: show the alias before the fix: 16 query rows (one rank's share of a 1024-row chunk at
#: 64-way query sharding) over the pools of an 8192-token context.
#: ``order``: the order alone on ``[rows, width]`` ids, a full row tile and a one-row tail two
#: gather pieces wide. Before the fix the tail's gather was placed on its data, and the device
#: returned that row wrong past the first piece.
#: ``ceiling``: the order alone at the widest row the gate admits, a full tile and a tail. Before
#: the fix the full tile's gather was placed on its data at its own ceiling.
GRAPHS = {"selection": (16, 8192 // _CONFIG.index_kpool),
          "order": (PARTITION_MAX + 1, 2 * GATHER_PIECE),
          "ceiling": (PARTITION_MAX + 1, SEARCH_MAX_FREE)}
ROW = "sentinel_gather_alias"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_KERNEL_FILE = _ROOT / "vllm_neuron" / "functional" / "dsa" / "sentinel_order.py"
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
#: Dump the BIR after the pass that emits the semaphores, the placement the NEFF runs. The
#: debug mode keeps the work directory the dump is written to; the backend options turn off
#: its other dumps and the performance simulator, which the check does not read.
_DUMP_FLAGS = ("--internal-print-after=lower_sync", "--internal-compiler-debug-mode=all",
               "--internal-backend-options=--print-after-all=false --enable-perf-sim=false")
#: A NEFF is a fixed-size header followed by a gzipped tar of the per-engine programs.
_NEFF_HEADER_BYTES = 1024
#: Bytes per element of the dtypes the order's gather operands carry: int32 ids, uint32 indices.
_ITEM_BYTES = {"int32": 4, "uint32": 4}


def _child(graph: str, rows: int, width: int) -> None:
    """Compile one of ``GRAPHS``; print the NEFF path."""
    import torch

    import libtorch_neuronx_lite  # noqa: F401  (registers the backends)
    import libtorch_neuronx_lite.compile.backend as backend

    from vllm_neuron.functional.dsa.sentinel_order import dsa_sentinel_order
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    meta = torch.device("meta")
    if graph == "selection":
        indexer = Glm5NextDSAIndexer(Glm5NextTextConfig())

        def run(query, keys, weights, seq_lens):
            return indexer.select_bounded_pools(indexer.score_pools(query, keys, weights),
                                                seq_lens)

        heads, dim = indexer.index_n_heads, indexer.index_head_dim
        args = (torch.empty((rows, heads, dim), dtype=torch.bfloat16, device=meta),
                torch.empty((width, dim), dtype=torch.bfloat16, device=meta),
                torch.empty((rows, heads), dtype=torch.float32, device=meta),
                torch.empty((rows,), dtype=torch.int32, device=meta))
    else:
        run = dsa_sentinel_order
        args = (torch.empty((rows, width), dtype=torch.int32, device=meta),)
    neff, message = "", ""
    try:
        torch.compile(run, backend="neuron_libtorch", fullgraph=True)(*args)
        message = "the compiled graph ran; build_executable was not replaced"
    except BaseException as caught:  # the compiler's failure is a result here
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:1500] or type(caught).__name__
    print(f"{ROW}|neff={neff or 'none'}|diagnostic={message or 'none'}", flush=True)


def _programs(neff: pathlib.Path) -> dict[str, bytes]:
    """The ``*.bin`` members of a NEFF: the instruction streams the device runs."""
    payload = io.BytesIO(neff.read_bytes()[_NEFF_HEADER_BYTES:])
    with tarfile.open(fileobj=payload, mode="r:gz") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()
                if m.isfile() and m.name.endswith(".bin")}


def _box(operand: dict, memloc: dict) -> tuple[frozenset, range]:
    """The SBUF partitions and the byte span one operand reads or writes in its memory location.

    The location holds rows of ``dims[1]`` bytes, its first on partition ``base``; the operand's
    ``ap`` is ``[[partition step, partitions], [step, count], ...]`` and ``offset`` its first
    element, both in elements of the operand's dtype. Two operands of one location (a tile's two
    halves) get disjoint boxes when they touch disjoint bytes.
    """
    item = _ITEM_BYTES[operand["dtype"]]
    row = memloc["dims"][1] // item
    (step, partitions), *inner = operand["ap"]
    first = operand["offset"]
    rows = frozenset((memloc.get("base") or 0) + (first + i * step) // row
                     for i in range(partitions))
    column = first % row
    last = column + sum((count - 1) * stride for stride, count in inner)
    return rows, range(memloc["addr"] + column * item, memloc["addr"] + (last + 1) * item)


def _overlap(a: tuple[frozenset, range], b: tuple[frozenset, range]) -> bool:
    """Whether two SBUF boxes share a partition and a byte."""
    return bool(a[0] & b[0]) and a[1].start < b[1].stop and b[1].start < a[1].stop


def _sentinel_gathers(dump: dict) -> list[dict]:
    """``{name, dst, data, indices}`` boxes of every gather the sentinel order emits."""
    memlocs = {m["name"]: m for a in dump["functions"][0]["allocations"]
               for m in a.get("memorylocations", [])}
    found = []
    for block in dump["functions"][0]["blocks"]:
        for inst in block["instructions"]:
            if inst["opcode"] != "Gather":
                continue
            if (inst.get("debug") or {}).get("filename") != str(_KERNEL_FILE):
                continue
            boxes = [_box(op, memlocs[op["memref"]]) for op in (inst["outs"][0], *inst["ins"][:2])]
            found.append(dict(zip(("name", "dst", "data", "indices"), (inst["name"], *boxes))))
    return found


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_the_sentinel_order_gather_writes_sbuf_its_sources_do_not_occupy(graph):
    """Every sentinel-order gather's destination is disjoint from its data and its indices."""
    rows, width = GRAPHS[graph]
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="sentinel_gather_alias_") as scratch:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch,
            NKI_COMPILE_CACHE_URL=os.path.join(scratch, "nki"), PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        assert shutil.which("neuronx-cc", path=environment["PATH"]), "neuronx-cc is not installed"
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "child", graph, str(rows),
             str(width)],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
        printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
        assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
        neff = pathlib.Path(printed[-1].split("|")[1].split("=", 1)[1])
        assert neff.is_file(), printed[-1]

        # The graph's own neuronx-cc command, its output moved and the dump flags added.
        dump_dir = pathlib.Path(scratch, "dump")
        dump_dir.mkdir()
        command = shlex.split((neff.parent / "command.txt").read_text())
        command[command.index("--output") + 1] = str(dump_dir / neff.name)
        command[command.index("--logfile") + 1] = str(dump_dir / "log-neuron-cc.txt")
        rebuilt = subprocess.run(command + list(_DUMP_FLAGS), cwd=dump_dir, env=environment,
                                 capture_output=True, text=True, timeout=1500, check=False)
        assert rebuilt.returncode == 0, rebuilt.stdout[-3000:]
        served = _programs(neff)
        assert served and _programs(dump_dir / neff.name) == served, (
            "the dump compile wrote another program; its placement is not the served one")

        gathers = {}
        for path in sorted(glob.glob(str(dump_dir / "nc0*" / "sg00" /
                                          "bir_debug.*.after-lower_sync.before-lower_act.*.json"))):
            core = pathlib.Path(path).parts[-3]
            gathers[core] = _sentinel_gathers(json.loads(pathlib.Path(path).read_text()))
    # One gather per row tile, so the check below cannot pass on a dump without them.
    assert sum(map(len, gathers.values())) == -(-rows // PARTITION_MAX), gathers
    aliased = [(core, g["name"], source) for core, found in gathers.items() for g in found
               for source in ("data", "indices") if _overlap(g["dst"], g[source])]
    assert not aliased, f"gather destination placed on its own source: {aliased}"


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
