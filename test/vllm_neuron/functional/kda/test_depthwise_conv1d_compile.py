# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc compiles the depthwise conv1d kernel with no SBUF alias and no stepped DMA.

Two placement hazards of trn2-1's 2026-10-08 pipeline notice are properties of the
compiled program, not of the kernel source, and the simulator shows neither:

* (c) an instruction whose destination shares SBUF with one of its own sources. The
  kernel avoids it by construction (each accumulation step writes the slot its
  operands do not occupy, in one tile allocated whole), and the compiler's
  post-schedule SBUF pass (``address_rotation_sb``) must not undo that.
* (d) a DMA whose SBUF partition set is not contiguous (stepped-partition access).

The test compiles :func:`depthwise_conv1d` on the CPU (``NEURON_LIBTORCH_CPU_COMPILE=1``)
under LNC2 at the two served prefill shapes, and at one geometry that takes every
general branch (batches, a channel tail, padding on both sides, a stride, five taps,
bfloat16). It compiles each graph's HLO again with the BIR dump after ``lower_sync``,
the last pass that places tiles, checks that the dump compile wrote the same NEFF
programs, and asserts on every core that each instruction from the kernel file
writes SBUF none of its inputs occupy and that each of its DMA descriptors touches
a contiguous run of partitions. It also counts the kernel's ScalarE heads, derived
from the tiling constants, so the checks cannot pass on a dump without the kernel.
The child stops before the executor is built, so no device node is opened, and the
test fails rather than skips when ``neuronx-cc`` is absent.
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

from vllm_neuron.functional.kda import depthwise_conv1d as conv
from vllm_neuron.functional.kda import depthwise_conv1d_kernel as kernel

#: ``name -> (batches, channels, width, taps, (pad_left, pad_right), stride_w, dtype)``.
#: The served calls produce ``2 * COL_TILE`` and ``4 * COL_TILE`` columns from
#: ``taps - 1`` columns of carried history: one and two column tiles per program.
SERVED_TAPS = 4
GRAPHS = {
    "served_1k": (1, 3 * kernel.PARTITION_MAX, 2 * kernel.COL_TILE + SERVED_TAPS - 1,
                  SERVED_TAPS, (0, 0), 1, "float32"),
    "served_2k": (1, 3 * kernel.PARTITION_MAX, 4 * kernel.COL_TILE + SERVED_TAPS - 1,
                  SERVED_TAPS, (0, 0), 1, "float32"),
    "general": (2, kernel.PARTITION_MAX + 8, 300, 5, (2, 1), 2, "bfloat16"),
}
ROW = "depthwise_conv1d_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_KERNEL_FILE = pathlib.Path(kernel.__file__).resolve()
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
#: Dump the BIR after the pass that emits the semaphores, the placement the NEFF runs.
#: The debug mode keeps the work directory the dump is written to; the backend options
#: turn off its other dumps and the performance simulator, which the check does not read.
_DUMP_FLAGS = ("--internal-print-after=lower_sync", "--internal-compiler-debug-mode=all",
               "--internal-backend-options=--print-after-all=false --enable-perf-sim=false")
#: A NEFF is a fixed-size header followed by a gzipped tar of the per-engine programs.
_NEFF_HEADER_BYTES = 1024
#: Bytes per element of the dtypes the kernel's SBUF operands carry.
_ITEM_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


def _child(graph: str) -> None:
    """Compile one of ``GRAPHS``; print the NEFF path."""
    import libtorch_neuronx_lite  # noqa: F401  (registers the backends)
    import libtorch_neuronx_lite.compile.backend as backend
    import torch

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    batches, channels, width, taps, pads, stride_w, dtype = GRAPHS[graph]
    meta = torch.device("meta")
    img = torch.empty((batches, channels, 1, width), dtype=getattr(torch, dtype), device=meta)
    filt = torch.empty((channels, 1, 1, taps), dtype=getattr(torch, dtype), device=meta)

    def run(img, filt):
        return conv.depthwise_conv1d(img, filt, padding=((0, 0), pads), stride=(1, stride_w))

    neff, message = "", ""
    try:
        torch.compile(run, backend="neuron_libtorch", fullgraph=True)(img, filt)
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
    """The SBUF partitions and the byte span one operand touches in its memory location.

    The location holds rows of ``dims[1]`` bytes, its first on partition ``base``; the
    operand's ``ap`` is ``[[partition step, partitions], [step, count], ...]`` and
    ``offset`` its first element, both in elements of the operand's dtype.
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


def _sbuf_boxes(operands: list[dict], memlocs: dict) -> list[tuple[frozenset, range]]:
    return [_box(op, memlocs[op["memref"]]) for op in operands
            if op.get("kind") == "physical_ap" and memlocs.get(op.get("memref"), {}).get("type") == "SB"]


def _from_kernel(inst: dict) -> bool:
    return (inst.get("debug") or {}).get("filename") == str(_KERNEL_FILE)


def _scan(dump: dict) -> dict:
    """Aliased compute instructions, stepped DMA descriptors and the ScalarE head count."""
    function = dump["functions"][0]
    memlocs = {m["name"]: m for a in function["allocations"]
               for m in a.get("memorylocations", [])}
    aliased, stepped, heads = [], [], 0
    for block in function["blocks"]:
        for inst in block["instructions"]:
            if not _from_kernel(inst):
                continue
            heads += inst["opcode"] == "Activation"
            outs = _sbuf_boxes(inst.get("outs", []), memlocs)
            ins = _sbuf_boxes(inst.get("ins", []), memlocs)
            if any(_overlap(o, i) for o in outs for i in ins):
                aliased.append(inst["name"])
    for queue in dump.get("queues", []):
        for block in queue.get("blocks", []):
            for dma in block.get("instructions", []):
                if not _from_kernel(dma):
                    continue
                for sub in dma.get("blocks", []):
                    for desc in sub.get("instructions", []):
                        for parts, _ in _sbuf_boxes(desc.get("ins", []) + desc.get("outs", []),
                                                    memlocs):
                            if sorted(parts) != list(range(min(parts), max(parts) + 1)):
                                stepped.append(desc["name"])
    return {"aliased": aliased, "stepped": stepped, "heads": heads}


def _expected_heads(graph: str, program: int) -> int:
    """ScalarE heads one program issues: one per used chain per tile, from the constants."""
    batches, channels, width, taps, (pad_left, pad_right), stride_w, _ = GRAPHS[graph]
    columns = conv.output_width(width, taps, ((0, 0), (pad_left, pad_right)), (1, stride_w))
    share = -(-columns // conv.LNC_SHARDS)
    owned = max(0, min(share, columns - program * share))
    column_tiles = -(-owned // kernel.column_tile_cap(taps, stride_w))
    channel_tiles = -(-channels // kernel.PARTITION_MAX)
    chains = min(kernel.CHAINS, taps)
    return batches * column_tiles * channel_tiles * chains


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_the_kernel_compiles_with_no_sbuf_alias_and_no_stepped_dma(graph):
    """On every core: no kernel instruction writes its own inputs, no DMA steps partitions."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="depthwise_conv1d_compile_") as scratch:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch,
            NKI_COMPILE_CACHE_URL=os.path.join(scratch, "nki"), PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        assert shutil.which("neuronx-cc", path=environment["PATH"]), "neuronx-cc is not installed"
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "child", graph],
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

        scans = {}
        for path in sorted(glob.glob(str(dump_dir / "nc0*" / "sg00" /
                                          "bir_debug.*.after-lower_sync.before-lower_act.*.json"))):
            core = pathlib.Path(path).parts[-3]
            scans[core] = _scan(json.loads(pathlib.Path(path).read_text()))
    assert sorted(scans) == [f"nc{p:02d}" for p in range(conv.LNC_SHARDS)], sorted(scans)
    for program, core in enumerate(sorted(scans)):
        found = scans[core]
        assert found["heads"] == _expected_heads(graph, program), (core, found["heads"])
        assert not found["aliased"], f"{core}: destination placed on a source: {found['aliased']}"
        assert not found["stepped"], f"{core}: stepped-partition DMA: {found['stepped']}"


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(sys.argv[2])
