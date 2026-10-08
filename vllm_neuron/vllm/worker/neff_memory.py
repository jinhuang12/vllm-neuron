# SPDX-License-Identifier: Apache-2.0
"""Device memory a set of compiled graphs needs, read from their NEFFs.

The KV cache is sized before warmup loads any graph, so the memory the graphs will
take has to be known from the compile cache, not observed. A NEFF says how much:

* ``kelf-0.json`` names one subgraph per physical NeuronCore of the logical core
  (``sg00``, ``sg01`` at logical-core size 2);
* each subgraph's ``def.json`` lists its variables. ``virtual`` variables live in
  the scratchpad at ``backing_variable_off``; ``file`` variables are constants the
  runtime copies to the device; ``input`` variables are the caller's tensors;
* the subgraph's ``*.bin`` members are instruction streams and activation tables,
  copied to the device at load;
* the engine files the definition names (``SP0.json`` and its siblings) hold the
  DMA descriptors, which the runtime turns into descriptor rings at load.

How the runtime holds them, as printed per physical core at every load with
``NEURON_RT_LOG_LEVEL=INFO`` (``TDRV:dml_log_dev_neff_mem``, runtime 2.34.10):

* one shared scratchpad per logical core, the largest any loaded graph needs and
  page-rounded (the runtime library's own message: "Total HBM scratchpad usage is
  the max across loaded NEFFs, rounded up to page size"). Two serve runs pin the
  rounding: 688.29 + 28.07 MiB took 768 MiB, 508.04 + 18.13 MiB took 576 MiB. Both
  are each core's high-water mark rounded up to a 64 MiB page and summed; no other
  page size reproduces both;
* per graph, for as long as it stays loaded (every warmed graph does): its code,
  its constants and its descriptor rings, on each core.

The estimate is checked against the runtime's own figures in
``test/vllm_neuron/worker/test_neff_memory.py``: on the bs=64 @ 8k serve line (15
graphs) the runtime held 1305.8 MiB per rank for graphs and this module estimates
1347 MiB.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MIB = 1024**2

#: Shared scratchpad page when neither runtime variable sets one. Inferred from
#: the two serve runs in the module docstring; 32 MiB and 128 MiB pages each
#: contradict one of them.
DEFAULT_SCRATCHPAD_PAGE_BYTES = 64 * MIB

#: Bytes per DMA descriptor, as the compiler's own artifact analyser counts them
#: (``neuronxcc/starfish/bin/analyze_neff_artifacts.py``).
DMA_DESCRIPTOR_BYTES = 16
#: Descriptors the analyser adds per side of every transfer for the semaphore update.
DMA_SEMAPHORE_DESCRIPTORS = 16

#: Runtime memory per loaded graph beyond its code, constants and rings: the
#: runtime's own ``runtime`` and ``collectives`` items (0.43 MiB and 0.07 MiB per
#: core) plus what each graph's first execution adds on the two cores (2.0 MiB
#: and 0.8 MiB measured across 15 loads). 3.7 MiB measured, rounded up.
RUNTIME_BYTES_PER_GRAPH = 4 * MIB
#: Runtime memory per logical core independent of the graph count: 7.6 MiB
#: before the first load plus 62.4 MiB at the first execution, measured alike on
#: the bs=64 line (15 graphs) and the standard line (3 graphs). 70 MiB, rounded up.
RUNTIME_FIXED_BYTES = 72 * MIB

_ENGINE_FILE_KEYS = ("act", "dve", "pe", "pool", "sp")
_GZIP_MAGIC = b"\x1f\x8b"


class NeffFormatError(ValueError):
    """The file is not a NEFF this reader understands."""


@dataclass(frozen=True)
class CoreMemory:
    """What one subgraph (one physical NeuronCore) needs on the device."""

    scratchpad_bytes: int
    code_bytes: int
    constant_bytes: int
    dma_ring_bytes: int

    @property
    def persistent_bytes(self) -> int:
        """Bytes held for as long as the graph stays loaded."""
        return self.code_bytes + self.constant_bytes + self.dma_ring_bytes


@dataclass(frozen=True)
class NeffMemory:
    """What one compiled graph needs on the device, per physical core."""

    path: str
    cores: tuple[CoreMemory, ...]
    #: Byte sizes of the graph's input tensors (first subgraph's definition).
    input_bytes: frozenset[int]

    @property
    def persistent_bytes(self) -> int:
        return sum(core.persistent_bytes for core in self.cores)

    def scratchpad_bytes(self, page_bytes: int) -> int:
        """The shared scratchpad this graph alone would make the runtime hold."""
        return sum(
            math.ceil(core.scratchpad_bytes / page_bytes) * page_bytes
            for core in self.cores
        )


@dataclass(frozen=True)
class GraphMemory:
    """The device memory a set of loaded graphs takes beyond their input tensors."""

    num_graphs: int
    #: The shared scratchpad: the largest any one graph needs, page-rounded.
    scratchpad_bytes: int
    #: Every graph's code, constants and descriptor rings, summed.
    resident_bytes: int
    #: Runtime bookkeeping: :data:`RUNTIME_FIXED_BYTES` plus
    #: :data:`RUNTIME_BYTES_PER_GRAPH` per graph.
    runtime_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.scratchpad_bytes + self.resident_bytes + self.runtime_bytes


@dataclass(frozen=True)
class CacheScan:
    """The graphs of one served configuration found in a compile cache."""

    root: str
    #: Complete cache entries holding a NEFF.
    entries: int
    #: The entries whose graph takes the configuration's KV cache tensor.
    graphs: tuple[NeffMemory, ...]
    #: Entries whose NEFF could not be read; they belong to no configuration.
    unreadable: tuple[str, ...] = field(default_factory=tuple)


def scratchpad_page_bytes() -> int:
    """The shared scratchpad page, from the runtime's variables when set (in MiB)."""
    for name in ("NEURON_SCRATCHPAD_PAGE_SIZE", "NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB"):
        value = os.environ.get(name)
        if value:
            return int(value) * MIB
    return DEFAULT_SCRATCHPAD_PAGE_BYTES


def _payload_offset(data: bytes, path: str) -> int:
    """Where the gzipped tar starts: the header length the container records."""
    if len(data) >= 24:
        (header_bytes,) = struct.unpack_from("<Q", data, 8)
        if 24 <= header_bytes < len(data) and data[header_bytes : header_bytes + 2] == _GZIP_MAGIC:
            return header_bytes
    raise NeffFormatError(f"{path}: no NEFF header followed by a gzipped tar")


def _read_members(path: str, keep_input: int | None) -> dict[str, bytes] | None:
    """Every member a memory estimate reads: definitions, engine files, binary sizes.

    With ``keep_input`` set, reading stops at the first subgraph's definition when
    the graph has no input of that many bytes, and None is returned: a graph of
    another configuration costs one partial decompression, not a full read.
    """
    data = Path(path).read_bytes()
    offset = _payload_offset(data, path)
    members: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    first_definition: str | None = None
    try:
        with tarfile.open(fileobj=io.BytesIO(data[offset:]), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                sizes[member.name] = member.size
                if not member.name.endswith(".json"):
                    continue
                handle = tar.extractfile(member)
                members[member.name] = handle.read() if handle else b""
                if member.name == "kelf-0.json":
                    graphs = json.loads(members[member.name]).get("graphs") or [{}]
                    first_definition = graphs[0].get("definition")
                if keep_input is not None and member.name == first_definition:
                    if keep_input not in _input_bytes(json.loads(members[member.name])):
                        return None
    except (tarfile.TarError, OSError, EOFError, json.JSONDecodeError) as exc:
        raise NeffFormatError(f"{path}: unreadable NEFF payload ({exc})") from exc
    members["__sizes__"] = json.dumps(sizes).encode()
    return members


def _input_bytes(definition: dict) -> frozenset[int]:
    return frozenset(
        int(var["size"])
        for var in definition.get("var", {}).values()
        if var.get("type") == "input"
    )


def _descriptors(transfer: dict, destination: dict | None = None) -> int:
    """Descriptors one transfer needs on each side, as the analyser counts them."""
    to_sizes = transfer.get("to_sizes") or (destination or {}).get("to_sizes") or [1]
    return (
        math.prod(transfer.get("from_sizes", [1])[1:])
        + DMA_SEMAPHORE_DESCRIPTORS
        + math.prod(to_sizes[1:])
        + DMA_SEMAPHORE_DESCRIPTORS
    )


def _dma_ring_bytes(engine: dict) -> int:
    count = 0
    for queue in engine.get("dma", []):
        for desc in queue.get("desc", []):
            if desc.get("from") is None:
                count += sum(_descriptors(part, desc) for part in desc.get("from_arr", []))
            else:
                count += _descriptors(desc)
    return count * DMA_DESCRIPTOR_BYTES


def _scratchpad_bytes(variables: dict) -> int:
    """High-water mark of the virtual variables, per backing area, summed."""
    areas: dict[str, int] = {}
    for var in variables.values():
        if var.get("type") != "virtual":
            continue
        area = var.get("fabric_path") or var.get("backing_buf") or "main"
        end = int(var.get("backing_variable_off", 0)) + int(var.get("size", 0))
        areas[area] = max(areas.get(area, 0), end)
    return sum(areas.values())


def read_neff_memory(
    path: str | os.PathLike, *, keep_input: int | None = None
) -> NeffMemory | None:
    """Read what a compiled graph needs on the device from its NEFF.

    Args:
        path: The NEFF.
        keep_input: When set, return None without reading further if the graph
            has no input tensor of this many bytes.

    Raises:
        NeffFormatError: the file is not a NEFF or its payload is unreadable.
    """
    path = str(path)
    members = _read_members(path, keep_input)
    if members is None:
        return None
    sizes: dict[str, int] = json.loads(members.pop("__sizes__"))
    try:
        kelf = json.loads(members["kelf-0.json"])
        cores = []
        input_bytes: frozenset[int] = frozenset()
        for index, graph in enumerate(kelf["graphs"]):
            name = graph["name"]
            definition = json.loads(members[graph["definition"]])
            variables = definition["var"]
            if index == 0:
                input_bytes = _input_bytes(definition)
            rings = 0
            for key in _ENGINE_FILE_KEYS:
                engine_file = definition.get(key)
                if engine_file:
                    rings += _dma_ring_bytes(json.loads(members[f"{name}/{engine_file}"]))
            cores.append(
                CoreMemory(
                    scratchpad_bytes=_scratchpad_bytes(variables),
                    code_bytes=sum(
                        size
                        for member, size in sizes.items()
                        if member.startswith(f"{name}/") and member.endswith(".bin")
                    ),
                    constant_bytes=sum(
                        int(var.get("size", 0))
                        for var in variables.values()
                        if var.get("type") == "file"
                    ),
                    dma_ring_bytes=rings,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise NeffFormatError(f"{path}: unexpected NEFF layout ({exc!r})") from exc
    return NeffMemory(path=path, cores=tuple(cores), input_bytes=input_bytes)


def graph_memory(graphs: Iterable[NeffMemory], *, page_bytes: int) -> GraphMemory:
    """The device memory the runtime holds once every graph in ``graphs`` is loaded.

    The scratchpad is shared, so it is the largest any one graph needs rather than a
    sum; everything else a graph holds stays resident for as long as it is loaded,
    so it adds up over the graphs.
    """
    graphs = list(graphs)
    if not graphs:
        return GraphMemory(0, 0, 0, 0)
    return GraphMemory(
        num_graphs=len(graphs),
        scratchpad_bytes=max(graph.scratchpad_bytes(page_bytes) for graph in graphs),
        resident_bytes=sum(graph.persistent_bytes for graph in graphs),
        runtime_bytes=RUNTIME_FIXED_BYTES + RUNTIME_BYTES_PER_GRAPH * len(graphs),
    )


def scan_compile_cache(root: str | os.PathLike, *, kv_input_bytes: int) -> CacheScan:
    """Find the graphs in a compile cache that serve the configuration at hand.

    Every graph of a served configuration takes that configuration's KV cache
    tensors as inputs, so an input of exactly ``kv_input_bytes`` (the largest KV
    cache tensor the configuration allocates) ties a NEFF to it. Graphs another
    configuration left in the same cache take a pool of another size and are not
    counted. Only complete entries (``.compilation_complete`` written) are read.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return CacheScan(root=str(root_path), entries=0, graphs=())
    entries = 0
    graphs: list[NeffMemory] = []
    unreadable: list[str] = []
    for entry in sorted(root_path.iterdir()):
        neff = entry / f"graph_{entry.name}.neff"
        if not (entry / ".compilation_complete").is_file() or not neff.is_file():
            continue
        entries += 1
        try:
            memory = read_neff_memory(neff, keep_input=kv_input_bytes)
        except (NeffFormatError, OSError):
            unreadable.append(str(neff))
            continue
        if memory is not None:
            graphs.append(memory)
    return CacheScan(
        root=str(root_path),
        entries=entries,
        graphs=tuple(graphs),
        unreadable=tuple(unreadable),
    )
