# SPDX-License-Identifier: Apache-2.0
"""Record a compiled graph as a small NEFF fixture for ``test_neff_memory.py``.

Run from the repository root, on a host that holds the source NEFF and the server
log of a serve run that loaded it with ``NEURON_RT_LOG_LEVEL=INFO``::

    python test/vllm_neuron/worker/fixtures/record_neff_fixture.py \\
        --neff <compile cache>/<key>/graph_<key>.neff \\
        --server-log <serve run>/server.log --nd 0 \\
        --graph "<what the graph is>" --serve-run "<serve run name>" \\
        --out test/vllm_neuron/worker/fixtures/<name>.neff

The fixture is a NEFF container that keeps what
:func:`vllm_neuron.vllm.worker.neff_memory.read_neff_memory` reads, and nothing else:

* a 1024-byte header carrying the format version, the header length and the payload
  length (the source header also carries a digest and the compile host's output
  path; neither is read, so neither is kept);
* ``kelf-0.json``, byte for byte;
* every subgraph definition (``sgNN/def.json``) with its top-level string fields
  (the engine file names) and every variable reduced to :data:`VARIABLE_FIELDS`;
* every engine file the definitions name, reduced to its DMA queues, each
  descriptor to :data:`DESCRIPTOR_FIELDS` (instruction lists are dropped);
* every ``*.bin`` member at its size, zero-filled: the reader counts code by
  member size, and zeros compress about a thousandfold.

Every other member (constants' data, debug info, ...) is dropped. Member order
is the source's; tar and gzip metadata are fixed, so the output is reproducible.

Next to the fixture, ``<name>.neff.provenance.json`` records the source, the
fixture's sha256 and byte count, and the runtime's own per-NEFF breakdown for the
source NEFF (``TDRV:dml_log_dev_neff_mem``) on ``ND <nd>`` NC 0 and NC 1 (the two
physical cores of logical core 0), as printed, which the tests compare against.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import struct
import tarfile
from pathlib import Path

#: The NEFF header: little-endian version, header length, payload length.
HEADER_FORMAT = "<QQQ"
HEADER_BYTES = 1024
#: The variable fields the reader reads (``neff_memory._scratchpad_bytes`` and the
#: constant and input sums).
VARIABLE_FIELDS = ("type", "size", "backing_variable_off", "fabric_path", "backing_buf")
#: The descriptor fields the reader's descriptor count reads (``from`` marks a
#: single transfer; ``from_arr`` holds the parts of a gathered one).
DESCRIPTOR_FIELDS = ("from", "from_sizes", "to_sizes", "from_arr")
#: The physical cores of one logical core at logical-core size 2.
PHYSICAL_CORES = (0, 1)

_BLOCK_HEAD = re.compile(r"\[ND\s*(\d+):NC\s*(\d+)\] Current Usage Total: (.*)")
_BLOCK_NEFF = re.compile(r"Per NEFF memory usage breakdown for \[.*graph_([0-9a-f]+)\.neff\]")
_BLOCK_ITEM = re.compile(r"\t\*? ?([A-Za-z ]+): (.*)")


def _reduce(value: dict, fields: tuple[str, ...]) -> dict:
    return {key: value[key] for key in fields if key in value}


def _reduce_definition(definition: dict) -> dict:
    reduced = {key: value for key, value in definition.items() if isinstance(value, str)}
    reduced["var"] = {
        name: _reduce(var, VARIABLE_FIELDS) for name, var in definition["var"].items()
    }
    return reduced


def _reduce_engine(engine: dict) -> dict:
    queues = []
    for queue in engine.get("dma", []):
        descriptors = []
        for desc in queue.get("desc", []):
            kept = _reduce(desc, DESCRIPTOR_FIELDS)
            if "from_arr" in kept:
                kept["from_arr"] = [_reduce(part, DESCRIPTOR_FIELDS) for part in kept["from_arr"]]
            descriptors.append(kept)
        queues.append({"queue": queue.get("queue"), "desc": descriptors})
    return {"dma": queues}


def _dumps(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def reduce_neff(source: Path) -> bytes:
    """The fixture container for the NEFF at ``source``."""
    data = source.read_bytes()
    (header_bytes,) = struct.unpack_from("<Q", data, 8)
    with tarfile.open(fileobj=io.BytesIO(data[header_bytes:]), mode="r:gz") as tar:
        members = [(m, tar.extractfile(m).read()) for m in tar if m.isfile()]
    by_name = {member.name: payload for member, payload in members}
    kelf = json.loads(by_name["kelf-0.json"])
    definitions = {}
    engines = set()
    for graph in kelf["graphs"]:
        definition = json.loads(by_name[graph["definition"]])
        definitions[graph["definition"]] = _reduce_definition(definition)
        subgraph = graph["definition"].rsplit("/", 1)[0]
        for value in definition.values():
            if isinstance(value, str) and value.endswith(".json"):
                engines.add(f"{subgraph}/{value}")
    subgraphs = tuple(graph["definition"].rsplit("/", 1)[0] + "/" for graph in kelf["graphs"])

    kept: list[tuple[str, bytes]] = []
    for member, payload in members:
        name = member.name
        if name == "kelf-0.json":
            kept.append((name, payload))
        elif name in definitions:
            kept.append((name, _dumps(definitions[name])))
        elif name in engines:
            kept.append((name, _dumps(_reduce_engine(json.loads(payload)))))
        elif name.endswith(".bin") and name.startswith(subgraphs):
            kept.append((name, bytes(member.size)))

    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w", format=tarfile.USTAR_FORMAT) as out:
        for name, payload in kept:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            out.addfile(info, io.BytesIO(payload))
    compressed = gzip.compress(body.getvalue(), compresslevel=9, mtime=0)
    header = struct.pack(HEADER_FORMAT, 2, HEADER_BYTES, len(compressed))
    return header.ljust(HEADER_BYTES, b"\0") + compressed


def runtime_breakdown(server_log: Path, key: str, nd: int) -> dict[str, dict[str, str]]:
    """The runtime's first per-NEFF breakdown of graph ``key`` on ND ``nd``, per core."""
    found: dict[str, dict[str, str]] = {}
    block: dict | None = None
    with server_log.open(errors="replace") as log:
        for line in log:
            head = _BLOCK_HEAD.match(line)
            if head:
                nd, nc = int(head.group(1)), int(head.group(2))
                block = {"nd": nd, "nc": nc, "neff": None, "items": {}}
                continue
            if block is None:
                continue
            if not line.strip():
                if block["neff"] == key and block["nd"] == nd and block["nc"] in PHYSICAL_CORES:
                    found.setdefault(str(block["nc"]), block["items"])
                block = None
                if len(found) == len(PHYSICAL_CORES):
                    break
                continue
            neff = _BLOCK_NEFF.match(line)
            if neff:
                block["neff"] = neff.group(1)
                continue
            item = _BLOCK_ITEM.match(line)
            if item and block["neff"] is not None:
                block["items"][item.group(1).strip()] = item.group(2).strip()
    missing = [nc for nc in map(str, PHYSICAL_CORES) if nc not in found]
    if missing:
        raise SystemExit(f"{server_log}: no breakdown of graph {key} on ND {nd} NC {missing}")
    return dict(sorted(found.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--neff", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--nd", type=int, default=0)
    parser.add_argument("--graph", required=True, help="what the graph is, for the provenance")
    parser.add_argument("--serve-run", required=True, help="the serve run, for the provenance")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    key = args.neff.parent.name
    fixture = reduce_neff(args.neff)
    args.out.write_bytes(fixture)
    provenance = {
        "fixture": args.out.name,
        "fixture_form": "reduced NEFF; see record_neff_fixture.py for what is kept",
        "graph": args.graph,
        "source_compile_cache_key": key,
        "source_bytes": args.neff.stat().st_size,
        "serve_run": args.serve_run,
        "sha256": hashlib.sha256(fixture).hexdigest(),
        "bytes": len(fixture),
        "runtime_breakdown": {
            "source": f"{args.serve_run} server log, TDRV:dml_log_dev_neff_mem, ND {args.nd}, "
            "first load of the graph; units as printed (binary, MB = MiB)",
            "per_core": runtime_breakdown(args.server_log, key, args.nd),
        },
        "produced_by": "python test/vllm_neuron/worker/fixtures/record_neff_fixture.py "
        f"--neff <compile cache>/{key}/graph_{key}.neff "
        f"--server-log <{args.serve_run}>/server.log --nd {args.nd} "
        f"--graph {json.dumps(args.graph)} --serve-run {args.serve_run} "
        f"--out test/vllm_neuron/worker/fixtures/{args.out.name}",
    }
    provenance_path = args.out.with_name(args.out.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{args.out}: {len(fixture)} B; {provenance_path}")


if __name__ == "__main__":
    main()
