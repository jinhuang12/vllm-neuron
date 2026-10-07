# SPDX-License-Identifier: Apache-2.0
"""Device memory a compiled graph needs, read from its NEFF.

A NEFF is a 1024-byte header followed by a gzipped tar. ``kelf-0.json`` names one
subgraph per physical NeuronCore (``sg00``, ``sg01`` at logical-core size 2); each
subgraph's ``def.json`` lists its variables, and the engine files it names hold the
instruction streams (``*.bin``) and the DMA descriptors (``dma`` arrays).

The runtime's own breakdown, printed per physical core at every NEFF load with
``NEURON_RT_LOG_LEVEL=INFO`` (``TDRV:dml_log_dev_neff_mem``), is the reference the
real-NEFF tests below compare against. The figures are from
``/home/ubuntu/glm53f-wt2/gate/runs/tip-b64-C/server.log`` (rank 0, ND 0 NC 0 / NC 1,
runtime 2.34.10, the bs=64 @ 8k line) and ``gate/runs/glue-A/server.log``.

The synthetic tests build a NEFF in a temporary directory, so they run anywhere. The
real-NEFF tests read the gate's warm compile caches and skip where those are absent.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import tarfile
from pathlib import Path

import pytest

from vllm_neuron.vllm.worker import neff_memory

MIB = 1024**2

# ---------------------------------------------------------------------------
# Synthetic NEFF
# ---------------------------------------------------------------------------

#: Two DMA descriptors per subgraph: one 1-D copy ([256] -> one descriptor per side)
#: and one 2-D copy ([64, 8] -> 8 descriptors per side). The estimator counts
#: prod(sizes[1:]) descriptors per side plus 16 for the semaphore update, 16 bytes each.
_DMA = [
    {
        "queue": "qSPIO0",
        "desc": [
            {
                "from": "input0",
                "from_sizes": [256],
                "to": "SB",
                "to_sizes": [256],
            }
        ],
    },
    {
        "queue": "qSPSpillReload0",
        "desc": [
            {
                "from": "t1",
                "from_sizes": [64, 8],
                "to": "SB",
                "to_sizes": [64, 8],
            }
        ],
    },
]
_DMA_DESCRIPTORS = (1 + 16 + 1 + 16) + (8 + 16 + 8 + 16)


def _subgraph_def(*, scratch_vars, inputs, constants) -> dict:
    var = {"SB": {"type": "state-buffer", "var_id": 0}}
    for index, (offset, size) in enumerate(scratch_vars):
        var[f"t{index}"] = {
            "type": "virtual",
            "var_id": 100 + index,
            "size": size,
            "backing_variable_off": offset,
            "ops": [],
        }
    for index, size in enumerate(inputs):
        var[f"input{index}"] = {"type": "input", "var_id": 200 + index, "size": size}
    var["output0"] = {"type": "output", "var_id": 300, "size": 64}
    for index, size in enumerate(constants):
        var[f"const{index}"] = {
            "type": "file",
            "var_id": 400 + index,
            "size": size,
            "file_name": f"const{index}.npy",
        }
    return {
        "name": "definition",
        "act": "Activation0.json",
        "act_instr": "Activation0.bin",
        "dve": "DVE0.json",
        "dve_instr": "DVE0.bin",
        "pe": "PE0.json",
        "pe_instr": "PE0.bin",
        "pool": "Pool0.json",
        "pool_instr": "Pool0.bin",
        "sp": "SP0.json",
        "sp_instr": "SP0.bin",
        "var": var,
    }


def write_neff(path: Path, subgraphs: list[dict], *, header_bytes: int = 1024) -> Path:
    """Write a NEFF container: ``subgraphs`` items carry ``def``, ``bins`` and ``dma``."""
    members: dict[str, bytes] = {}
    kelf_graphs = []
    for index, subgraph in enumerate(subgraphs):
        name = f"sg{index:02d}"
        kelf_graphs.append({"definition": f"{name}/def.json", "name": name})
        members[f"{name}/def.json"] = json.dumps(subgraph["def"]).encode()
        for bin_name, size in subgraph["bins"].items():
            members[f"{name}/{bin_name}"] = b"\0" * size
        for engine in ("Activation0", "DVE0", "PE0", "Pool0"):
            members[f"{name}/{engine}.json"] = json.dumps({"instr": [], "dma": []}).encode()
        members[f"{name}/SP0.json"] = json.dumps(
            {"instr": [], "dma": subgraph["dma"]}
        ).encode()
        # A debug file: in the container, never loaded, never counted.
        members[f"{name}/debug_info_backend_PE.dbg"] = b"\0" * 4096
    members = {
        "kelf-0.json": json.dumps({"graphs": kelf_graphs, "version": "0.5"}).encode(),
        "neff.json": b"{}",
        "info.json": b"{}",
        **members,
    }
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    body = payload.getvalue()
    header = struct.pack("<QQQ", 2, header_bytes, header_bytes + len(body))
    path.write_bytes(header.ljust(header_bytes, b"\0") + body)
    return path


def _two_core_neff(path: Path, *, sg0_scratch: int, sg1_scratch: int, kv_bytes: int):
    """A logical-core-2 NEFF: scratch high-water marks per core, one KV input."""
    return write_neff(
        path,
        [
            {
                "def": _subgraph_def(
                    # Two overlapping variables: the high-water mark is offset + size
                    # of the furthest one, not the sum of the sizes.
                    scratch_vars=[(0, sg0_scratch // 2), (sg0_scratch // 4, sg0_scratch - sg0_scratch // 4)],
                    inputs=[kv_bytes, 4096],
                    constants=[1536, 32768],
                ),
                "bins": {"PE0.bin": 3 * MIB, "SP0.bin": MIB, "act_table_bkt.bin": 4096},
                "dma": _DMA,
            },
            {
                "def": _subgraph_def(
                    scratch_vars=[(0, sg1_scratch)],
                    inputs=[kv_bytes, 4096],
                    constants=[1536],
                ),
                "bins": {"PE0.bin": 2 * MIB},
                "dma": _DMA,
            },
        ],
    )


def test_a_neff_reports_each_cores_scratchpad_code_constants_and_rings(tmp_path) -> None:
    """Per core: high-water mark, every .bin, every file variable, 16 B per descriptor."""
    path = _two_core_neff(
        tmp_path / "graph_a.neff", sg0_scratch=100 * MIB, sg1_scratch=10 * MIB, kv_bytes=7 * MIB
    )

    memory = neff_memory.read_neff_memory(path)

    assert [core.scratchpad_bytes for core in memory.cores] == [100 * MIB, 10 * MIB]
    assert [core.code_bytes for core in memory.cores] == [4 * MIB + 4096, 2 * MIB]
    assert [core.constant_bytes for core in memory.cores] == [1536 + 32768, 1536]
    assert [core.dma_ring_bytes for core in memory.cores] == [16 * _DMA_DESCRIPTORS] * 2
    assert memory.input_bytes == frozenset({7 * MIB, 4096})
    assert memory.persistent_bytes == (
        (4 * MIB + 4096) + 2 * MIB + (1536 + 32768) + 1536 + 2 * 16 * _DMA_DESCRIPTORS
    )


def test_the_header_length_is_read_from_the_container(tmp_path) -> None:
    """A header of another length still finds the tar."""
    path = write_neff(
        tmp_path / "graph_b.neff",
        [
            {
                "def": _subgraph_def(scratch_vars=[(0, MIB)], inputs=[64], constants=[]),
                "bins": {"PE0.bin": 1024},
                "dma": [],
            }
        ],
        header_bytes=2048,
    )

    memory = neff_memory.read_neff_memory(path)

    assert [core.scratchpad_bytes for core in memory.cores] == [MIB]
    assert memory.input_bytes == frozenset({64})


def test_a_file_that_is_not_a_neff_is_refused_by_name(tmp_path) -> None:
    path = tmp_path / "graph_c.neff"
    path.write_bytes(b"not a neff" * 100)

    with pytest.raises(neff_memory.NeffFormatError) as refusal:
        neff_memory.read_neff_memory(path)

    assert str(path) in str(refusal.value)


def test_the_scratchpad_rounds_each_core_up_to_the_page(tmp_path) -> None:
    """508.04 MiB + 18.13 MiB -> 512 + 64 = 576 MiB at the runtime's 64 MiB page."""
    path = _two_core_neff(
        tmp_path / "graph_d.neff",
        sg0_scratch=532_721_668,
        sg1_scratch=19_013_632,
        kv_bytes=MIB,
    )

    memory = neff_memory.read_neff_memory(path)

    assert memory.scratchpad_bytes(64 * MIB) == 576 * MIB
    assert memory.scratchpad_bytes(512 * MIB) == 1024 * MIB


def test_the_graph_need_is_the_largest_scratchpad_plus_every_graphs_resident_bytes(
    tmp_path,
) -> None:
    """The scratchpad is shared (max over graphs); code, constants and rings add up."""
    big = neff_memory.read_neff_memory(
        _two_core_neff(tmp_path / "graph_e.neff", sg0_scratch=300 * MIB, sg1_scratch=MIB, kv_bytes=MIB)
    )
    small = neff_memory.read_neff_memory(
        _two_core_neff(tmp_path / "graph_f.neff", sg0_scratch=100 * MIB, sg1_scratch=MIB, kv_bytes=MIB)
    )

    need = neff_memory.graph_memory([big, small], page_bytes=64 * MIB)

    assert need.scratchpad_bytes == (320 + 64) * MIB
    assert need.resident_bytes == big.persistent_bytes + small.persistent_bytes
    assert need.runtime_bytes == (
        neff_memory.RUNTIME_FIXED_BYTES + 2 * neff_memory.RUNTIME_BYTES_PER_GRAPH
    )
    assert need.total_bytes == need.scratchpad_bytes + need.resident_bytes + need.runtime_bytes
    assert need.num_graphs == 2


def test_no_graph_needs_no_memory() -> None:
    need = neff_memory.graph_memory([], page_bytes=64 * MIB)

    assert need.total_bytes == 0
    assert need.num_graphs == 0


@pytest.mark.parametrize(
    ("env", "value", "expected_mib"),
    [
        (None, None, 64),
        ("NEURON_SCRATCHPAD_PAGE_SIZE", "512", 512),
        ("NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB", "128", 128),
    ],
)
def test_the_page_size_follows_the_runtime_variables(monkeypatch, env, value, expected_mib) -> None:
    monkeypatch.delenv("NEURON_SCRATCHPAD_PAGE_SIZE", raising=False)
    monkeypatch.delenv("NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB", raising=False)
    if env is not None:
        monkeypatch.setenv(env, value)

    assert neff_memory.scratchpad_page_bytes() == expected_mib * MIB


def _cache_entry(root: Path, key: str, *, kv_bytes: int, complete: bool = True) -> Path:
    entry = root / key
    entry.mkdir(parents=True)
    _two_core_neff(entry / f"graph_{key}.neff", sg0_scratch=50 * MIB, sg1_scratch=MIB, kv_bytes=kv_bytes)
    if complete:
        (entry / ".compilation_complete").write_text("completed:0\n")
    return entry


def test_the_cache_scan_keeps_complete_graphs_that_take_the_kv_cache(tmp_path) -> None:
    """Only finished entries whose NEFF has an input of the KV cache tensor's size."""
    served = 570_556_416
    _cache_entry(tmp_path, "aaa", kv_bytes=served)
    _cache_entry(tmp_path, "bbb", kv_bytes=served)
    _cache_entry(tmp_path, "ccc", kv_bytes=21_102_592)  # another configuration's pool
    _cache_entry(tmp_path, "ddd", kv_bytes=served, complete=False)  # compile still running
    (tmp_path / "nki").mkdir()  # the NKI kernel cache lives beside the graphs
    broken = tmp_path / "eee"
    broken.mkdir()
    (broken / "graph_eee.neff").write_bytes(b"truncated")
    (broken / ".compilation_complete").write_text("completed:0\n")

    scan = neff_memory.scan_compile_cache(tmp_path, kv_input_bytes=served)

    assert sorted(Path(m.path).parent.name for m in scan.graphs) == ["aaa", "bbb"]
    assert scan.entries == 4  # complete entries with a NEFF, the broken one included
    assert [Path(p).parent.name for p in scan.unreadable] == ["eee"]


def test_a_missing_cache_directory_is_an_empty_scan(tmp_path) -> None:
    scan = neff_memory.scan_compile_cache(tmp_path / "absent", kv_input_bytes=1)

    assert scan.graphs == ()
    assert scan.entries == 0


# ---------------------------------------------------------------------------
# Real NEFFs from the gate's warm compile caches
# ---------------------------------------------------------------------------

GATE_ROOT = Path("/home/ubuntu/glm53f-wt2")
#: bs=64 @ 8k serve line (15 graphs: prefill 1024/kv8192 + 7 batch x 2 ctx decode).
TIP_B64_CACHE = GATE_ROOT / "gate-cache-tip-b64/neuron/compile_cache"
#: Standard line plus the bs=64 line (21 graphs, two KV pool sizes).
TIP_STD_CACHE = GATE_ROOT / "gate-cache-tip-std/neuron/compile_cache"
#: Standard line, fused-glue branch (3 graphs).
GLUE_CACHE = GATE_ROOT / "gate-cache-glue/neuron/compile_cache"
PREFILL_KEY = "7d0ae663f1c6c0f94be5612ed2ce6d4f"

#: One latent-pool layer at 64 x 8192 (4353 blocks x 131072 B) and at 1 x 4096
#: (161 blocks): the graph input that ties a NEFF to its serve line.
B64_POOL_BYTES = 4353 * 131072
STD_POOL_BYTES = 161 * 131072

# The runtime's breakdown for the prefill graph (tip-b64-C, ND 0 NC 0 / NC 1).
RT_PREFILL_CODE_MB = (93.147, 70.647)
RT_PREFILL_CONSTANTS_KB = 579.008
RT_PREFILL_RINGS_MB = (0.957 + 36.263 + 0.004, 1.289 + 10.806 + 0.004)
RT_SHARED_SCRATCHPAD_MB = 768.0
# After all 15 loads (tip-b64-C rank 0): NC 0 14.457 GB total of which 13.378 GB
# tensors, NC 1 0.196 GB. The graphs' share is everything but the tensors.
RT_B64_GRAPH_MB = 14803.97 - 13.378 * 1024 + 200.88
# glue-A rank 0 after its 3 loads: NC 0 8.027 GB of which 7.269 GB tensors, NC 1 0.087 GB;
# shared scratchpad 576 MB.
RT_GLUE_GRAPH_MB = (8.027 - 7.269) * 1024 + 0.087 * 1024
RT_GLUE_SHARED_SCRATCHPAD_MB = 576.0

needs_tip_b64 = pytest.mark.skipif(
    not (TIP_B64_CACHE / PREFILL_KEY).is_dir(), reason="gate compile cache not on this host"
)


def _mb(value: int) -> float:
    return value / MIB


@needs_tip_b64
def test_the_prefill_neff_reads_as_the_runtime_accounts_it() -> None:
    """Code and constants match the runtime to 0.1 MiB; rings within 10%."""
    memory = neff_memory.read_neff_memory(
        TIP_B64_CACHE / PREFILL_KEY / f"graph_{PREFILL_KEY}.neff"
    )

    assert len(memory.cores) == 2
    for core, runtime_code in zip(memory.cores, RT_PREFILL_CODE_MB):
        assert _mb(core.code_bytes) == pytest.approx(runtime_code, abs=0.1)
        assert core.constant_bytes / 1024 == pytest.approx(RT_PREFILL_CONSTANTS_KB, abs=0.01)
    rings = sum(core.dma_ring_bytes for core in memory.cores)
    assert _mb(rings) == pytest.approx(sum(RT_PREFILL_RINGS_MB), rel=0.10)
    # The runtime holds 768 MB of shared scratchpad once this graph is loaded.
    assert _mb(memory.scratchpad_bytes(64 * MIB)) == RT_SHARED_SCRATCHPAD_MB
    assert B64_POOL_BYTES in memory.input_bytes


@needs_tip_b64
def test_the_bs64_line_graph_need_covers_what_the_runtime_held() -> None:
    """15 graphs: the estimate is at or above the runtime's figure, and within 10%."""
    scan = neff_memory.scan_compile_cache(TIP_B64_CACHE, kv_input_bytes=B64_POOL_BYTES)
    need = neff_memory.graph_memory(scan.graphs, page_bytes=64 * MIB)

    assert need.num_graphs == 15
    assert _mb(need.scratchpad_bytes) == RT_SHARED_SCRATCHPAD_MB
    assert RT_B64_GRAPH_MB <= _mb(need.total_bytes) <= 1.10 * RT_B64_GRAPH_MB


@pytest.mark.skipif(not GLUE_CACHE.is_dir(), reason="gate compile cache not on this host")
def test_a_second_serve_run_is_covered_too() -> None:
    """glue-A's 3 graphs: 576 MB shared scratchpad and the total within 10%."""
    scan = neff_memory.scan_compile_cache(GLUE_CACHE, kv_input_bytes=STD_POOL_BYTES)
    need = neff_memory.graph_memory(scan.graphs, page_bytes=64 * MIB)

    assert need.num_graphs == 3
    assert _mb(need.scratchpad_bytes) == RT_GLUE_SHARED_SCRATCHPAD_MB
    assert RT_GLUE_GRAPH_MB <= _mb(need.total_bytes) <= 1.10 * RT_GLUE_GRAPH_MB


@pytest.mark.skipif(not TIP_STD_CACHE.is_dir(), reason="gate compile cache not on this host")
def test_a_cache_holding_two_serve_lines_is_split_by_the_kv_input() -> None:
    """The 21-graph cache: the bs=64 line's 15 graphs, the standard line's own."""
    b64 = neff_memory.scan_compile_cache(TIP_STD_CACHE, kv_input_bytes=B64_POOL_BYTES)
    std = neff_memory.scan_compile_cache(TIP_STD_CACHE, kv_input_bytes=STD_POOL_BYTES)

    assert len(b64.graphs) == 15
    assert len(std.graphs) == b64.entries - 15
    assert not {m.path for m in b64.graphs} & {m.path for m in std.graphs}
