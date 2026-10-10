# SPDX-License-Identifier: Apache-2.0
"""Device memory a compiled graph needs, read from its NEFF.

A NEFF is a 1024-byte header followed by a gzipped tar. ``kelf-0.json`` names one
subgraph per physical NeuronCore (``sg00``, ``sg01`` at logical-core size 2); each
subgraph's ``def.json`` lists its variables, and the engine files it names hold the
instruction streams (``*.bin``) and the DMA descriptors (``dma`` arrays).

The runtime's own breakdown, printed per physical core at every NEFF load with
``NEURON_RT_LOG_LEVEL=INFO`` (``TDRV:dml_log_dev_neff_mem``), is the reference the
recorded and real-NEFF tests compare against. The figures are from the server log
of the recorded bs=64 @ 8k serve run (rank 0: ND 0 NC 0 / NC 1, runtime 2.34.10, the bs=64 @ 8k
line).

Three kinds of test:

* synthetic: a NEFF built in a temporary directory;
* recorded: ``fixtures/glm53f_bs64x8k_decode_b1_ctx2048.neff``, one decode graph of
  that line reduced to what the reader reads (``fixtures/record_neff_fixture.py``
  says what is kept and how it was made), and beside it the runtime's breakdown for
  that graph (``*.provenance.json``). These two kinds run anywhere;
* real cache: the 15 NEFFs of the recorded run's compile cache. They run only when the
  test-only variable ``VLLM_NEURON_TEST_COMPILE_CACHE_ROOT`` names that cache's
  ``neuron/compile_cache`` directory, and are skipped when it is unset. It is a test
  knob, not a serving knob, so it is read here and is not registered in
  ``vllm_neuron/envs.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import tarfile
from pathlib import Path

import pytest

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
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
# The KV cache tensor that ties a graph to its serve line
# ---------------------------------------------------------------------------


def _pool_layer_bytes(max_num_seqs: int, max_model_len: int) -> int:
    """One latent-pool layer of GLM-5.3-Flash at TP=64, the largest KV cache tensor.

    Blocks of 128 tokens x 512 bf16 latents: per sequence ``max_model_len / 128``
    attention blocks plus one block per KDA group, plus vLLM's null block (the
    derivation is ``test_kv_budget_glm53f.py``'s
    ``test_the_opt_in_line_prices_one_block_per_kda_group_per_request``).
    """
    kda_groups = -(-kv.KDA_LAYERS // kv.MLA_LAYERS)
    blocks = max_num_seqs * (-(-max_model_len // kv.BLOCK_SIZE_TOKENS) + kda_groups) + 1
    return blocks * kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN


#: The bs=64 @ 8k line: 64 x (64 + 4) + 1 = 4353 blocks of 131072 B.
B64_POOL_BYTES = _pool_layer_bytes(64, 8192)
#: The standard line (1 x 4096, one block per KDA group): 1 x (32 + 4) + 1 = 37 blocks.
STD_POOL_BYTES = _pool_layer_bytes(1, 4096)


#: The runtime's size units, which are binary although printed as KB, MB and GB.
RUNTIME_UNIT_BYTES = {"B": 1, "KB": 1024, "MB": MIB, "GB": 1024 * MIB}


def _runtime_mib(printed: str) -> float:
    """A size as the runtime prints it (``"2.781MB"``), in MiB."""
    value, unit = re.fullmatch(r"([0-9.]+)([KMG]?B)", printed).groups()
    return float(value) * RUNTIME_UNIT_BYTES[unit] / MIB


def _mb(value: int) -> float:
    return value / MIB


# ---------------------------------------------------------------------------
# A recorded decode graph of the bs=64 @ 8k line
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"
#: The 1-sequence decode graph at context bucket 2048, recorded from that run's
#: compile cache by ``fixtures/record_neff_fixture.py``.
DECODE_NEFF = FIXTURES / "glm53f_bs64x8k_decode_b1_ctx2048.neff"
DECODE_PROVENANCE = json.loads(
    (FIXTURES / f"{DECODE_NEFF.name}.provenance.json").read_text()
)
#: The runtime's breakdown of that graph on ND 0 NC 0 and NC 1, as printed.
DECODE_RUNTIME = DECODE_PROVENANCE["runtime_breakdown"]["per_core"]
#: The runtime's descriptor rings: IO, spill/reload and its own.
RUNTIME_RING_ITEMS = ("dma rings io", "dma rings spill", "dma rings runtime")


def _fixture_entry(root: Path, key: str) -> Path:
    """A complete compile cache entry holding the recorded decode graph."""
    entry = root / key
    entry.mkdir(parents=True)
    (entry / f"graph_{key}.neff").write_bytes(DECODE_NEFF.read_bytes())
    (entry / ".compilation_complete").write_text("completed:0\n")
    return entry


def test_the_fixture_is_the_recorded_one() -> None:
    """The provenance's digest and size pin the fixture: an edit to it fails here."""
    data = DECODE_NEFF.read_bytes()

    assert len(data) == DECODE_PROVENANCE["bytes"]
    assert hashlib.sha256(data).hexdigest() == DECODE_PROVENANCE["sha256"]


def test_the_recorded_decode_graph_reads_as_the_runtime_accounts_it() -> None:
    """Per core, against the runtime's print for this graph: never less, and close.

    * constants: equal (the runtime prints KB to three decimals);
    * code: every ``*.bin`` member, at most 0.1 MiB over the runtime's figure, the
      tolerance of the prefill graph's check below;
    * rings: never fewer than the runtime's. The analyser's count over-counts a small
      graph's rings (here 2.05 / 1.67 MiB against 1.47 / 1.18 MiB);
    * so the whole charge is bounded instead: what the reader charges the graph is
      within 10% of the runtime's own total for it, which adds the runtime's
      bookkeeping items to code, constants and rings.
    """
    memory = neff_memory.read_neff_memory(DECODE_NEFF)

    assert len(memory.cores) == len(DECODE_RUNTIME)
    for core, printed in zip(memory.cores, (DECODE_RUNTIME["0"], DECODE_RUNTIME["1"])):
        code = _runtime_mib(printed["model code"])
        rings = sum(_runtime_mib(printed[item]) for item in RUNTIME_RING_ITEMS)
        assert core.constant_bytes / 1024 == pytest.approx(
            _runtime_mib(printed["model constants"]) * 1024, abs=0.01
        )
        assert code <= _mb(core.code_bytes) <= code + 0.1
        assert rings <= _mb(core.dma_ring_bytes)
        assert _mb(core.persistent_bytes) <= 1.10 * _runtime_mib(printed["Total"])
    assert B64_POOL_BYTES in memory.input_bytes


def test_a_cache_holding_two_serve_lines_is_split_by_the_kv_input(tmp_path) -> None:
    """The recorded bs=64-line graph and a standard-line graph: each scan keeps its own."""
    _fixture_entry(tmp_path, "b64")
    _cache_entry(tmp_path, "std", kv_bytes=STD_POOL_BYTES)

    b64 = neff_memory.scan_compile_cache(tmp_path, kv_input_bytes=B64_POOL_BYTES)
    std = neff_memory.scan_compile_cache(tmp_path, kv_input_bytes=STD_POOL_BYTES)

    assert [Path(m.path).parent.name for m in b64.graphs] == ["b64"]
    assert [Path(m.path).parent.name for m in std.graphs] == ["std"]
    assert b64.entries == std.entries == 2


# ---------------------------------------------------------------------------
# The real compile cache of the recorded bs=64 @ 8k serve run
# ---------------------------------------------------------------------------

#: Test-only knob (see the module docstring): the recorded run's compile cache.
COMPILE_CACHE_ROOT_ENV = "VLLM_NEURON_TEST_COMPILE_CACHE_ROOT"
needs_compile_cache = pytest.mark.skipif(
    not os.environ.get(COMPILE_CACHE_ROOT_ENV),
    reason=f"{COMPILE_CACHE_ROOT_ENV} unset: it names the tip-b64-C compile cache "
    "(bs=64 @ 8k line, 15 graphs) the runtime figures are from",
)


def compile_cache_root() -> Path:
    """The directory ``VLLM_NEURON_TEST_COMPILE_CACHE_ROOT`` names; it must exist."""
    root = Path(os.environ[COMPILE_CACHE_ROOT_ENV])
    assert root.is_dir(), f"{COMPILE_CACHE_ROOT_ENV}={root} is not a directory"
    return root


def _cache_neff(key: str) -> Path:
    path = compile_cache_root() / key / f"graph_{key}.neff"
    assert path.is_file(), f"{path}: not in {COMPILE_CACHE_ROOT_ENV}; is it the tip-b64-C cache?"
    return path


#: The bs=64 line warms 15 graphs: 1 prefill bucket x 1 segment, 7 batch x 2 ctx decode.
TIP_B64_GRAPHS = 15
#: The runtime log names each NEFF by its compile cache key; this is the prefill graph.
PREFILL_KEY = "7d0ae663f1c6c0f94be5612ed2ce6d4f"
# The runtime's breakdown for the prefill graph (recorded run, ND 0 NC 0 / NC 1).
RT_PREFILL_CODE_MB = (93.147, 70.647)
RT_PREFILL_CONSTANTS_KB = 579.008
RT_PREFILL_RINGS_MB = (0.957 + 36.263 + 0.004, 1.289 + 10.806 + 0.004)
RT_SHARED_SCRATCHPAD_MB = 768.0
# After all 15 loads (recorded run, rank 0): NC 0 14.457 GB total of which 13.378 GB
# tensors, NC 1 0.196 GB. The graphs' share is everything but the tensors.
RT_B64_GRAPH_MB = 14803.97 - 13.378 * 1024 + 200.88


@needs_compile_cache
def test_the_fixture_reads_as_its_source_neff() -> None:
    """The reduction drops nothing the reader reads: same cores, same inputs."""
    source_key = DECODE_PROVENANCE["source_compile_cache_key"]
    source = neff_memory.read_neff_memory(_cache_neff(source_key))
    recorded = neff_memory.read_neff_memory(DECODE_NEFF)

    assert recorded.cores == source.cores
    assert recorded.input_bytes == source.input_bytes


@needs_compile_cache
def test_the_prefill_neff_reads_as_the_runtime_accounts_it() -> None:
    """Code and constants match the runtime to 0.1 MiB; rings within 10%."""
    memory = neff_memory.read_neff_memory(_cache_neff(PREFILL_KEY))

    assert len(memory.cores) == 2
    for core, runtime_code in zip(memory.cores, RT_PREFILL_CODE_MB):
        assert _mb(core.code_bytes) == pytest.approx(runtime_code, abs=0.1)
        assert core.constant_bytes / 1024 == pytest.approx(RT_PREFILL_CONSTANTS_KB, abs=0.01)
    rings = sum(core.dma_ring_bytes for core in memory.cores)
    assert _mb(rings) == pytest.approx(sum(RT_PREFILL_RINGS_MB), rel=0.10)
    # The runtime holds 768 MB of shared scratchpad once this graph is loaded.
    assert _mb(memory.scratchpad_bytes(64 * MIB)) == RT_SHARED_SCRATCHPAD_MB
    assert B64_POOL_BYTES in memory.input_bytes


@needs_compile_cache
def test_the_bs64_line_graph_need_covers_what_the_runtime_held() -> None:
    """15 graphs: the estimate is at or above the runtime's figure, and within 10%."""
    scan = neff_memory.scan_compile_cache(compile_cache_root(), kv_input_bytes=B64_POOL_BYTES)
    need = neff_memory.graph_memory(scan.graphs, page_bytes=64 * MIB)

    assert need.num_graphs == TIP_B64_GRAPHS
    assert _mb(need.scratchpad_bytes) == RT_SHARED_SCRATCHPAD_MB
    assert RT_B64_GRAPH_MB <= _mb(need.total_bytes) <= 1.10 * RT_B64_GRAPH_MB
