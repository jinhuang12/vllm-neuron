# SPDX-License-Identifier: Apache-2.0
"""The DCP candidate merge past one unfolded tile's width: rows folded over more partitions.

``dsa_dcp_merge_select`` (``vllm_neuron/functional/dsa/dcp_candidate_merge.py``) merges every
rank's local top-``k`` into the selection one rank makes over the whole pool axis. An unfolded
128-row tile holds a row's ``CP * k`` candidates on one partition, at most ``MAX_CANDIDATES``
of them. A wider row is served by tiles of fewer rows (``tile_rows``), each row folded over
enough partitions that every partition's columns fit. The checks here:

* the tile plan: up to ``MAX_CANDIDATES`` it is the plan of 128-row tiles; past it, every
  tile of every row count holds at most what fits a partition, and the host refuses only a
  width no fold serves;
* the kernel on the NKI simulator, under the LNC2 grid, against a CPU reference written here
  from the rule (:func:`reference_merge`), bit for bit in ids and slot order, at 4096
  candidates a row (8 ranks x 512, an unfolded tile), 4250 (2 ranks x 2125, the first width
  past ``MAX_CANDIDATES``) and 8192 (16 ranks x 512, the DCP16 decode and prefill width),
  with duplicates, ties at the ``k``-th value, pads, short and empty rows, and local pool ids
  at the top of a 65,536-token rank, at a decode row and at the prefill chunks;
* neuronx-cc, in child processes that open no device, builds the kernel at 8192 candidates
  and at the widest folded partition ``tile_fits`` admits with one SBUF allocation pass and no
  spill, every matmul a single fp32 instruction outside the double-row mode.

Run with ``VLLM_NEURON_CPU_MODE=1`` (``test/conftest.py`` pins it) and ``NKI_SIMULATOR=1``
(this directory's ``conftest.py`` defaults it).
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import pytest
import torch

from vllm_neuron.functional.dsa import dcp_candidate_merge as merge
from vllm_neuron.functional.dsa import decode_select
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, BOUND_FILL_MARK
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

_CONFIG = Glm5NextTextConfig()
#: Pools each rank selects and hands the merge: ``index_topk // index_kpool``.
SELECT_K = int(_CONFIG.index_topk) // int(_CONFIG.index_kpool)
#: Pools per 128-token ownership block, the kernel's ``pools_per_block``.
PPB = 128 // int(_CONFIG.index_kpool)
#: Pools a rank holds at most: the decode indexer's widest candidate axis (65,536 tokens).
LOCAL_POOLS = int(decode_select.MAX_DECODE_INDEX_CANDIDATES)
#: ``(CP, k)`` at the three widths: an unfolded tile, the first width past it, DCP16.
UNFOLDED = (8, SELECT_K)
FIRST_PAST = (2, 2125)
DCP16 = (16, SELECT_K)
#: Rows that select each tile layout under the LNC2 grid (two programs, rows split evenly).
ROW_COUNTS = (1, 8, 64, 65, 128, 129)
#: The 1024-token prefill chunk, served up to 8 ranks.
CHUNK_ROWS = 1024
#: The smallest prefill chunk at 16 ranks: one ownership stride (16 ranks x 128-token blocks),
#: the bucket alignment ``bucket_utils`` holds a DCP run to.
DCP16_CHUNK_ROWS = DCP16[0] * 128
#: ``(CP, k)`` of the widest 2-rank row a 64-row tile serves: folded over 2 partitions, each
#: holding the widest column count whose two tiles, with the fold's operands, still fit.
FOLDED_EDGE = (2, max(w for w in range(1, merge.MAX_CANDIDATES + 1) if merge.tile_fits(w, True)))
MARK32 = np.float32(BOUND_FILL_MARK)


# ---------------------------------------------------------------------------------------------
# Reference, from the rule
# ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def global_table(cp: int, ppb: int = PPB, local_pools: int = LOCAL_POOLS) -> np.ndarray:
    """``[cp, local_pools]`` int64: each rank's local pool as a global pool id.

    Walks the global pool axis one ownership block of ``ppb`` pools at a time: block ``b``
    belongs to rank ``b % cp`` and is that rank's next local block.
    """
    table = np.full((cp, local_pools), -1, dtype=np.int64)
    filled = [0] * cp
    for block in range(cp * local_pools // ppb):
        rank = block % cp
        table[rank, filled[rank]:filled[rank] + ppb] = np.arange(block * ppb, (block + 1) * ppb)
        filled[rank] += ppb
    return table


def reference_merge(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``[CP, R, k]`` int32: every rank's merge output, by the rule.

    Per row, the ``CP * k`` candidates in the order real before pad, value descending
    (``-0.0`` equal to ``+0.0``), global pool id ascending; the first ``k`` real ones are
    selected. A candidate is real when its local id is not ``-1`` and its fp32 value is above
    ``BOUND_FILL_MARK``. Rank ``r`` gets its own selected local ids in its slot order, ``-1``
    in every other slot.
    """
    cp, rows, k = (int(d) for d in values.shape)
    v = values.numpy().astype(np.float32).transpose(1, 0, 2).reshape(rows, cp * k)
    lid = ids.numpy().astype(np.int64).transpose(1, 0, 2).reshape(rows, cp * k)
    real = (lid >= 0) & (v > MARK32)
    rank = np.broadcast_to(np.repeat(np.arange(cp), k), (rows, cp * k))
    gid = global_table(cp)[rank, np.clip(lid, 0, None)]
    order = np.lexsort((np.where(real, gid, np.iinfo(np.int64).max),
                        np.where(real, -v.astype(np.float64), 0.0),
                        np.where(real, 0, 1)), axis=1)
    taken = np.zeros((rows, cp * k), dtype=bool)
    np.put_along_axis(taken, order[:, :k], True, axis=1)
    out = np.where(taken & real, lid, -1).astype(np.int32)
    return torch.from_numpy(np.ascontiguousarray(out.reshape(rows, cp, k).transpose(1, 0, 2)))


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

PATTERNS = ("random", "duplicates", "boundary_tie", "all_equal", "max_ids", "signed_zero",
            "pads", "short", "second_core_ties")


def _rng(*key) -> np.random.Generator:
    """A generator seeded by ``key`` alone, the same in every process."""
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _local_ids(g: np.random.Generator, cp: int, rows: int, k: int, top: bool) -> np.ndarray:
    """``[cp, rows, k]`` int32 distinct local ids a rank-row; from the top of the range when
    ``top``, so global ids reach ``cp * LOCAL_POOLS - 1``."""
    ids = np.empty((cp, rows, k), dtype=np.int32)
    for r in range(cp):
        for i in range(rows):
            if top:
                ids[r, i] = g.permutation(np.arange(LOCAL_POOLS - k, LOCAL_POOLS))
            else:
                ids[r, i] = g.choice(LOCAL_POOLS, size=k, replace=False)
    return ids


def build_case(pattern: str, cp: int, rows: int, k: int, seed: int = 0) -> dict:
    """One merge input of ``pattern``: ``values [cp, rows, k]`` fp32, ``ids`` int32."""
    g = _rng(pattern, cp, rows, k, seed)
    ids = _local_ids(g, cp, rows, k, top=pattern == "max_ids")
    # bf16-rounded scores tie, as the indexer's do.
    values = torch.from_numpy(g.standard_normal((cp, rows, k)).astype(np.float32))
    values = values.to(torch.bfloat16).float().numpy()
    if pattern == "duplicates":
        # Few distinct values: every row has many equal candidates within and across ranks.
        values = g.integers(0, 6, size=(cp, rows, k)).astype(np.float32)
    elif pattern == "boundary_tie":
        # Per row: k - 2 candidates above 0.5, six at 0.5 (three in the first rank's input,
        # three in the last's), the rest below: the k-th value is 0.5 and it ties in excess.
        flat = np.full((rows, cp * k), -1.5, dtype=np.float32)
        for i in range(rows):
            ties = np.concatenate([g.choice(k, size=3, replace=False),
                                   (cp - 1) * k + g.choice(k, size=3, replace=False)])
            rest = np.setdiff1d(np.arange(cp * k), ties)
            above = g.choice(rest, size=k - 2, replace=False)
            flat[i, above] = 1.0 + g.random(k - 2).astype(np.float32)
            flat[i, ties] = 0.5
        values = flat.reshape(rows, cp, k).transpose(1, 0, 2)
    elif pattern == "all_equal":
        values[:] = 1.0
    elif pattern == "signed_zero":
        values[:] = np.where(g.random((cp, rows, k)) < 0.5, np.float32(0.0), np.float32(-0.0))
        values[0, :, : k // 2] = 1.0
    elif pattern == "pads":
        pad = g.random((cp, rows, k))
        ids = np.where(pad < 0.2, np.int32(-1), ids)
        values = np.where(pad < 0.1, np.float32(1.0e4), values)  # pad by id, value above all
        values = np.where((pad >= 0.2) & (pad < 0.3), np.float32(BOUND_FILL), values)
        values = np.where((pad >= 0.3) & (pad < 0.32), MARK32, values)
    elif pattern == "short":
        # k // 2 real candidates a row, spread over the ranks; the last row has none.
        keep = np.zeros((rows, cp * k), dtype=bool)
        for i in range(rows - 1):
            keep[i, g.choice(cp * k, size=k // 2, replace=False)] = True
        keep = keep.reshape(rows, cp, k).transpose(1, 0, 2)
        ids = np.where(keep, ids, np.int32(-1))
    elif pattern == "second_core_ties":
        # Ties only in the rows the second program of an LNC2 pair owns.
        share = (rows + 1) // 2
        values[:, share:, :] = g.integers(0, 3, size=(cp, rows - share, k)).astype(np.float32)
    return {"values": torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)),
            "ids": torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32)),
            "pattern": pattern, "cp": cp, "rows": rows, "k": k}


def _pattern_applies(pattern: str, rows: int) -> bool:
    return pattern != "second_core_ties" or rows >= 2


# ---------------------------------------------------------------------------------------------
# CPU: the reference and the tile plan
# ---------------------------------------------------------------------------------------------


def test_the_reference_agrees_with_the_module_oracle() -> None:
    for cp, k in (UNFOLDED, FIRST_PAST, DCP16):
        for pattern in PATTERNS:
            if not _pattern_applies(pattern, 9):
                continue
            case = build_case(pattern, cp, 9, k)
            want = reference_merge(case["values"], case["ids"])
            for rank in range(cp):
                oracle = merge.dsa_dcp_merge_select_torch_oracle(
                    case["values"], case["ids"], cp_rank=rank, pools_per_block=PPB)
                assert torch.equal(oracle, want[rank]), (cp, k, pattern, rank)


def test_the_cases_hold_what_they_claim() -> None:
    """Ties at the k-th value across ranks, duplicates, pads, empty rows: present."""
    cp, k = DCP16
    tied = build_case("boundary_tie", cp, 4, k)["values"].permute(1, 0, 2).reshape(4, cp * k)
    kth = tied.sort(dim=1, descending=True).values[:, k - 1]
    assert bool((kth == 0.5).all())
    assert bool(((tied == 0.5).sum(dim=1) == 6).all())
    dup = build_case("duplicates", DCP16[0], 4, DCP16[1])
    assert len(torch.unique(dup["values"])) <= 6
    pads = build_case("pads", DCP16[0], 4, DCP16[1])
    assert bool((pads["ids"] < 0).any()) and bool((pads["values"] <= BOUND_FILL_MARK).any())
    short = build_case("short", DCP16[0], 4, DCP16[1])
    assert int((short["ids"][:, -1] >= 0).sum()) == 0
    assert int((short["ids"][:, 0] >= 0).sum()) == DCP16[1] // 2
    top = build_case("max_ids", DCP16[0], 2, DCP16[1])
    assert int(top["ids"].max()) == LOCAL_POOLS - 1
    assert int(global_table(DCP16[0])[:, LOCAL_POOLS - 1].max()) == DCP16[0] * LOCAL_POOLS - 1
    assert DCP16[0] * LOCAL_POOLS - 1 < merge.GLOBAL_ID_LIMIT


def test_the_tile_plan_is_unchanged_up_to_an_unfolded_tile() -> None:
    """Up to MAX_CANDIDATES a row, tiles carry 128 rows, the plan of the unfolded kernel."""
    for cp, k in ((1, 1), (2, 512), (4, 512), UNFOLDED, (8, 531), (2, 2124), (3, 1416)):
        assert cp * k <= merge.MAX_CANDIDATES
        assert merge.tile_rows(cp, k) == merge.PARTITIONS
        for rows in (1, 8, 64, 65, 128, 129, 300, CHUNK_ROWS, DCP16_CHUNK_ROWS):
            for programs in (1, 2):
                assert merge.program_groups_at(rows, programs, merge.tile_rows(cp, k)) == \
                    merge.program_groups(rows, programs), (cp, k, rows, programs)


@pytest.mark.parametrize("cp,k,want_rows", [(*FIRST_PAST, 64), (*DCP16, 64), (32, 512, 32),
                                             (16, 1024, 32), (*FOLDED_EDGE, 64),
                                             (FOLDED_EDGE[0], FOLDED_EDGE[1] + 1, 32)])
def test_a_wide_row_is_served_by_tiles_whose_partitions_fit(cp, k, want_rows) -> None:
    """Past MAX_CANDIDATES every tile of every row count folds each row over enough
    partitions that a partition's columns, with the fold's own operands, fit SBUF."""
    assert cp * k > merge.MAX_CANDIDATES
    assert merge.tile_rows(cp, k) == want_rows
    for rows in (1, 8, 63, 64, 65, 128, 129, CHUNK_ROWS, DCP16_CHUNK_ROWS):
        for programs in (1, 2):
            for _stores, tiles in merge.program_groups_at(rows, programs, want_rows)[0]:
                for _start, n in tiles:
                    f = merge.tile_fold(n, cp, k)
                    assert n * f <= merge.PARTITIONS
                    assert (cp * k) % f == 0
                    assert merge.tile_fits((cp * k) // f, f > 1), (cp, k, rows, n, f)


def test_the_host_serves_the_wide_widths_and_refuses_a_width_no_fold_serves() -> None:
    for cp, k in (FIRST_PAST, DCP16):
        values = torch.zeros(cp, 1, k)
        ids = torch.zeros(cp, 1, k, dtype=torch.int32)
        assert merge._validate(values, ids, 0, PPB) == (cp, 1, k)
    # 128 ranks x 8192: folded over all 128 partitions a row, 8192 columns a partition.
    assert merge.tile_rows(128, 8192) is None
    with pytest.raises(merge.DcpCandidateMergeError, match="fit no tile"):
        merge._validate(torch.zeros(128, 1, 8192), torch.zeros(128, 1, 8192, dtype=torch.int32),
                        0, PPB)


# ---------------------------------------------------------------------------------------------
# CPU: the kernel on the NKI simulator
# ---------------------------------------------------------------------------------------------


def _simulate(monkeypatch) -> None:
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert merge.can_run_dsa_dcp_merge_select(torch.zeros(1), torch.zeros(1))
    assert merge.merge_programs() == 2


def _check(case: dict, ranks) -> list:
    """Every rank's kernel output against the reference; the failures."""
    want = reference_merge(case["values"], case["ids"])
    failures = []
    for rank in ranks:
        merge.reset_dcp_merge_select_dispatch_counters()
        got = merge.dsa_dcp_merge_select(case["values"], case["ids"], cp_rank=rank,
                                         pools_per_block=PPB)
        counters = merge.dcp_merge_select_dispatch_counters()
        if counters != (1, 0) or got.dtype != torch.int32 or not torch.equal(got, want[rank]):
            failures.append((case["pattern"], case["cp"], case["rows"], case["k"], rank,
                             counters, int((got != want[rank]).sum())))
    return failures


@pytest.mark.parametrize("cp,k", [UNFOLDED, FIRST_PAST, DCP16],
                         ids=["4096-unfolded", "4250-first-past", "8192-dcp16"])
def test_the_simulator_kernel_matches_the_reference(cp, k, monkeypatch) -> None:
    _simulate(monkeypatch)
    failures = []
    for rows in ROW_COUNTS:
        for pattern in PATTERNS:
            if not _pattern_applies(pattern, rows):
                continue
            case = build_case(pattern, cp, rows, k)
            ranks = range(cp) if rows <= 8 else (0, cp // 2, cp - 1)
            failures += _check(case, ranks)
    assert not failures, failures[:5]


@pytest.mark.parametrize("cp,k,rows", [(*UNFOLDED, CHUNK_ROWS), (*DCP16, CHUNK_ROWS),
                                       (*DCP16, DCP16_CHUNK_ROWS)],
                         ids=["4096-unfolded", "8192-dcp16", "8192-dcp16-chunk"])
def test_the_simulator_kernel_matches_the_reference_on_a_prefill_chunk(cp, k, rows,
                                                                       monkeypatch) -> None:
    _simulate(monkeypatch)
    failures = []
    for pattern in ("random", "boundary_tie", "max_ids", "pads", "second_core_ties"):
        failures += _check(build_case(pattern, cp, rows, k), (0, cp - 1))
    assert not failures, failures[:5]


# ---------------------------------------------------------------------------------------------
# CPU: neuronx-cc builds, no device
# ---------------------------------------------------------------------------------------------

_BUILD_ROW = "dcp_merge_wide_build"
#: The simulator and CPU-mode variables are dropped and a trn2 CPU compile is pinned, so the
#: child builds a NEFF and opens no device.
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "NEURON_LOGICAL_NC_CONFIG": "2", "PYTHONDONTWRITEBYTECODE": "1"}
#: ``(CP, rows, k, unfolded)`` builds: DCP16 at a decode row and at both prefill chunks, the
#: first width past an unfolded tile and the folded edge at a chunk; and the control, 32 ranks at
#: a chunk in 128-row tiles, which must take a second SBUF allocation pass and spill, so the
#: probe tells a fitting build from a spilling one (at DCP16 that build takes the second pass
#: but logs no spill traffic).
BUILDS = ((16, 1, SELECT_K, False), (16, CHUNK_ROWS, SELECT_K, False),
          (16, DCP16_CHUNK_ROWS, SELECT_K, False), (2, CHUNK_ROWS, 2125, False),
          (FOLDED_EDGE[0], CHUNK_ROWS, FOLDED_EDGE[1], False))
CONTROL_BUILD = (32, CHUNK_ROWS, SELECT_K, True)


def unfolded_tile_rows(cp: int, k: int) -> int:
    """The control's tile height: 128 rows whatever the width. A module-level function, so the
    NKI front end resolves it by name."""
    return merge.PARTITIONS


def build_one(cp: int, rows: int, k: int, work: str, unfolded: bool) -> None:
    """Trace and build the kernel on the LNC2 grid to ``work/kernel.neff``; print one row.

    ``unfolded`` builds 128-row tiles whatever the width, the plan before rows folded past
    ``MAX_CANDIDATES``.
    """
    from dataclasses import replace

    from nki.compiler.driver import _compile_bir_to_neff
    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

    inputs = {"values_hbm": np.zeros((cp, rows, k), dtype=np.float32),
              "ids_hbm": np.zeros((cp, rows, k), dtype=np.int32), "cp_rank": cp - 1,
              "pools_per_block": PPB, "source_digest": merge.SOURCE_DIGEST}
    print(f"{_BUILD_ROW}|tree|module={merge.__file__}", flush=True)
    if unfolded:
        merge.tile_rows = unfolded_tile_rows
    frontend = resolve_frontend_cls()
    compiling = merge.dsa_dcp_merge_select_kernel[2]._to_subclass(CompileKernel,
                                                                  _frontend_cls=frontend)
    neff = os.path.join(work, "kernel.neff")
    options = replace(compiling._compile_opts(), artifacts_dir=work, output_path=neff)
    nir = compile_kernel_to_nir(compiling, inputs=inputs, compile_opts=options,
                                frontend=frontend(), enable_cache=False)
    _compile_bir_to_neff(nir, options, {n: v for n, v in inputs.items()
                                        if isinstance(v, np.ndarray)})
    print(f"{_BUILD_ROW}|built|lnc={options.lnc}|neff_bytes={os.path.getsize(neff)}", flush=True)


def build_facts(work: str) -> dict:
    """A build's matmuls (count, perf modes, operand dtypes) from ``module.mlir``, and whether
    the SBUF allocator took a second pass or spilled, from ``log-neuron-cc.txt``."""
    mlir = pathlib.Path(work, "module.mlir").read_text()
    matmuls = "\n".join(line for line in mlir.splitlines() if "nisa.matmul(" in line)
    log = pathlib.Path(work, "log-neuron-cc.txt").read_text()
    spill = [int(n) for n in re.findall(r"\[Spill Optimization\] reduced DMA traffic (\d+)", log)]
    return {
        "matmuls": matmuls.count("nisa.matmul("),
        "perf_modes": sorted(set(re.findall(r"perf_opt=(\w+)", matmuls))),
        "operand_dtypes": sorted(set(re.findall(r"memref<[0-9x]+x(\w+), #nisa\.mem<", matmuls))),
        "allocator_retry": "FORWARD_ALLOCATE : 0" in log,
        "spill_traffic_reduced": spill,
    }


def run_builds(builds) -> dict:
    """Every build in its own child process, all at once; ``{(cp, rows, k, unfolded): facts}``."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]))
    environment["PATH"] = os.pathsep.join(
        [str(pathlib.Path(sys.executable).parent), environment.get("PATH", "")])
    facts = {}
    with tempfile.TemporaryDirectory(prefix="dcp_merge_wide_build_") as scratch:
        children = []
        for cp, rows, k, unfolded in builds:
            work = os.path.join(scratch, f"cp{cp}_r{rows}_k{k}_u{int(unfolded)}")
            os.makedirs(work)
            children.append(((cp, rows, k, unfolded), work, subprocess.Popen(
                [sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "build-one",
                 str(cp), str(rows), str(k), work, str(int(unfolded))], cwd=work,
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        for key, work, child in children:
            out, err = child.communicate(timeout=1800)
            assert child.returncode == 0, (key, err[-3000:])
            printed = [line for line in out.splitlines() if line.startswith(_BUILD_ROW + "|")]
            assert any(f"module={REPO_ROOT}/" in line for line in printed), printed
            assert any("|built|" in line for line in printed), printed
            facts[key] = build_facts(work)
    return facts


def test_neuronx_cc_builds_the_wide_merge_in_one_sbuf_pass() -> None:
    facts = run_builds((*BUILDS, CONTROL_BUILD))
    control = facts.pop(CONTROL_BUILD)
    assert control["allocator_retry"] and max(control["spill_traffic_reduced"]) > 0, (
        f"the 128-row-tile build at 16384 candidates shows no second allocation pass, so the "
        f"probe cannot tell a fitting build from a spilling one: {control}")
    for key, fact in facts.items():
        assert not fact["allocator_retry"], (key, fact)
        assert fact["spill_traffic_reduced"] and max(fact["spill_traffic_reduced"]) == 0, (
            key, fact)
        assert fact["matmuls"] > 0, (key, fact)
        assert fact["perf_modes"] == ["none_"], (key, fact)
        assert fact["operand_dtypes"] == ["f32"], (key, fact)


if __name__ == "__main__" and len(sys.argv) == 7 and sys.argv[1] == "build-one":
    build_one(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
              sys.argv[6] == "1")
