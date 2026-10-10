# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc builds the DSA decode chain's wide-axis kernels, and the score kernel's
instruction stream does not grow with the candidate axis.

The simulator runs a kernel body as numpy, and the front-end compile
(``test_decode_batch_frontend_compile.py``) stops before the backend verifier. This test
runs the whole compile (Dynamo -> HLO -> neuronx-cc -> NEFF) on the CPU, no device, one
child process per kernel call (``benchmark_dsa_decode_ctx.py compile-one --full``):

* ``dsa_decode_select_kernel`` at e3f38f8's widest row, compacted whole (16384 at B = 1
  and at B = 64, eight requests per GpSimd call), and past it, where the segments of at
  most 8192 candidates and their merge run: 20000 (three segments, a short last one),
  65536 at B = 4 on two programs (a merge of fewer than eight lists per GpSimd call),
  262144 at B = 1 (two merge levels). The backend refused the segmented ones before the
  compaction stopped starting compute instructions on partition ``16 i`` ("Invalid
  access of 1 partitions starting at partition 16").
* ``dsa_decode_scores_kernel`` at B = 4 on two programs for 32768, 65536 and 262144
  candidates (ctx 128k, 256k, 1M): past ``UNROLL_BLOCKS`` blocks the block walk is a
  device loop, so the NEFF is the same size at all three; and at B = 1 on two programs
  for 262144 and for two axes where each core's run of blocks takes the loop and one core
  also takes a block past it (45156 candidates; 16385 with ``unroll_blocks`` 1, where
  an earlier split of the blocks put the loop on one core only and the backend refused
  it: ``NCC_IXGM002 Expected function sg0000 in subgraph 0 to have 5 basic blocks, but
  on core 1 it has 1``).

The compile needs no device. A runtime opened without ``NEURON_RT_VISIBLE_CORES`` resets
every chip of the host, so where ``/dev/neuron*`` exists each child runs under
``bwrap`` with a fresh minimal ``/dev`` (no Neuron device node inside; nothing the child
runs can open one). These tests therefore run on a device host as well as on a CPU host,
and skip only where neither ``neuronx-cc`` nor, on a device host, ``bwrap`` is installed.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_BENCH = _ROOT / "test/hardware/benchmark_dsa_decode_ctx.py"
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_LIBTORCH_REMOTE_CACHE")

SELECT_CASES = ((1, 16384, 1), (1, 20000, 1), (4, 65536, 2), (64, 16384, 2), (1, 262144, 1))
SCORE_FLAT_CANDIDATES = (32768, 65536, 262144)

_DEVICE_NODES = bool(glob.glob("/dev/neuron*"))
_BWRAP = shutil.which("bwrap")
# A child on a device host sees the host's files and a /dev without device nodes.
_NO_DEVICE = (() if not _DEVICE_NODES else
              (_BWRAP, "--dev-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--"))
_needs_compiler = pytest.mark.skipif(
    (_DEVICE_NODES and _BWRAP is None)
    or (shutil.which("neuronx-cc") is None
        and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists()),
    reason="neuronx-cc is not installed, or this host has /dev/neuron* and no bwrap to hide "
           "them from the compile")


def _compile(kernel: str, batch: int, candidates: int, programs: int,
             unroll_blocks: int | None = None) -> dict:
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="dsa_ctx_cpu_compile_") as scratch:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch, NKI_COMPILE_CACHE_URL=f"{scratch}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, str(_BENCH), "compile-one", "--full",
             "--tree", "after", "--kernel", kernel, "--batch", str(batch),
             "--cands", str(candidates), "--programs", str(programs)]
            + ([] if unroll_blocks is None else ["--unroll-blocks", str(unroll_blocks)]),
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=900,
            check=False)
    rows = [json.loads(line.split("COMPILE_ROW ", 1)[1])
            for line in done.stdout.splitlines() if line.startswith("COMPILE_ROW ")]
    assert rows, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
    return rows[0]


@_needs_compiler
@pytest.mark.parametrize("batch, candidates, programs", SELECT_CASES,
                         ids=[f"B{b}-C{c}-g{g}" for b, c, g in SELECT_CASES])
def test_neuronx_cc_builds_the_selection_past_one_segment(batch, candidates, programs):
    row = _compile("select", batch, candidates, programs)
    assert row.get("neff_bytes"), row


@_needs_compiler
def test_the_score_kernel_neff_does_not_grow_with_the_candidate_axis():
    sizes = {c: _compile("scores", 4, c, 2) for c in SCORE_FLAT_CANDIDATES}
    assert all(row.get("neff_bytes") for row in sizes.values()), sizes
    neff = [sizes[c]["neff_bytes"] for c in SCORE_FLAT_CANDIDATES]
    # Unrolled, the stream grew about 2.3x per 4x of axis (105 KB, 242 KB at 16k, 64k).
    assert max(neff) <= 1.05 * min(neff), sizes


@_needs_compiler
def test_neuronx_cc_builds_one_request_on_both_cores_at_a_million_tokens():
    row = _compile("scores", 1, 262144, 2)
    assert row.get("neff_bytes"), row


@_needs_compiler
@pytest.mark.parametrize("candidates, unroll_blocks", [(45156, None), (16385, 1)],
                         ids=["C45156", "C16385-unroll1"])
def test_neuronx_cc_builds_one_request_whose_cores_both_run_the_loop(candidates,
                                                                      unroll_blocks):
    row = _compile("scores", 1, candidates, 2, unroll_blocks)
    assert row.get("neff_bytes"), row
