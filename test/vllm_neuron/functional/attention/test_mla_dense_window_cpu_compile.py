# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc builds the dense-window MLA prefill kernel at the served chunk shapes.

The simulator runs a kernel body as numpy and never checks SBUF capacity or the backend
verifier. This test runs the whole compile (Dynamo -> HLO -> neuronx-cc -> NEFF) on the
CPU (``NEURON_LIBTORCH_CPU_COMPILE=1``, trn2, LNC2) through the seam the call site uses,
at 1 head (TP=64) and latent 512:

* 1024 query rows over a 2048-row window (the p1 chunk: segment 1024 + chunk 1024);
* 2048 query rows over a 2048-row window (one 2048-token chunk);
* 1024 query rows over a 2176-row window (17 pages: the widest window the identity
  bound of 2051 rows needs at page 128).

The child stops before the executor is built (``build_executable`` is replaced), so it
never opens the Neuron runtime, and its platform target is set, so nothing asks the
runtime for one. A NEFF on disk is the pass, with no ``/dev/neuron*`` node open in the
child afterwards. The test prints the compile seconds and the first core's SBUF and PSUM
allocator outcome from the compiler's own log (``-s`` shows them).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

ROW = "mla_dense_window_cpu_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
#: ``(query rows, window pages)`` at page 128, latent 512, one head.
CASES = ((1024, 16), (2048, 16), (1024, 17))
LATENT, PAGE, SCALE = 512, 128, 0.0625


def _child(rows: int, pages: int) -> None:
    import torch

    import libtorch_neuronx_lite  # noqa: F401  (registers the backends)
    import libtorch_neuronx_lite.compile.backend as backend

    from test.vllm_neuron.functional.attention.neuron_device_nodes import (
        open_neuron_device_nodes,
    )
    from vllm_neuron.functional.attention import mla_dense_window as DW

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    meta = torch.device("meta")
    bf, i32 = torch.bfloat16, torch.int32

    def fn(q, bank, table, seq_lens, written, offset):
        return DW.mla_dense_window_attention(q, bank, seq_lens, SCALE, block_table_row=table,
                                             written=written, write_offset=offset,
                                             page_size=PAGE)

    args = (torch.empty((rows, 1, LATENT), dtype=bf, device=meta),
            torch.empty((2 * pages * PAGE, LATENT), dtype=bf, device=meta),
            torch.empty((pages, 1), dtype=i32, device=meta),
            torch.empty((rows,), dtype=i32, device=meta),
            torch.empty((rows, LATENT), dtype=bf, device=meta),
            torch.empty((1, 1), dtype=i32, device=meta))
    neff, message = "", ""
    started = time.monotonic()
    try:
        torch.compile(fn, backend="neuron_libtorch", fullgraph=True)(*args)
        message = "the compiled graph ran; build_executable was not replaced"
    except BaseException as caught:  # the compiler's failure is a result here
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:1500] or type(caught).__name__
    seconds = time.monotonic() - started
    ok = bool(neff) and os.path.isfile(neff)
    print(f"{ROW}|module={DW.__file__}|ok={ok}|seconds={seconds:.1f}|neff={neff or 'none'}"
          f"|programs={DW._programs(rows)}|device_nodes={len(open_neuron_device_nodes())}"
          f"|diagnostic={message or 'none'}", flush=True)


#: The allocator lines of ``log-neuron-cc.txt`` that state the SBUF and PSUM outcome.
_ALLOCATOR = (r"Available free byte per partition:\s*\d+", r"SB spills = \d+ tensors",
              r"Allocated: [0-9.]+ \(\d+\)", r"Spilled: [0-9.]+ \(\d+\)",
              r"\d+% PSUM utilization after allocation",
              r"\d+ pinned tensors will require about \d+ bytes/partition")


def _allocator_lines(scratch: str) -> list[str]:
    """The first core's SBUF/PSUM allocator outcome, from the compiler's own log."""
    found = []
    for log in pathlib.Path(scratch).rglob("log-neuron-cc.txt"):
        text = log.read_text(errors="ignore")
        for pattern in _ALLOCATOR:
            hit = re.search(r"\(nc00/sg00\) \[(?:SB|PSUM)_Allocator\]:\s*(" + pattern + ")",
                            text)
            if hit:
                found.append(hit.group(1))
    return found


@pytest.mark.skipif(shutil.which("neuronx-cc") is None
                    and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists(),
                    reason="neuronx-cc is not installed")
@pytest.mark.parametrize("rows,pages", CASES, ids=[f"q{r}-w{p * PAGE}" for r, p in CASES])
def test_neuronx_cc_builds_the_dense_window_kernel(rows, pages):
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    scratch = tempfile.mkdtemp(prefix="mla_dense_window_compile_")
    try:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch, PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "child", str(rows),
             str(pages)],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
        printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
        assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
        fields = dict(part.split("=", 1) for part in printed[-1].split("|")[1:])
        print(printed[-1])
        for line in _allocator_lines(scratch):
            print(f"{ROW}|allocator|{line}")
        assert fields["module"].startswith(f"{_ROOT}/"), fields
        assert fields["ok"] == "True", printed[-1]
        assert fields["programs"] == "2", fields
        assert fields["device_nodes"] == "0", fields
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(int(sys.argv[2]), int(sys.argv[3]))
