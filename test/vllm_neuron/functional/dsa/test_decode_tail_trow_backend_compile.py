# SPDX-License-Identifier: Apache-2.0
"""The ``T``-row ring kernel passes neuronx-cc's backend verifier on one and two programs.

The simulator and the front-end compile (``test_decode_trow_frontend_compile.py``) stop
before the backend verifier, so they cannot see a rule it enforces on a DMA access
pattern. The two-program form writes each program's ring rows back as two strided column
runs of the ring tile, a pattern the one-program form does not use. This test compiles
``dsa_decode_ring_rows_kernel`` through neuronx-cc (``nki``'s standalone path, which runs
the same BIR verifier) in a child process that opens no device node, at LNC1 and LNC2:

* the served verify and one-token steps (one request on a one-slot bank, ring depth 8,
  rows 4 and 1), and four requests on an 11-slot bank (the indirect bank read) at rows 4;
* every compile succeeds, except two programs at one row, which the kernel refuses by
  name (the seam never launches it: one row takes one program).

BIRSim is not run. It refuses the one-program kernel as well: ``_ring_member``'s first
predicated copy reads its destination tile before any instruction has written it
(``NCC_ISIM019`` uninitialized read), which the device does not check; each partition
takes exactly one group's copy, so no unwritten value reaches a result. The simulator
tests check the numbers.

The pattern is ``test_glue_backend_compile.py``'s. Where ``/dev/neuron*`` exists the child
runs under ``bwrap`` with a fresh minimal ``/dev``, as ``test_decode_ctx_cpu_compile.py``
does, so nothing it runs can open a device.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROW = "dsa_ring_backend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG")
_PIN = {"NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "PYTHONDONTWRITEBYTECODE": "1"}
POOL, HEAD_DIM, DEPTH = 4, 128, 8
#: (requests, bank slots, rows per request).
CASES = ((1, 1, 4), (1, 1, 1), (4, 11, 4))
#: The kernel's refusal of two programs at one row.
ONE_ROW_REFUSAL = "two programs split the rows: at least 2"
#: Row 0's position of request 0 (the compile does not depend on it); request ``b``'s is
#: ``START + 3 b``.
START = 2045

_DEVICE_NODES = bool(glob.glob("/dev/neuron*"))
_BWRAP = shutil.which("bwrap")
_NO_DEVICE = (() if not _DEVICE_NODES else
              (_BWRAP, "--dev-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--"))


def _child() -> None:
    import ml_dtypes
    import numpy as np

    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import StandaloneKernel

    from vllm_neuron.functional.dsa import decode_tail_update as TU

    bf = ml_dtypes.bfloat16

    def noop(compiled, inputs, outputs):
        return None

    def compile_only(lnc, args):
        kernel = TU.dsa_decode_ring_rows_kernel
        launched = kernel[lnc] if lnc > 1 else kernel
        launched._to_subclass(StandaloneKernel, _frontend_cls=resolve_frontend_cls(),
                              _executor=noop)(**args)

    def build(batch, slots_total, rows):
        rng = np.random.default_rng(700 + rows + batch)
        return dict(
            tail_hbm=rng.standard_normal((slots_total, 2 * DEPTH * HEAD_DIM)).astype(bf),
            slots_hbm=rng.permutation(slots_total)[:batch].astype(np.int32).reshape(batch, 1),
            key_hbm=rng.standard_normal((batch * rows, HEAD_DIM)).astype(bf),
            score_hbm=rng.standard_normal((batch * rows, HEAD_DIM)).astype(bf),
            ape_hbm=rng.standard_normal((POOL, HEAD_DIM)).astype(np.float32),
            pos_hbm=(START + 3 * np.arange(batch)).astype(np.int32).reshape(batch, 1),
            pool_size=POOL, rows=rows, source_digest=TU.ROWS_SOURCE_DIGEST)

    for batch, slots_total, rows in CASES:
        for lnc in (1, 2):
            row = dict(B=batch, slots=slots_total, rows=rows, lnc=lnc)
            try:
                compile_only(lnc, build(batch, slots_total, rows))
                row.update(compiled=True, reading="none")
            except Exception as refusal:  # the backend's refusal is the row's reading
                row.update(compiled=False, reading=" ".join(str(refusal).split())[:1500])
            print(ROW + "|" + json.dumps(row), flush=True)
    fds = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            fds += "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}")
        except OSError:
            pass
    print(ROW + "|" + json.dumps(dict(tree=TU.__file__, neuron_fds=fds)), flush=True)


@pytest.mark.skipif(
    (_DEVICE_NODES and _BWRAP is None)
    or (shutil.which("neuronx-cc") is None
        and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists()),
    reason="neuronx-cc is not installed, or this host has /dev/neuron* and no bwrap to hide "
           "them from the compile")
def test_the_ring_kernel_passes_the_backend_on_one_and_two_programs(tmp_path):
    # neuronx-cc writes its intermediates into the working directory, so the child runs
    # in a scratch one, not the tree.
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    venv_bin = str(pathlib.Path(sys.executable).parent)
    environment.update(_PIN, PYTHONPATH=str(_ROOT),
                       PATH=venv_bin + os.pathsep + environment.get("PATH", ""))
    done = subprocess.run([*_NO_DEVICE, sys.executable, str(pathlib.Path(__file__).resolve()),
                           "child"],
                          cwd=tmp_path, env=environment, capture_output=True, text=True,
                          timeout=1500, check=False)
    rows = [json.loads(line.split("|", 1)[1]) for line in done.stdout.splitlines()
            if line.startswith(ROW + "|")]
    for row in rows:
        print(row)
    assert done.returncode == 0, done.stderr[-3000:]
    assert rows and rows[-1]["neuron_fds"] == 0, rows[-1:]
    assert rows[-1]["tree"].startswith(f"{_ROOT}/"), rows[-1]
    cases = rows[:-1]
    assert len(cases) == len(CASES) * 2
    one_row_pair = [row for row in cases if row["lnc"] == 2 and row["rows"] == 1]
    assert one_row_pair and all(not row["compiled"] and ONE_ROW_REFUSAL in row["reading"]
                                for row in one_row_pair), one_row_pair
    refused = [row for row in cases if not row["compiled"] and row not in one_row_pair]
    assert refused == [], refused


if __name__ == "__main__" and sys.argv[1:] == ["child"]:
    _child()
