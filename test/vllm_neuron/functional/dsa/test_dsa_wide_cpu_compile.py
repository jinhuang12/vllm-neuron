# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc builds the DSA indexer kernels whose width grows with the context.

The simulator runs a kernel body as numpy and never checks SBUF capacity or the
backend verifier, and the front-end compile stops before the allocator. This test
runs the whole compile (Dynamo -> HLO -> neuronx-cc -> NEFF) on the CPU
(``NEURON_LIBTORCH_CPU_COMPILE=1``) for the two kernels whose SBUF tiles scale with
the candidate count C = max_model_len / 4, at the widths a 64k, 256k and 1M context
gives them:

* ``_causal_bound_nki`` at ``[8192, C]`` (one 8192-token prefill chunk), C in
  {16384, 32768, 65536};
* ``dsa_decode_scores_kernel`` at C in {65536 (B=4, two programs), 262144 (B=1)}.

The child stops before the executor is built (``build_executable`` is replaced), so it
never opens the Neuron runtime or a device node. A NEFF on disk is the pass.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

ROW = "dsa_wide_cpu_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_RT_VISIBLE_CORES")

CASES = (
    ("causal_bound", 8192, 16384, 1),
    ("causal_bound", 8192, 32768, 1),
    ("causal_bound", 8192, 65536, 1),
    ("decode_scores", 4, 65536, 2),
    ("decode_scores", 1, 262144, 1),
)


def _child(kind: str, rows: int, width: int, grid: int) -> None:
    import torch

    import libtorch_neuronx_lite  # noqa: F401  (registers the backends)
    import libtorch_neuronx_lite.compile.backend as backend
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    meta = torch.device("meta")
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    if kind == "causal_bound":
        from vllm_neuron.functional.dsa import causal_bound as CB

        def fn(scores, lens):
            return wrap_nki(CB._causal_bound_nki)(scores, lens, 4)

        args = (torch.empty((rows, width), dtype=f32, device=meta),
                torch.empty((rows, 1), dtype=i32, device=meta))
        module = CB.__file__
    else:
        from vllm_neuron.functional.dsa import decode_batch as DB

        def fn(q, w, bank, slots, lens, pos, pooled):
            call = wrap_nki(DB.dsa_decode_scores_kernel)
            if grid == 2:
                call = call[2]
            return call(q, w, bank, slots, lens, pos, pooled, width, 4, DB.SOURCE_DIGEST)

        batch = rows
        args = (torch.empty((batch, 32, 128), dtype=bf, device=meta),
                torch.empty((batch, 32), dtype=f32, device=meta),
                torch.empty((2, width + 1, 128), dtype=bf, device=meta),
                torch.empty((batch, 1), dtype=i32, device=meta),
                torch.empty((batch, 1), dtype=i32, device=meta),
                torch.empty((batch, 1), dtype=i32, device=meta),
                torch.empty((batch, 128), dtype=bf, device=meta))
        module = DB.__file__
    neff, message = "", ""
    try:
        torch.compile(fn, backend="neuron_libtorch", fullgraph=True)(*args)
        message = "the compiled graph ran; build_executable was not replaced"
    except BaseException as caught:  # the compiler's failure is a result here
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:1500] or type(caught).__name__
    ok = bool(neff) and os.path.isfile(neff)
    print(f"{ROW}|module={module}|ok={ok}|neff={neff or 'none'}|diagnostic={message or 'none'}",
          flush=True)


@pytest.mark.skipif(shutil.which("neuronx-cc") is None
                    and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists(),
                    reason="neuronx-cc is not installed")
@pytest.mark.parametrize("kind,rows,width,grid", CASES,
                         ids=[f"{k}-{r}x{w}-g{g}" for k, r, w, g in CASES])
def test_neuronx_cc_builds_the_wide_kernel(kind, rows, width, grid):
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="dsa_wide_compile_") as scratch:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch, PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "child", kind, str(rows),
             str(width), str(grid)],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
    fields = dict(part.split("=", 1) for part in printed[-1].split("|")[1:4])
    assert fields["module"].startswith(f"{_ROOT}/"), fields
    assert fields["ok"] == "True", printed[-1]


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
