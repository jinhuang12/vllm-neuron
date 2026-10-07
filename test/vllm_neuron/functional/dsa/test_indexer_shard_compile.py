# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc builds the per-rank selection kernels at the sharded row counts, on the CPU.

Sharded, a rank of the prefill selection runs ``R = ceil(T / d)`` query rows: 16 at
``T = 1024, d = 64`` and 128 at ``T = 1024, d = 8``. The CPU simulator runs a kernel body
as plain Python, so it proves the selection but not that the compiler takes the kernel at
those shapes. This test compiles, to a NEFF, inside one child process per entry that pins
the trn2 target and opens no device node:

* ``_score_gemm_nki`` at ``[R, 32, 128] x [65536, 128]`` -- the candidate width of a
  262,144-token context (``max_model_len // 4``);
* ``rotational_topk`` at ``[R, 65536]``, ``k = 512``, with the config the seam builds for
  that row count. At 16 rows that config differs from every config the replicated path
  builds (tile 8, 16 stages against tile 16, 8 stages at 128 or 1024 rows).

A control entry, a body that reads an undefined name, must be refused, which proves the
children really trace and compile what they are given. Each entry prints its trace and
neuronx-cc seconds; the test records them (``record_property``) for the report. The
pattern is ``test_decode_batch_frontend_compile.py``'s, taken one step further: that test
stops at the NKI front end, this one runs neuronx-cc and checks the NEFF.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROW = "indexer_shard_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

#: The longest context the report prices (``max_model_len``) and the prefill chunk.
CONTEXT = 262144
CHUNK = 1024
#: The two designs the report compares: every TP rank, and groups of 8.
DEGREES = (64, 8)
ROWS = tuple(-(-CHUNK // d) for d in DEGREES)
ENTRIES = [("score", r) for r in ROWS] + [("topk", r) for r in ROWS] + [("control", 1)]


def _dials() -> tuple[int, int, int, int]:
    """``(heads, head_dim, cands, select_k)`` at production dials and :data:`CONTEXT`."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    cfg = Glm5NextTextConfig()
    pool = int(cfg.index_kpool)
    return (int(cfg.index_n_heads), int(cfg.index_head_dim), CONTEXT // pool,
            int(cfg.index_topk) // pool)


def _open_device_nodes() -> int:
    held = [os.path.realpath(f"/proc/self/fd/{h}") for h in os.listdir("/proc/self/fd")]
    return len([one for one in held if "/dev/neuron" in one])


def _compile_one(kind: str, rows: int, work: str) -> None:
    """Trace and compile one entry to ``work/kernel.neff``; print one ``ROW|`` line."""
    from dataclasses import replace

    import ml_dtypes
    import nki
    import nki.language as nl
    import numpy as np
    from nki.compiler.driver import _compile_bir_to_neff
    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

    heads, dim, cands, select_k = _dials()
    if kind == "score":
        from vllm_neuron.functional.dsa import score_gemm as module

        kernel = module._score_gemm_nki
        inputs = {"q_hbm": np.zeros((rows, heads, dim), dtype=ml_dtypes.bfloat16),
                  "k_hbm": np.zeros((cands, dim), dtype=ml_dtypes.bfloat16),
                  "w_hbm": np.zeros((rows, heads), dtype=np.float32)}
    elif kind == "topk":
        from vllm_neuron.functional.dsa import topk_select as module

        config = module._nki_config(rows, cands, select_k, nl.float32)
        kernel = module.rotational_topk[config.n_prgs]
        inputs = {"inp": np.zeros((rows, cands), dtype=np.float32), "config": config}
    else:
        @nki.jit
        def body_that_reads_an_undefined_name(x_hbm):
            out = nl.ndarray(x_hbm.shape, dtype=nl.float32, buffer=nl.shared_hbm)
            held = nl.ndarray((1, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
            nl.store(out[0:1, :], value=held * no_such_name_anywhere)  # noqa: F821
            return out

        module = sys.modules[__name__]
        kernel = body_that_reads_an_undefined_name
        inputs = {"x_hbm": np.zeros((1, 128), dtype=np.float32)}

    print(ROW + "|tree|module=" + str(module.__file__), flush=True)
    frontend = resolve_frontend_cls()
    compiling = kernel._to_subclass(CompileKernel, _frontend_cls=frontend)
    neff = os.path.join(work, "kernel.neff")
    options = replace(compiling._compile_opts(), artifacts_dir=work, output_path=neff)
    message, trace_s, ncc_s = "", 0.0, 0.0
    try:
        start = time.perf_counter()
        nir = compile_kernel_to_nir(compiling, inputs=inputs, compile_opts=options,
                                    frontend=frontend(), enable_cache=False)
        traced = time.perf_counter()
        tensors = {k: v for k, v in inputs.items() if isinstance(v, np.ndarray)}
        _compile_bir_to_neff(nir, options, tensors)
        trace_s, ncc_s = traced - start, time.perf_counter() - traced
    except BaseException as refusal:  # a refusal is a result here, not a test error
        message = " ".join(str(refusal).split())[:2000] or type(refusal).__name__
    size = os.path.getsize(neff) if os.path.exists(neff) else 0
    print(f"{ROW}|entry={kind}_r{rows}|refused={bool(message)}|lnc={options.lnc}"
          f"|trace_s={trace_s:.2f}|ncc_s={ncc_s:.2f}|neff_bytes={size}"
          f"|neuron_fds={_open_device_nodes()}|diagnostic={message or 'none'}", flush=True)


def _rows() -> list[dict[str, str]]:
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=os.pathsep.join(
        [str(_ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]))
    # The NKI driver runs ``neuronx-cc`` from PATH; it sits beside this interpreter.
    environment["PATH"] = os.pathsep.join(
        [str(pathlib.Path(sys.executable).parent), environment.get("PATH", "")])
    rows = []
    # One child per entry, all at once: the two score builds dominate (about a minute
    # each on a trn2 host), and the compiler leaves its files in the scratch directory.
    with tempfile.TemporaryDirectory(prefix="indexer_shard_compile_") as scratch:
        children = []
        for kind, count in ENTRIES:
            work = os.path.join(scratch, f"{kind}_r{count}")
            os.makedirs(work)
            children.append(subprocess.Popen(
                [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-one", kind,
                 str(count), work],
                cwd=work, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True))
        for child in children:
            out, err = child.communicate(timeout=1800)
            assert child.returncode == 0, err[-3000:]
            printed = [line for line in out.splitlines() if line.startswith(ROW + "|")]
            tree = [line for line in printed if line.startswith(ROW + "|tree|")]
            assert tree and f"module={_ROOT}/" in tree[0], tree
            for line in printed:
                if line.startswith(ROW + "|entry="):
                    head, _, diagnostic = line.partition("|diagnostic=")
                    fields = dict(one.split("=", 1) for one in head.split("|")[1:])
                    rows.append({**fields, "diagnostic": diagnostic})
    return rows


def test_neuronx_cc_builds_the_per_rank_kernels_at_the_longest_context(record_property) -> None:
    rows = {row["entry"]: row for row in _rows()}
    assert set(rows) == {f"{kind}_r{count}" for kind, count in ENTRIES}, sorted(rows)
    assert {row["neuron_fds"] for row in rows.values()} == {"0"}, rows
    control = rows.pop("control_r1")
    assert control["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it compiled nothing: "
        f"{control}")
    refused = [f"{name}: {row['diagnostic']}" for name, row in rows.items()
               if row["refused"] != "False"]
    assert not refused, refused
    for name, row in rows.items():
        assert int(row["neff_bytes"]) > 0, (name, row)
    from vllm_neuron.functional.dsa.topk_select import _NUM_PROGRAMS

    # The top-k builds at the seam's own program count.
    assert {rows[f"topk_r{r}"]["lnc"] for r in ROWS} == {str(_NUM_PROGRAMS)}
    seconds = {name: {"trace_s": float(row["trace_s"]), "neuronx_cc_s": float(row["ncc_s"]),
                      "neff_bytes": int(row["neff_bytes"])} for name, row in rows.items()}
    record_property("compile_seconds", seconds)


if __name__ == "__main__" and sys.argv[1:2] == ["compile-one"]:
    _compile_one(sys.argv[2], int(sys.argv[3]), sys.argv[4])
