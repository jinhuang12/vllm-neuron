# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts the decode attention and low-precision projection entries.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
compiler's front end refuses. This file compiles every new entry at the served shapes --
decode attention dense and selected at B in {1, 4} x T in {1, 2, 4, 6} query rows per
request on one and two programs, and the
low-precision projection at each DSA site's TP=64 width at M in {1, 4} on one and two
programs -- inside a child process that pins the platform target, opens no device node,
and proves it parsed bodies by refusing one that reads an undefined name. The pattern is
``test_mla_sparse_frontend_compile.py``'s.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import nki
import nki.language as nl

ROW = "decode_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

LATENT, PAGE, BANK_PAGES = 512, 128, 64
DENSE_PAGES, SELECTED_PAGES, TOPK_ROWS = 16, 32, 2048
# Query rows per request: the one-token step and the verify step's 1 + k at k in {1, 3, 5}.
QUERY_ROWS = (1, 2, 4, 6)
SITES = ((4096, 1536, True), (1536, 256, True), (4096, 512, True), (256, 4096, True),
         (1536, 4096, False), (4096, 128, False), (4096, 32, False))


@nki.jit
def body_that_reads_an_undefined_name(x_hbm):
    """Refused by a front end that parses bodies; accepted only by one that does not."""
    out = nl.ndarray(x_hbm.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    held = nl.ndarray((1, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
    nl.store(out[0:1, :], value=held * no_such_name_anywhere)  # noqa: F821
    return out


def _open_device_nodes() -> int:
    held = [os.path.realpath(f"/proc/self/fd/{h}") for h in os.listdir("/proc/self/fd")]
    return len([one for one in held if "/dev/neuron" in one])


def _compile_each_entry() -> None:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.functional.attention import mla_decode as MD
    from vllm_neuron.functional.attention import mla_projections as MP

    bf, i32 = torch.bfloat16, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def grid(entry, programs):
        call = wrap_nki(entry)
        return call[2] if programs == 2 else call

    built = []
    for batch in (1, 4):
        for rows in QUERY_ROWS:
            for programs in (1, 2):
                name = f"b{batch}_t{rows}_g{programs}"
                built.append((f"dense_{name}", lambda b=batch, t=rows, g=programs: grid(
                    MD.mla_decode_dense_kernel, g)(
                    fake((b * t, 1, LATENT), bf), fake((BANK_PAGES * PAGE, LATENT), bf),
                    fake((b, DENSE_PAGES), i32), fake((b,), i32), fake((b * t, LATENT), bf),
                    0.0625, PAGE, MD.SOURCE_DIGEST)))
                built.append((f"selected_{name}", lambda b=batch, t=rows, g=programs: grid(
                    MD.mla_decode_selected_kernel, g)(
                    fake((b * t, 1, LATENT), bf), fake((BANK_PAGES * PAGE, LATENT), bf),
                    fake((b, SELECTED_PAGES), i32), fake((b,), i32), fake((b * t, LATENT), bf),
                    fake((b * t, TOPK_ROWS), i32), 0.0625, PAGE, MD.SOURCE_DIGEST)))
    for rows in (1, 4):
        for idim, odim, fp8 in SITES:
            for programs in (1, 2):
                name = f"proj_m{rows}_{idim}x{odim}_{'fp8' if fp8 else 'bf16'}_g{programs}"
                if fp8:
                    call = lambda r=rows, k=idim, n=odim, g=programs: grid(
                        MP.mla_projection_lowp_kernel, g)(
                        fake((r, k), bf), fake((k, n), torch.float8_e4m3fn),
                        fake((g, k // 128, n // 128 // g), torch.float32), 0)
                else:
                    call = lambda r=rows, k=idim, n=odim, g=programs: grid(
                        MP.mla_projection_lowp_kernel, g)(
                        fake((r, k), bf), fake((k, n), bf), None, 0)
                built.append((name, call))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + MD.__file__, flush=True)
    for name, call in built:
        message = ""
        try:
            with FakeTensorMode():
                call()
        except BaseException as refusal:  # a refusal is a result here, not a test error
            message = " ".join(str(refusal).split())[:2000]
        print(f"{ROW}|entry={name}|refused={bool(message)}|neuron_fds={_open_device_nodes()}"
              f"|diagnostic={message or 'none'}", flush=True)


def _rows() -> list[dict[str, str]]:
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    # The compiler leaves per-kernel debug files in its working directory, so the child
    # runs in a scratch directory and the worktree stays clean.
    with tempfile.TemporaryDirectory(prefix="decode_frontend_") as scratch:
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-entry"],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=1800,
            check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    tree = [line for line in printed if line.startswith(ROW + "|tree|")]
    assert tree and f"module={_ROOT}/" in tree[0], tree
    rows = []
    for line in printed:
        if not line.startswith(ROW + "|entry="):
            continue
        head, _, diagnostic = line.partition("|diagnostic=")
        fields = dict(one.split("=", 1) for one in head.split("|")[1:])
        rows.append({**fields, "diagnostic": diagnostic})
    return rows


def test_the_front_end_accepts_every_new_entry_at_the_served_shapes() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    assert len(entries) == 2 * 2 * len(QUERY_ROWS) * 2 + 2 * len(SITES) * 2, len(entries)
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
