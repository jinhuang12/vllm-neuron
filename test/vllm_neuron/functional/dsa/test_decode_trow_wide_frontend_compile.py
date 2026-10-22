# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts the wide ``T``-row scores kernel at the 1M / DCP 8 shapes.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
compiler's front end refuses; the old kernel's output transpose at 32768 candidates
(``[128, 256]``) is one. This compiles ``dsa_decode_scores_rows_wide_kernel`` at the
served calls of the 1M / DCP 8 decode graph (32768 candidates on a one-slot bank, the
query rotated on chip, T = 1 with row 0's position and T = 4 with each row's own, one
program), at a ragged last tile in the second output group (16400 and 32740) and a
whole one (16512) at T in {1, 4} x B in {1, 4} on one and two programs, and at 16384
(one group). It also compiles the old kernel at the served T = 4 call, which the front
end must refuse at its transpose, as the device host did. All inside a child process
that pins the platform target, opens no device node, and proves it parsed bodies by
refusing one that reads an undefined name. The pattern is
``test_decode_trow_frontend_compile.py``'s.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import nki
import nki.language as nl

ROW = "dsa_trow_wide_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

HEADS, DIM, POOL, SLOTS = 32, 128, 4, 3
#: The served axis of the 1M / DCP 8 form (131072 tokens per rank over pools of 4).
SERVED_CANDIDATES = 32768
SERVED_ROWS = (1, 4)
GROUP_CANDIDATES = (16384, 16400, 16512, 32740)
GROUP_ROWS = (1, 4)
GROUP_BATCHES = (1, 4)


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

    from vllm_neuron.functional.dsa import decode_trow as TR
    from vllm_neuron.functional.dsa import decode_trow_wide as TW

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def scores(kernel, digest, slots, cands, rows, batch, programs, rotate, per_row):
        call = wrap_nki(kernel)
        if programs == 2:
            call = call[2]
        positions = batch * rows if per_row else batch
        return call(
            fake((batch * rows, HEADS, DIM), bf), fake((batch * rows, HEADS), f32),
            fake((slots, cands + 1, DIM), bf), fake((batch, 1), i32),
            fake((positions, 1), i32), fake((batch * rows, DIM), bf), cands, POOL, rows,
            rotate, digest)

    def wide(*args):
        return scores(TW.dsa_decode_scores_rows_wide_kernel, TW.SOURCE_DIGEST, *args)

    built = []
    for rows in SERVED_ROWS:
        built.append((f"served_c{SERVED_CANDIDATES}_t{rows}",
                      lambda t=rows: wide(1, SERVED_CANDIDATES, t, 1, 1, True, t > 1)))
    for cands in GROUP_CANDIDATES:
        for rows in GROUP_ROWS:
            for batch in GROUP_BATCHES:
                for programs in (1, 2):
                    built.append((f"wide_c{cands}_t{rows}_b{batch}_g{programs}",
                                  lambda c=cands, t=rows, b=batch, g=programs: wide(
                                      SLOTS, c, t, b, g, False, False)))
    built.append(("old_served_c32768_t4", lambda: scores(
        TR.dsa_decode_scores_rows_kernel, TR.SOURCE_DIGEST, 1, SERVED_CANDIDATES, 4, 1, 1,
        True, True)))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + TW.__file__, flush=True)
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
    with tempfile.TemporaryDirectory(prefix="dsa_trow_wide_frontend_") as scratch:
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-entry"],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=3600,
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


def test_the_front_end_accepts_the_wide_kernel_and_refuses_the_old_one_at_32768() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    by_name = {row["entry"]: row for row in rows}
    assert by_name["undefined_name_body"]["refused"] == "True", (
        "the child accepted a body that reads an undefined name, so it parsed none")
    old = by_name["old_served_c32768_t4"]
    assert old["refused"] == "True" and "transpose" in old["diagnostic"].lower(), old
    entries = [row for name, row in by_name.items()
               if name not in ("undefined_name_body", "old_served_c32768_t4")]
    want = len(SERVED_ROWS) + len(GROUP_CANDIDATES) * len(GROUP_ROWS) * len(GROUP_BATCHES) * 2
    assert len(entries) == want, (len(entries), want)
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
