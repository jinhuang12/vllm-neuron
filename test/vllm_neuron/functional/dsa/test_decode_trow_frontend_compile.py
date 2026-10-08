# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts the ``T``-row DSA decode kernels at the served shapes.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
compiler's front end refuses. This compiles ``dsa_decode_ring_rows_kernel`` at ring depth
4 (T in {1, 2}) and 8 (T in {1, 2, 4, 6}) at B in {1, 4, 130} (130 spans two partition
tiles), and ``dsa_decode_scores_rows_kernel`` at ctx 4096 and 8192 (1024 and 2048
candidates) and a ragged 300, at T in {1, 2, 4, 6} x B in {1, 4} on one and two
programs, and both at B=1 on a one-slot bank (the one-request carrier's view, read
statically), inside a child process that pins the platform target, opens no device node,
and proves it parsed bodies by refusing one that reads an undefined name. The pattern is
``test_decode_batch_frontend_compile.py``'s.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import nki
import nki.language as nl

ROW = "dsa_trow_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

HEADS, DIM, POOL, SLOTS = 32, 128, 4, 140
#: (ring depth, rows per request): the served ring and the deeper ring a speculative
#: config allocates, each at every row count it takes.
RING_FORMS = ((4, 1), (4, 2), (8, 1), (8, 2), (8, 4), (8, 6))
RING_BATCHES = (1, 4, 130)
SCORE_CANDIDATES = (1024, 2048, 300)
SCORE_ROWS = (1, 2, 4, 6)
SCORE_BATCHES = (1, 4)
ONE_SLOT_CANDIDATES = (1024, 300)


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

    from vllm_neuron.functional.dsa import decode_tail_update as TU
    from vllm_neuron.functional.dsa import decode_trow as TR

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def grid(entry, programs):
        call = wrap_nki(entry)
        return call[2] if programs == 2 else call

    def ring(slots, depth, rows, batch):
        return wrap_nki(TU.dsa_decode_ring_rows_kernel)(
            fake((slots, 2 * depth * DIM), bf), fake((batch, 1), i32),
            fake((batch * rows, DIM), bf), fake((batch * rows, DIM), bf),
            fake((POOL, DIM), f32), fake((batch, 1), i32), POOL, rows, TU.ROWS_SOURCE_DIGEST)

    def scores(slots, cands, rows, batch, programs):
        return grid(TR.dsa_decode_scores_rows_kernel, programs)(
            fake((batch * rows, HEADS, DIM), bf), fake((batch * rows, HEADS), f32),
            fake((slots, cands + 1, DIM), bf), fake((batch, 1), i32), fake((batch, 1), i32),
            fake((batch * rows, DIM), bf), cands, POOL, rows, TR.SOURCE_DIGEST)

    built = []
    for depth, rows in RING_FORMS:
        for batch in RING_BATCHES:
            built.append((f"ring_d{depth}_t{rows}_b{batch}",
                          lambda d=depth, t=rows, b=batch: ring(SLOTS, d, t, b)))
    for cands in SCORE_CANDIDATES:
        for rows in SCORE_ROWS:
            for batch in SCORE_BATCHES:
                for programs in (1, 2):
                    built.append((f"scores_c{cands}_t{rows}_b{batch}_g{programs}",
                                  lambda c=cands, t=rows, b=batch, g=programs: scores(
                                      SLOTS, c, t, b, g)))
    built.append(("ring_d8_t4_b1_one_slot", lambda: ring(1, 8, 4, 1)))
    for cands in ONE_SLOT_CANDIDATES:
        built.append((f"scores_c{cands}_t4_b1_one_slot", lambda c=cands: scores(1, c, 4, 1, 1)))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + TR.__file__, flush=True)
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
    with tempfile.TemporaryDirectory(prefix="dsa_trow_frontend_") as scratch:
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


def test_the_front_end_accepts_both_row_kernels_at_the_served_shapes() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    want = (len(RING_FORMS) * len(RING_BATCHES)
            + len(SCORE_CANDIDATES) * len(SCORE_ROWS) * len(SCORE_BATCHES) * 2
            + 1 + len(ONE_SLOT_CANDIDATES))
    assert len(entries) == want, (len(entries), want)
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
