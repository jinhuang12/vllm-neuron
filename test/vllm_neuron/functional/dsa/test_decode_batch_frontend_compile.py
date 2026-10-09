# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts the batched DSA decode kernels at the served shapes.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
compiler's front end refuses. This compiles ``dsa_decode_ring_step_kernel`` at B in
{1, 16, 64, 130} (130 spans two partition tiles) and ``dsa_decode_scores_kernel`` at
ctx 4096 and 8192 (1024 and 2048 candidates), a ragged 300, and the wide 16385 and 65536
(ctx 65540 and 262144: the device loop over score blocks, with and without a tail block),
at B in {1, 16}, on one and two programs, and both at B=1 on a one-slot bank (the
one-request carrier's view, read statically; 16500 takes the loop there), inside a child process that pins the platform target, opens no device
node, and proves it parsed bodies by refusing one that reads an undefined name. The
pattern is ``test_mla_decode_frontend_compile.py``'s.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import nki
import nki.language as nl

ROW = "dsa_batch_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

HEADS, DIM, POOL, SLOTS = 32, 128, 4, 140
RING_BATCHES = (1, 16, 64, 130)
#: Past 4 uniform blocks of 32 tiles (16384 candidates) the score blocks run as a device
#: loop: 16385 (the loop and a one-row tail block) and 65536 (the loop alone).
SCORE_CANDIDATES = (1024, 2048, 300, 16385, 65536)
SCORE_BATCHES = (1, 16)
#: Candidate counts read from a one-slot bank: whole tiles, a ragged last tile, the loop.
ONE_SLOT_CANDIDATES = (1024, 300, 16500)


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

    from vllm_neuron.functional.dsa import decode_batch as DB

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def grid(entry, programs):
        call = wrap_nki(entry)
        return call[2] if programs == 2 else call

    built = []
    for batch in RING_BATCHES:
        built.append((f"ring_b{batch}", lambda b=batch: wrap_nki(
            DB.dsa_decode_ring_step_kernel)(
            fake((SLOTS, 2 * POOL * DIM), bf), fake((b, 1), i32), fake((b, DIM), bf),
            fake((b, DIM), bf), fake((POOL, DIM), f32), fake((b, 1), i32), POOL,
            DB.SOURCE_DIGEST)))
    for cands in SCORE_CANDIDATES:
        for batch in SCORE_BATCHES:
            for programs in (1, 2):
                built.append((f"scores_c{cands}_b{batch}_g{programs}",
                              lambda c=cands, b=batch, g=programs: grid(
                                  DB.dsa_decode_scores_kernel, g)(
                                  fake((b, HEADS, DIM), bf), fake((b, HEADS), f32),
                                  fake((SLOTS, c + 1, DIM), bf), fake((b, 1), i32),
                                  fake((b, 1), i32), fake((b, 1), i32), fake((b, DIM), bf),
                                  c, POOL, DB.score_blocks(b, c, g), DB.UNROLL_BLOCKS,
                                  DB.SOURCE_DIGEST)))
    built.append(("ring_b1_one_slot", lambda: wrap_nki(DB.dsa_decode_ring_step_kernel)(
        fake((1, 2 * POOL * DIM), bf), fake((1, 1), i32), fake((1, DIM), bf),
        fake((1, DIM), bf), fake((POOL, DIM), f32), fake((1, 1), i32), POOL,
        DB.SOURCE_DIGEST)))
    for cands in ONE_SLOT_CANDIDATES:
        built.append((f"scores_c{cands}_b1_one_slot", lambda c=cands: wrap_nki(
            DB.dsa_decode_scores_kernel)(
            fake((1, HEADS, DIM), bf), fake((1, HEADS), f32), fake((1, c + 1, DIM), bf),
            fake((1, 1), i32), fake((1, 1), i32), fake((1, 1), i32), fake((1, DIM), bf),
            c, POOL, DB.score_blocks(1, c, 1), DB.UNROLL_BLOCKS, DB.SOURCE_DIGEST)))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + DB.__file__, flush=True)
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
    with tempfile.TemporaryDirectory(prefix="dsa_batch_frontend_") as scratch:
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


def test_the_front_end_accepts_both_batched_kernels_at_the_served_shapes() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    assert len(entries) == (len(RING_BATCHES) + len(SCORE_CANDIDATES) * len(SCORE_BATCHES) * 2
                            + 1 + len(ONE_SLOT_CANDIDATES))
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
