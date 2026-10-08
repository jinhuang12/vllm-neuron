# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts both fused KDA decode kernels at the served shapes.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
compiler's front end refuses. This compiles ``kda_fused_decode_tstep_kernel`` at
``T`` in {2, 4, 6} tokens per request (``T = 1 + k`` around the gate's k = 3) and
``B`` in {1, 64} requests, on one and two programs, in both conv layouts and with
and without the padding operands, and the one-token ``kda_fused_decode_kernel`` at
the same batches, inside a child process that pins the platform target, opens no
device node, and proves it parsed bodies by refusing one that reads an undefined
name. The pattern is ``test_decode_batch_frontend_compile.py``'s. It never skips:
without a device the target is pinned, with one the device is not opened.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import nki
import nki.language as nl

ROW = "kda_tstep_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

HEADS, KDIM, TAPS = 1, 128, 4
TOKEN_COUNTS = (2, 4, 6)
BATCHES = (1, 64)
PROGRAMS = (1, 2)


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

    from vllm_neuron.functional.kda import fused_decode as FD

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    width = HEADS * KDIM
    channels = 3 * width

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def grid(entry, programs):
        call = wrap_nki(entry)
        return call[programs] if programs == 2 else call

    def operands(batch, tokens, dim_first, masked):
        rows = batch * tokens
        conv_shape = (batch, channels, TAPS - 1) if dim_first else (batch, TAPS - 1, channels)
        return dict(
            q_in=fake((rows, width), f32), k_in=fake((rows, width), f32),
            v_in=fake((rows, width), f32), raw_gate=fake((rows, width), f32),
            raw_beta=fake((rows, HEADS), f32), conv_state=fake(conv_shape, bf),
            q_w=fake((width, TAPS), bf), k_w=fake((width, TAPS), bf),
            v_w=fake((width, TAPS), bf), a_log=fake((1, HEADS), f32),
            dt_bias=fake((1, width), f32), rec_state=fake((batch, HEADS, KDIM, KDIM), f32),
            start_pos=fake((1, batch), i32),
            real_tokens=fake((1, batch), i32) if masked else None,
            row_mask=fake((batch, tokens), f32) if masked else None,
            lower=-5.0, dim_first=dim_first,
        )

    built = []
    for tokens in TOKEN_COUNTS:
        for batch in BATCHES:
            for programs in PROGRAMS:
                for dim_first in (False, True):
                    for masked in (False, True):
                        built.append((
                            f"tstep_t{tokens}_b{batch}_g{programs}_ds{int(dim_first)}"
                            f"_m{int(masked)}",
                            lambda t=tokens, b=batch, g=programs, d=dim_first, m=masked:
                            grid(FD.kda_fused_decode_tstep_kernel, g)(
                                **operands(b, t, d, m), tokens=t),
                        ))
    for batch in BATCHES:
        for programs in PROGRAMS:
            built.append((
                f"onetoken_b{batch}_g{programs}",
                lambda b=batch, g=programs: grid(FD.kda_fused_decode_kernel, g)(
                    **{k: v for k, v in operands(b, 1, False, True).items()}),
            ))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + FD.__file__, flush=True)
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
    with tempfile.TemporaryDirectory(prefix="kda_tstep_frontend_") as scratch:
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


def test_the_front_end_accepts_both_fused_decode_kernels_at_the_served_shapes() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    assert len(entries) == (len(TOKEN_COUNTS) * len(BATCHES) * len(PROGRAMS) * 2 * 2
                            + len(BATCHES) * len(PROGRAMS))
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
