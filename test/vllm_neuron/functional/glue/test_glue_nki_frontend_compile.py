# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts every glue kernel at the served geometry, under LNC2.

The simulator runs a kernel body as plain Python, so it cannot see a call form the
front end refuses. This test compiles the kernels instead (the pattern of
``test/vllm_neuron/functional/moe/test_moe_nki_frontend_compile.py``): fake operands of
one rank's served shapes at B = 1 and B = 64, two programs, in a child process whose
environment pins the platform target, so the compile opens no device node.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROW = "glue_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1", "NEURON_LOGICAL_NC_CONFIG": "2"}
KERNELS = ("mhc_pre", "kda_projections", "kda_output", "combine_bf16")
BATCHES = (1, 64)


def _emit(*fields: object) -> None:
    print(ROW + "|" + "|".join(str(field) for field in fields), flush=True)


def _flat(text: object, cap: int = 400) -> str:
    return " ".join(str(text).split())[:cap]


def _open_device_nodes() -> int:
    count = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            if "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}"):
                count += 1
        except OSError:
            pass
    return count


def _compile_each_kernel() -> None:
    import time
    from types import SimpleNamespace

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
    from vllm_neuron.functional.mhc import hyper_connection

    bf16, fp32 = torch.bfloat16, torch.float32
    hidden, streams, mix, width, rank = 4096, 4, 24, 128, 128

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    attn = SimpleNamespace(
        q_proj_weight=fake((width, hidden), bf16), k_proj_weight=fake((width, hidden), bf16),
        v_proj_weight=fake((width, hidden), bf16), b_proj_weight=fake((1, hidden), bf16),
        f_a_proj_weight=fake((rank, hidden), bf16), f_b_proj_weight=fake((width, rank), bf16),
        g_a_proj_weight=fake((rank, hidden), bf16), g_b_proj_weight=fake((width, rank), bf16),
        o_norm_weight=fake((width,), bf16), o_proj_weight=fake((hidden, width), bf16),
        num_kv_heads_per_rank=1, head_dim=width, rms_norm_eps=1e-6)

    def cases(batch):
        return (
            ("mhc_pre", lambda: mhc_pre.mhc_pre_fused(
                fake((batch, streams, hidden), bf16), fake((mix, streams * hidden), bf16),
                fake((3,), fp32), fake((mix,), fp32), rms_eps=1e-6, hc_eps=1e-6,
                post_mult=2.0)),
            ("kda_projections", lambda: kda_projections._KERNELS[2](
                x=fake((batch, hidden), bf16), q_w=attn.q_proj_weight, k_w=attn.k_proj_weight,
                v_w=attn.v_proj_weight, b_w=attn.b_proj_weight, f_a_w=attn.f_a_proj_weight,
                f_b_w=attn.f_b_proj_weight, g_a_w=attn.g_a_proj_weight,
                g_b_w=attn.g_b_proj_weight, DMA_TRANSPOSE=True)),
            ("kda_output", lambda: kda_output._KERNELS[2](
                core=fake((batch, width), fp32), out_gate=fake((batch, width), fp32),
                o_norm=attn.o_norm_weight, o_proj=attn.o_proj_weight, HEADS=1, EPS=1e-6,
                DMA_TRANSPOSE=True)),
            ("combine_bf16", lambda: hyper_connection.hyper_connection_combine(
                fake((batch, hidden), bf16), fake((batch, streams, hidden), bf16),
                fake((batch, streams, 1), fp32), fake((batch, streams, streams), fp32))),
        )

    _emit("venue", f"module={mhc_pre.__file__}", f"torch={torch.__version__}",
          "cpu_compile=" + os.environ.get("VLLM_NEURON_CPU_COMPILE", "unset"),
          "target=" + os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "unset"),
          "lnc=" + os.environ.get("NEURON_LOGICAL_NC_CONFIG", "unset"))
    for batch in BATCHES:
        for name, call in cases(batch):
            started = time.time()
            message = ""
            try:
                with FakeTensorMode():
                    call()
            except Exception as refusal:  # the front end's refusal is the row's reading
                message = _flat(refusal, 4000)
            _emit(f"kernel={name}", f"B={batch}", f"refused={bool(message)}",
                  f"seconds={time.time() - started:.1f}", f"neuron_fds={_open_device_nodes()}",
                  f"diagnostic={message or 'none'}")


def _rows_from_a_child() -> list[str]:
    environment = {name: value for name, value in os.environ.items() if name not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-kernel"],
        cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=1500, check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    for line in printed:
        print(line, flush=True)
    child = "|".join((ROW, "child", f"rc={done.returncode}", f"rows={len(printed)}",
                      "stderr_tail=" + _flat(done.stderr.splitlines()[-1]
                                             if done.stderr.strip() else "none")))
    print(child, flush=True)
    return [*printed, child]


def _kernel_rows(printed: list[str]) -> list[dict[str, str]]:
    rows = []
    for line in printed:
        if not line.startswith(ROW + "|kernel="):
            continue
        head, _, diagnostic = line.partition("|diagnostic=")
        fields = dict(one.split("=", 1) for one in head.split("|")[1:])
        rows.append({**fields, "diagnostic": diagnostic})
    return rows


def test_the_front_end_accepts_every_glue_kernel():
    printed = _rows_from_a_child()
    rows = _kernel_rows(printed)
    child = [line for line in printed if line.startswith(ROW + "|child|")]
    assert child and "|rc=0|" in child[0], f"the compile child did not come back clean: {child}"
    venue = [line for line in printed if line.startswith(ROW + "|venue|")]
    assert venue and f"|module={_ROOT}/" in venue[0], f"compiled another tree: {venue}"
    assert [(row["kernel"], int(row["B"])) for row in rows] == [
        (name, batch) for batch in BATCHES for name in KERNELS], rows
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    refused = [f"{row['kernel']} B={row['B']}: {row['diagnostic']}"
               for row in rows if row["refused"] == "True"]
    assert refused == [], "the front end refused a glue kernel: " + " ~ ".join(refused)


if __name__ == "__main__":
    _compile_each_kernel()
