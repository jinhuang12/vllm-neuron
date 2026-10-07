# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts ``expert_decode_kernel`` at the served shapes.

The simulator runs the kernel body as plain Python, so it cannot see a refusal
of the front end (for example a bit-vector op whose ``dst`` dtype differs from
its ``src`` dtype). This test traces the kernel instead, with fake operands of
one rank's bank at EP=16 (18 experts, H=4096, I=512, 288 router outputs), at
T in {1, 4, 16, 17, 64} (both schedules of ``decode_plan``) and one and two
programs, in a child process pinned to the trn2 target that opens no device
node (as ``test_moe_nki_frontend_compile.py`` does for the prefill kernels).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROW = "expert_decode_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}
TOKENS = (1, 4, 16, 17, 64)
PROGRAMS = (1, 2)


def _open_device_nodes() -> int:
    count = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            if "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}"):
                count += 1
        except OSError:
            pass
    return count


def _compile_each_shape() -> None:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_neuron.functional.moe import fused_fp8

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    print(f"{ROW}|venue|module={fused_fp8.__file__}", flush=True)
    for tokens in TOKENS:
        for programs in PROGRAMS:
            message = "none"
            try:
                with FakeTensorMode():
                    out = fused_fp8._TOKEN_DECODE_EXPERTS[programs](
                        hidden=fake((tokens, 4096), torch.bfloat16),
                        affinity=fake((tokens, 16, 18), torch.float32),
                        rank=fake((1, 1), torch.int32),
                        weights=fake((18, 12, 128, 32, 128), torch.float8_e4m3fn),
                        scales=fake((18, 12, 32), torch.float32),
                        bounds=fake((128, 3), torch.float32),
                        OUT_FP32=False, WEIGHT_FP8=False)
                if tuple(out.shape) != (tokens, 4096):
                    message = f"output shape {tuple(out.shape)}"
            except Exception as refusal:  # the row carries the front end's refusal
                message = " ".join(str(refusal).split())[:2000]
            print(f"{ROW}|T={tokens}|programs={programs}|fds={_open_device_nodes()}"
                  f"|diagnostic={message}", flush=True)


def test_the_front_end_accepts_expert_decode_at_every_schedule():
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "compile"],
        cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=900,
        check=False)
    rows = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    assert done.returncode == 0, done.stderr[-2000:]
    assert rows and f"|module={_ROOT}/" in rows[0], f"compiled another tree: {rows[:1]}"
    shapes = [line for line in rows if "|T=" in line]
    assert len(shapes) == len(TOKENS) * len(PROGRAMS), rows
    assert all("|fds=0|" in line for line in shapes), f"opened a device node: {shapes}"
    refused = [line for line in shapes if not line.endswith("|diagnostic=none")]
    assert refused == [], "the front end refused expert_decode_kernel: " + " ~ ".join(refused)


if __name__ == "__main__":
    _compile_each_shape()
