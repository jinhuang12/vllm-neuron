"""The NKI front end accepts every shipped MoE kernel at the served geometry.

The front end parses a kernel's own source before it places one tile, and it refuses a
call form outside its subset. The simulator never parses that source -- it runs the body
as plain Python -- so no simulator item can see such a refusal. This item compiles the
three kernels instead, with fake operands of the served shapes, inside a child process
that pins the platform target in its environment. The target makes the toolchain read
what it builds for from the environment, so the compile opens no device node, and the
child counts its own open device nodes to prove it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROW = "nki_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
# The child writes no bytecode: it is the first thing to import the compiler in an environment,
# and a cache file left in a shared installation is a change to it that this item does not intend.
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}


def _emit(*fields: object) -> None:
    """Print one reading, one line, so a reader can anchor it by key."""
    print(ROW + "|" + "|".join(str(field) for field in fields), flush=True)


def _flat(text: object, cap: int = 400) -> str:
    """One line of at most `cap` characters, whatever the text carried."""
    return " ".join(str(text).split())[:cap]


def _open_device_nodes() -> int:
    """How many device nodes this process holds open."""
    count = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            if "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}"):
                count += 1
        except OSError:
            pass
    return count


def _compile_each_kernel() -> None:
    """Compile the three kernels in this process and print one row for each."""
    import re
    import time

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_neuron.functional.moe import moe_blockwise_fp8 as live

    experts, hidden, intermediate, tokens, block = 18, 4096, 512, 2048, 256
    blocks = 81
    padded = block * blocks
    bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
    fp32, i32 = torch.float32, torch.int32

    def fake(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="meta")

    cases = (
        ("gate_up", lambda: live.moe_gate_up_blockwise_fp8(
            fake((tokens + 1, hidden), bf16), fake((experts, hidden, 2 * intermediate), fp8),
            fake((experts, 128, 256), fp32), fake((padded,), i32), fake((blocks,), i32), block)),
        ("swiglu", lambda: live.moe_swiglu_transposed(
            fake((padded, 2 * intermediate), fp32), 7.0, 7.0)),
        ("down", lambda: live.moe_down_blockwise_fp8(
            fake((intermediate, padded), fp32), fake((experts, intermediate, hidden), fp8),
            fake((experts, 128, 128), fp32), fake(((tokens + 1) * experts, 1), fp32),
            fake((padded,), i32), fake((blocks,), i32), block, tokens)),
    )
    _emit("venue", f"module={live.__file__}", f"torch={torch.__version__}",
          "cpu_compile=" + os.environ.get("VLLM_NEURON_CPU_COMPILE", "unset"),
          "target=" + os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "unset"),
          "simulator=" + os.environ.get("NKI_SIMULATOR", "unset"))
    _emit("geometry", f"E={experts}", f"H={hidden}", f"I={intermediate}", f"T={tokens}",
          f"block={block}", f"blocks={blocks}", f"padded={padded}")
    for name, call in cases:
        started = time.time()
        message = ""
        try:
            with FakeTensorMode():
                call()
        except Exception as refusal:  # the front end's refusal is the reading
            message = _flat(refusal)
        multiplicity = re.search(r"\[x(\d+)\]", message)
        _emit(f"kernel={name}", f"refused={bool(message)}",
              f"x={multiplicity.group(1) if multiplicity else 0}",
              f"seconds={time.time() - started:.1f}", f"neuron_fds={_open_device_nodes()}",
              f"keyword_expansion={'keyword expansion is not supported' in message}",
              f"diagnostic={message or 'none'}")


def _rows_from_a_child() -> list[str]:
    """Run the compiles in a child of this tree and echo every row it printed."""
    environment = {name: value for name, value in os.environ.items() if name not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-kernel"],
        cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=900, check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    for line in printed:
        print(line, flush=True)
    _emit("child", f"rc={done.returncode}", f"rows={len(printed)}",
          "stderr_tail=" + _flat(done.stderr.splitlines()[-1] if done.stderr.strip() else "none"))
    return printed


def _kernel_rows(printed: list[str]) -> list[dict[str, str]]:
    """One dictionary per kernel row, keyed by the field names the child printed."""
    return [
        dict(field.split("=", 1) for field in line.split("|")[1:])
        for line in printed if line.startswith(ROW + "|kernel=")
    ]


def test_the_front_end_accepts_every_shipped_moe_kernel():
    """The front end compiles the gate-up, activation and down kernels of this tree."""
    printed = _rows_from_a_child()
    rows = _kernel_rows(printed)
    venue = [line for line in printed if line.startswith(ROW + "|venue|")]
    assert venue and f"|module={_ROOT}/" in venue[0], (
        f"the child compiled another tree than {_ROOT}: {venue}")
    assert [row["kernel"] for row in rows] == ["gate_up", "swiglu", "down"], (
        f"the child did not compile the three kernels: {rows}")
    assert {row["neuron_fds"] for row in rows} == {"0"}, (
        f"a compile opened a device node, so it was not a device-free compile: {rows}")
    refused = [
        f"{row['kernel']} x{row['x']}: {row['diagnostic']}"
        for row in rows if row["refused"] == "True"
    ]
    assert refused == [], "the front end refused a shipped kernel: " + " ~ ".join(refused)


if __name__ == "__main__":
    _compile_each_kernel()
