# SPDX-License-Identifier: Apache-2.0
"""The backend lowers and neuronx-cc compiles the checkpoint write: aliased kernels, DMAs only.

The write runs in a traced prefill graph on the device, so its proof is the compile, not
the simulator. Each test compiles a graph of :func:`write_kda_checkpoints` calls on the CPU
(``NEURON_LIBTORCH_CPU_COMPILE=1``) under LNC2, at the served checkpoint geometry, with
int32 block ids built on the host, and reads the compile's own work files.

* One write of ``n = 8`` checkpoints through the latent layer's cache view: the backend's
  NKI pass gives the one kernel call a compiled kernel and the operand alias ``{0: 0}``;
  the HLO entry computation aliases its one output to the cache view's parameter; on every
  core every instruction neuronx-cc reports is a data-movement instruction, and no core
  runs a matrix multiply.
* Three writes on one page view, as the KDA layers that share a pool tensor make them,
  none handed the pages another returned: the backend's aliasing pass gives the second
  write the first one's result and the third the second one's, and the graph's one
  aliased output is the third write's result.

The child stops before the executor is built, so no device node is opened, and the test
fails rather than skips when ``neuronx-cc`` is absent.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROW = "checkpoint_write_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
BLOCKS, ROWS_PER_PAGE, HEAD, COUNT, PAGE = 16, 128, 512, 8, 131072
GRAPHS = ("one-write", "three-writes")


def _child(graph: str) -> None:
    """Compile ``graph``; print the NEFF path."""
    import libtorch_neuronx_lite  # noqa: F401  (the package's patches and overrides)
    import libtorch_neuronx_lite.compile.backend as backend
    import torch

    from vllm_neuron.vllm.worker.glm5next_state_banks import (KdaCheckpointLayout,
                                                              write_kda_checkpoints)

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    layout = KdaCheckpointLayout(((3, 384), (1, 128, 128)), (torch.bfloat16, torch.float32),
                                 (0, 2304), PAGE)
    meta = torch.device("meta")

    def rows(count):
        return (torch.empty(count, 3, 384, dtype=torch.bfloat16, device=meta),
                torch.empty(count, 1, 128, 128, dtype=torch.float32, device=meta))

    def one_write(latent, block_ids, conv, recurrent):
        return write_kda_checkpoints(latent, block_ids, [conv, recurrent], layout)

    def three_writes(pages, a, conv_a, rec_a, b, conv_b, rec_b, c, conv_c, rec_c):
        write_kda_checkpoints(pages, a, [conv_a, rec_a], layout)
        write_kda_checkpoints(pages, b, [conv_b, rec_b], layout)
        write_kda_checkpoints(pages, c, [conv_c, rec_c], layout)
        return a + 1

    if graph == "one-write":
        fn = one_write
        args = (torch.empty(BLOCKS * ROWS_PER_PAGE, 1, HEAD, dtype=torch.bfloat16, device=meta),
                torch.empty(COUNT, dtype=torch.int32, device=meta), *rows(COUNT))
    else:
        fn = three_writes
        args = [torch.empty(BLOCKS, PAGE, dtype=torch.int8, device=meta)]
        for _ in range(3):
            args += [torch.empty(2, dtype=torch.int32, device=meta), *rows(2)]
    neff, message = "", ""
    try:
        # The backend's compile function itself: the package registers it by name only
        # where a device node is visible or a CPU mode is on, and the child runs with
        # neither.
        torch.compile(fn, backend=backend.compile, fullgraph=True)(*args)
        message = "the compiled graph ran; build_executable was not replaced"
    except BaseException as caught:  # the compiler's failure is a result here
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:1500] or type(caught).__name__
    print(f"{ROW}|neff={neff or 'none'}|diagnostic={message or 'none'}", flush=True)


def _compile(graph: str, scratch: str) -> pathlib.Path:
    """Compile ``graph`` in a child process; the NEFF's work directory."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(
        NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
        NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
        NEURON_LIBTORCH_CACHE_ROOT=scratch,
        NKI_COMPILE_CACHE_URL=os.path.join(scratch, "nki"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(_ROOT),
        PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
    assert shutil.which("neuronx-cc", path=environment["PATH"]), "neuronx-cc is not installed"
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "child", graph],
        cwd=scratch, env=environment, capture_output=True, text=True, timeout=1500,
        check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
    neff = pathlib.Path(printed[-1].split("|")[1].split("=", 1)[1])
    assert neff.is_file(), printed[-1]
    return neff.parent


def _kernel_calls(work: pathlib.Path) -> list[tuple[str, str]]:
    """``(backend_config, operand_output_aliases)`` of each NKI call after the NKI pass."""
    dump = (work / "passes" / "02_after_nki_kernel_backend_config_pass.txt").read_text()
    return re.findall(r"nki_kernel_wrapper\(kernel_idx = \d+, grid = .*?, "
                      r"backend_config = '([^']*)', operand_output_aliases = (\{[^}]*\})", dump)


def _kernel_operands(work: pathlib.Path) -> list[tuple[str, str]]:
    """``(call, its pages operand)`` of each NKI call after the aliasing pass, in order."""
    dump = (work / "passes" / "03_after_aliasing_output_rewrite.txt").read_text()
    return re.findall(r"%(\w+) : \[num_users=\d+\] = call_function\[target=torch\.ops\."
                      r"higher_order\.nki_kernel_wrapper\].*?args: \(%(\w+),", dump)


def _entry_alias(work: pathlib.Path) -> list[tuple[list[int], list[int]]]:
    """``(output index, aliased parameter's dimensions)`` of each entry alias of the HLO."""
    from libtorch_neuronx_lite.pyhlo import hlo_pb2

    module = hlo_pb2.HloModuleProto()
    module.ParseFromString((work / "graph.hlo").read_bytes())
    entry = next(c for c in module.computations if c.id == module.entry_computation_id)
    params = {i.parameter_number: list(i.shape.dimensions) for i in entry.instructions
              if i.opcode == "parameter"}
    return [(list(e.output_shape_index), params[e.parameter_number])
            for e in module.input_output_alias.entries]


def _core_counts(log: str) -> dict[str, dict[str, int]]:
    """Per core: instructions, data-movement instructions and matrix multiplies."""
    lines = log.splitlines()
    cores: dict[str, dict[str, int]] = {}
    for j, line in enumerate(lines):
        if m := re.search(r"\((nc\d+)/sg\d+\) \[ModuleForkPass\]: Inputs to report_stats: "
                          r".*instructions=(\d+)", line):
            cores.setdefault(m.group(1), {})["instructions"] = int(m.group(2))
        if m := re.search(r"\((nc\d+)/sg\d+\) \[ReportStats\]: Data Movement Statistics", line):
            moved = 0
            for row in lines[j + 1:j + 40]:
                cells = [c.strip() for c in row.split("│")]
                if len(cells) >= 5 and cells[3].isdigit():
                    moved += int(cells[3])
                if row.startswith("└"):
                    break
            cores.setdefault(m.group(1), {})["data_movement"] = moved
        if m := re.search(r"\((nc\d+)/sg\d+\) \[ReportStats\]: MM Stats: #MatMults (\d+)", line):
            cores.setdefault(m.group(1), {})["matmults"] = int(m.group(2))
    return cores


def test_the_write_compiles_to_one_aliased_kernel_of_dma_only():
    with tempfile.TemporaryDirectory(prefix="checkpoint_write_compile_") as scratch:
        work = _compile("one-write", scratch)
        calls = _kernel_calls(work)
        assert len(calls) == 1 and calls[0][0] and calls[0][1] == "{0: 0}", calls
        assert _entry_alias(work) == [([0], [BLOCKS * ROWS_PER_PAGE, 1, HEAD])]
        cores = _core_counts((work / "log-neuron-cc.txt").read_text())
    assert sorted(cores) == ["nc00", "nc01"], sorted(cores)
    for core, counts in cores.items():
        assert counts.get("matmults", 0) == 0, (core, counts)
        assert counts.get("instructions", 0) == counts.get("data_movement", 0), (core, counts)
    assert cores["nc00"]["instructions"] > 0, cores


def test_three_writes_on_one_page_view_are_chained_by_the_backend():
    with tempfile.TemporaryDirectory(prefix="checkpoint_write_compile_") as scratch:
        work = _compile("three-writes", scratch)
        calls = _kernel_calls(work)
        operands = _kernel_operands(work)
        aliases = _entry_alias(work)
    assert len(calls) == 3 and all(c and a == "{0: 0}" for c, a in calls), calls
    assert len(operands) == 3, operands
    (first, pages), (second, after_first), (_, after_second) = operands
    assert pages.startswith("l_pages"), operands
    assert (after_first, after_second) == (first, second), operands
    assert aliases == [([1], [BLOCKS, PAGE])], aliases


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(sys.argv[2])
