# SPDX-License-Identifier: Apache-2.0
"""The DSA decode kernels whose program grows with the context -- the ring step, the
scores and the selection -- as neuronx-cc compiles them, order every pair of
instructions that touch the same bytes (:mod:`sbuf_order`).

On the device the selection returned wrong rows where the compiled program placed a
gather's output over its own input, or left a DMA unordered against a write to the same
SBUF bytes; the CPU simulator runs a kernel body as numpy and sees neither. Here each
kernel is compiled the way ``libtorch_neuronx_lite.nki.nki_compile`` compiles one
(``CompileKernel(func, lnc, target)._compile_opts()``: the SDK's default pipeline), with
the backend's instruction dump after ``lower_sync``, on the launch grid its host side
picks, and every core's dump is checked.

The compiled program is a function of the kernel source and the SDK, so a clean shape
stays clean until either changes: :data:`CLEAN_ON` names the SDK the grid was clean on,
and a failure prints it beside the installed one. Shapes, at the served LNC:

* the served lines (always, about a minute): every ``num_seqs`` bucket up to
  :data:`MAX_NUM_SEQS` at an 8192-token ``max_model_len``, and one request at 4096 and
  32768 tokens;
* the grid (``DSA_SBUF_ORDER_GRID=1``, about three minutes): every bucket at every
  ``max_model_len`` in :data:`GRID_CONTEXTS`.

Each compile runs in a child process that sees no Neuron device node (``bwrap`` with a
fresh ``/dev`` on a device host), as ``test_decode_ctx_cpu_compile.py`` does. The first
tests run the check itself on hand-built dumps, one per finding it reports.
"""

from __future__ import annotations

import concurrent.futures
import glob
import importlib.metadata
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

from test.vllm_neuron.functional.dsa import sbuf_order
from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[4]
_CFG = decode_config()
POOL = int(_CFG.index_kpool)
#: Pools the selection keeps: the model's top-k tokens in whole pools.
K = int(_CFG.index_topk) // POOL
#: SDK packages the compiled program depends on, and the versions the grid was clean on.
CLEAN_ON = {"nki": "0.6.0+31049202112.g85070674",
            "neuronx-cc": "2.27.5334.0+f702b353",
            "libtorch-neuronx-lite": "2.11.0.1.0.1284+f49d8626"}
#: The served logical core: both physical cores of an LNC2 core.
SERVED_LNC = 2
#: The largest ``max_num_seqs`` served; the buckets are its defaults.
MAX_NUM_SEQS = 64
#: ``max_model_len`` of the line every bucket is checked at, and of the lone requests.
LINE_CONTEXT = 8192
SINGLE_CONTEXTS = (4096, 32768)
#: ``max_model_len`` of the grid: from the narrowest axis the selection runs at to the
#: widest whose compile time the indexer report measures.
GRID_CONTEXTS = (4096, 8192, 16384, 32768, 65536)
#: Concurrent compile children.
_WORKERS = 8
#: Settings of a CPU test run the compile child must not inherit (as
#: ``test_decode_ctx_cpu_compile.py``).
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_LIBTORCH_REMOTE_CACHE")

_DEVICE_NODES = bool(glob.glob("/dev/neuron*"))
_BWRAP = shutil.which("bwrap")
_NO_DEVICE = (() if not _DEVICE_NODES else
              (_BWRAP, "--dev-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--"))
_needs_compiler = pytest.mark.skipif(
    (_DEVICE_NODES and _BWRAP is None)
    or (shutil.which("neuronx-cc") is None
        and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists()),
    reason="neuronx-cc is not installed, or this host has /dev/neuron* and no bwrap to hide "
           "them from the compile")


# ---------------------------------------------------------------------------------------
# The check on hand-built dumps
# ---------------------------------------------------------------------------------------

#: A 4-partition fp32 tile of 16 columns (64 bytes a partition).
_PARTS, _COLS = 4, 16


def _loc(name, addr=0, base=0):
    return {"name": name, "type": "SB", "addr": addr, "dims": [_PARTS, _COLS * 4],
            "base": base, "bank": 0}


def _tile(memref, partition_step=_COLS):
    return {"kind": "physical_ap", "memref": memref, "dtype": "float32",
            "ap": [[partition_step, _PARTS], [1, _COLS]], "offset": 0}


def _inst(name, engine, opcode="TensorTensor", ins=(), outs=(), waits=(), updates=(),
          queue=None):
    return {"name": name, "engine": engine, "opcode": opcode, "ins": list(ins),
            "outs": list(outs), "queue": queue,
            "sync_info": {"on_wait": [{"from": w, "id": i, "wait_value": v}
                                      for w, i, v in waits],
                          "on_update": [{"id": i, "update_value": v} for i, v in updates]}}


def _found(tmp_path, instructions, locs):
    dump = {"functions": [{"allocations": [{"kind": "Internal", "memorylocations": locs}],
                           "blocks": [{"instructions": instructions}]}],
            "queues": []}
    path = tmp_path / "dump.json"
    path.write_text(json.dumps(dump))
    return sbuf_order.check_dump(str(path))


def test_two_engines_writing_one_tile_without_a_wait_race(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", outs=[_tile("x")]),
                              _inst("b", "Pool", outs=[_tile("x")])], [_loc("x")])
    assert len(found.unsync) == 1 and found.unsync[0].startswith("WAW a ")
    assert not found.clean


def test_a_semaphore_wait_orders_the_pair(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", outs=[_tile("x")], updates=[(7, 1)]),
                              _inst("b", "Pool", ins=[_tile("x")], waits=[("a", 7, 1)])],
                   [_loc("x")])
    assert found.clean and found.ordered == 1


def test_a_wait_below_the_update_count_is_not_an_edge(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", updates=[(7, 1)]),
                              _inst("c", "DVE", outs=[_tile("x")], updates=[(7, 1)]),
                              _inst("b", "Pool", ins=[_tile("x")], waits=[("c", 7, 1)])],
                   [_loc("x")])
    assert len(found.early_waits) == 1 and not found.clean


def test_a_semaphore_reset_restarts_the_count(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", updates=[(7, 1)]),
                              dict(_inst("reset", "ALL", opcode="GroupResetSemaphores"),
                                   sema_group=[7]),
                              _inst("c", "DVE", outs=[_tile("x")], updates=[(7, 1)]),
                              _inst("b", "Pool", ins=[_tile("x")], waits=[("c", 7, 1)])],
                   [_loc("x")])
    assert found.clean


def test_one_engine_and_a_barrier_order_their_accesses(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", outs=[_tile("x")]),
                              _inst("b", "DVE", ins=[_tile("x")]),
                              _inst("all", "ALL", opcode="AllEngineBarrier"),
                              _inst("c", "Pool", outs=[_tile("x")])], [_loc("x")])
    assert found.clean and found.ordered == 3


def test_tiles_at_one_address_in_two_quadrants_do_not_overlap(tmp_path):
    found = _found(tmp_path, [_inst("a", "DVE", outs=[_tile("x")]),
                              _inst("b", "Pool", outs=[_tile("y")])],
                   [_loc("x", base=0), _loc("y", base=32)])
    assert found.clean and found.ordered == 0


def test_a_dma_completes_only_through_a_wait(tmp_path):
    copy = _inst("d", "SP", opcode="DMACopy", outs=[_tile("x")], queue="q", updates=[(3, 1)])
    unordered = _found(tmp_path, [copy, _inst("r", "SP", ins=[_tile("x")])], [_loc("x")])
    waited = _found(tmp_path, [copy, _inst("r", "SP", ins=[_tile("x")], waits=[("d", 3, 1)])],
                    [_loc("x")])
    assert len(unordered.unsync) == 1 and waited.clean


def test_a_pair_ordered_by_one_queue_alone_counts_apart(tmp_path):
    # r waits for e only; d completes first because d and e share a queue.
    found = _found(tmp_path, [_inst("d", "SP", opcode="DMACopy", outs=[_tile("x")], queue="q"),
                              _inst("e", "SP", opcode="DMACopy", outs=[_tile("y")], queue="q",
                                    updates=[(3, 1)]),
                              _inst("r", "Pool", ins=[_tile("x")], waits=[("e", 3, 1)])],
                   [_loc("x"), _loc("y", addr=_COLS * 4)])
    assert found.clean and found.queue_order == 1 and found.ordered == 0


def test_a_library_op_over_its_own_input_is_reported(tmp_path):
    found = _found(tmp_path, [_inst("g", "Pool", opcode="Gather", ins=[_tile("x")],
                                    outs=[_tile("x")])], [_loc("x")])
    assert found.library_alias and not found.clean


def test_a_dma_over_stepped_partitions_is_reported(tmp_path):
    stepped = _tile("x", partition_step=2 * _COLS)
    loc = dict(_loc("x"), dims=[2 * _PARTS, _COLS * 4])
    found = _found(tmp_path, [_inst("d", "SP", opcode="DMACopy", ins=[stepped], queue="q")],
                   [loc])
    assert found.stepped_dma and not found.clean


# ---------------------------------------------------------------------------------------
# The compiled kernels
# ---------------------------------------------------------------------------------------


def _programs(monkeypatch, kernel: str, batch: int, candidates: int) -> int:
    """The launch grid each kernel's host side picks at the served LNC."""
    from vllm_neuron.functional.dsa import decode_batch as DB
    from vllm_neuron.functional.dsa import decode_select as DS
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", str(SERVED_LNC))
    if kernel == "select":
        return DS.decode_select_programs(batch)
    if kernel == "ring":
        return DB._programs(batch, 2)
    return DB._programs(batch, -(-candidates // DB.PARTITIONS))


def _shapes(monkeypatch, contexts_by_batch):
    """``(kernel, B, candidates, programs)``: the ring step once per ``B`` (it has no
    candidate axis), the scores and the selection at every context."""
    shapes = []
    for batch, contexts in contexts_by_batch:
        if not any(s[:2] == ("ring", batch) for s in shapes):
            shapes.append(("ring", batch, contexts[0] // POOL,
                           _programs(monkeypatch, "ring", batch, 0)))
        for kernel in ("scores", "select"):
            shapes += [(kernel, batch, ctx // POOL,
                        _programs(monkeypatch, kernel, batch, ctx // POOL))
                       for ctx in contexts]
    return shapes


def _sdk() -> dict:
    out = {}
    for name in CLEAN_ON:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _compile(kernel: str, batch: int, candidates: int, programs: int):
    """Compile one kernel at ``(batch, candidates)`` and check every core's dump."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="dsa_sbuf_order_") as art:
        environment.update(
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NKI_COMPILE_CACHE_URL=f"{art}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, str(_HERE), art, kernel, str(batch),
             str(candidates), str(programs)],
            cwd=art, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
        assert done.returncode == 0, (kernel, batch, candidates, done.stdout[-2000:],
                                      done.stderr[-3000:])
        dumps = sorted(glob.glob(f"{art}/nc0*/sg00/bir_debug.*after-lower_sync*.json")
                       or glob.glob(f"{art}/sg00/bir_debug.*after-lower_sync*.json"))
        assert len(dumps) == programs, (kernel, batch, candidates, dumps)
        return [(pathlib.Path(d).relative_to(art).parts[0], sbuf_order.check_dump(d))
                for d in dumps]


def _check(shapes):
    with concurrent.futures.ThreadPoolExecutor(_WORKERS) as pool:
        results = dict(zip(shapes, pool.map(lambda s: _compile(*s), shapes)))
    bad = {}
    for (kernel, batch, candidates, programs), cores in results.items():
        label = f"{kernel} B={batch} candidates={candidates} programs={programs}"
        for core, f in cores:
            print(f"{label} {core}: {f.summary()}")
        dirty = [(core, f.summary(), f.unsync[:4], f.library_alias[:4], f.stepped_dma[:4],
                  f.early_waits[:4]) for core, f in cores if not f.clean]
        if dirty:
            bad[label] = dirty
    assert not bad, {"unordered accesses": bad, "installed": _sdk(), "clean on": CLEAN_ON}


@_needs_compiler
def test_the_served_lines_compile_with_every_access_ordered(monkeypatch):
    batches = get_default_num_seqs_buckets(MAX_NUM_SEQS)
    _check(_shapes(monkeypatch, [(b, (LINE_CONTEXT,)) for b in batches]
                   + [(1, SINGLE_CONTEXTS)]))


@_needs_compiler
@pytest.mark.skipif(os.environ.get("DSA_SBUF_ORDER_GRID") != "1",
                    reason="the grid takes about three minutes: DSA_SBUF_ORDER_GRID=1")
def test_the_grid_compiles_with_every_access_ordered(monkeypatch):
    batches = get_default_num_seqs_buckets(MAX_NUM_SEQS)
    _check(_shapes(monkeypatch, [(b, GRID_CONTEXTS) for b in batches]))


def _compile_child(art: str, kernel: str, batch: int, candidates: int, programs: int):
    """Child process: compile one kernel to a NEFF with the ``lower_sync`` dump in ``art``,
    on the arguments the trn2 harness gives it (``benchmark_dsa_decode_ctx._kernel_call``)."""
    import inspect
    from dataclasses import replace

    import nki.language as nl
    import torch
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    from nki.language.buffers import shared_hbm
    from nki.language.tensor import NkiTensor

    from test.hardware.benchmark_dsa_decode_ctx import _kernel_call

    jitted, arguments, _ = _kernel_call("after", kernel, batch, candidates, programs)
    func = jitted.func
    dtypes = {torch.float32: nl.float32, torch.bfloat16: nl.bfloat16, torch.int32: nl.int32}
    inputs = {}
    for name in inspect.signature(func).parameters:
        value = arguments[name]
        inputs[name] = (NkiTensor(name=name, shape=tuple(value.shape), dtype=dtypes[value.dtype],
                                  storage=None, buffer=shared_hbm)
                        if isinstance(value, torch.Tensor) else value)
    compiled = CompileKernel(func=func, lnc=programs, target="trn2", artifacts_dir=art)
    opts = replace(compiled._compile_opts(), output_path=os.path.join(art, "kernel.neff"),
                   neuronx_cc_args=("--internal-print-after=lower_sync",))
    bir = compile_to_bir(func, frontend=ParserFrontend(enable_backend_opt=False),
                         inputs=inputs, compile_opts=opts, enable_cache=False)
    compile_bir_to_neff(opts, bir, input_arrays=[],
                        argument_names=[s.name for s in bir.descriptor.input_specs],
                        output_arg_names=[s.name for s in bir.descriptor.output_specs])


if __name__ == "__main__":
    _compile_child(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6]))
