# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc builds the masked decode attention and the selection's mask entry at the
served shapes, and each compiled program orders every pair of instructions that touch
the same bytes (:mod:`sbuf_order`).

The simulator runs a kernel body as numpy and the front-end compile
(``test_mla_decode_frontend_compile.py``) stops before the backend, so neither sees a
backend refusal, an SBUF allocation that does not fit, or a DMA left unordered against a
write to the same bytes (on the device the selection returned wrong rows from one;
``test_decode_sbuf_order.py``). Each kernel here is compiled as that file compiles one --
the front end to BIR, then ``neuronx-cc`` to a NEFF with the backend's instruction dump
after ``lower_sync`` -- on the launch grid its host side picks at the served LNC, and
every core's dump is checked:

* ``mla_decode_masked_kernel``: one request of 2, 4 and 6 rows (the verify step at
  ``k`` in {1, 3, 5}) over the selecting bucket's 4096-row window (32 pages of 128, 1024
  pools of 4);
* ``dsa_decode_select_mask_kernel``: the same rows, one selection each over the
  window's 1024 candidate pools, 512 selected.

Each compile runs in a child process that sees no Neuron device node (``bwrap`` with a
fresh ``/dev`` on a device host), as ``test_decode_ctx_cpu_compile.py`` does.
"""

from __future__ import annotations

import concurrent.futures
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

from test.vllm_neuron.functional.dsa import sbuf_order

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[4]

LATENT, PAGE, POOL, SELECT_K = 512, 128, 4, 512
WINDOW_PAGES, BANK_PAGES = 32, 64
CANDIDATES = WINDOW_PAGES * PAGE // POOL
MASKED_ROWS = (2, 4, 6)
#: The served logical core: both physical cores of an LNC2 core.
SERVED_LNC = 2

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


def _programs(monkeypatch, kernel: str, rows: int) -> int:
    """The launch grid each kernel's host side picks at the served LNC."""
    from vllm_neuron.functional.attention import mla_decode_masked as MM
    from vllm_neuron.functional.dsa import decode_select as DS
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", str(SERVED_LNC))
    if kernel == "masked":
        assert MM.mla_decode_masked_serves(heads=1, latent=LATENT, page=PAGE,
                                           window=WINDOW_PAGES * PAGE, rows=rows, pool=POOL)
        return MM._programs()
    return DS.decode_select_programs(rows)


def _compile(kernel: str, rows: int, programs: int):
    """Compile one kernel in a child and check every core's ``lower_sync`` dump."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="mla_decode_masked_compile_") as art:
        environment.update(
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NKI_COMPILE_CACHE_URL=f"{art}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, str(_HERE), art, kernel, str(rows), str(programs)],
            cwd=art, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
        assert done.returncode == 0, (kernel, rows, done.stdout[-2000:], done.stderr[-3000:])
        assert os.path.isfile(os.path.join(art, "kernel.neff")), (kernel, rows)
        dumps = sorted(glob.glob(f"{art}/nc0*/sg00/bir_debug.*after-lower_sync*.json")
                       or glob.glob(f"{art}/sg00/bir_debug.*after-lower_sync*.json"))
        assert len(dumps) == programs, (kernel, rows, dumps)
        return [(pathlib.Path(d).relative_to(art).parts[0], sbuf_order.check_dump(d))
                for d in dumps]


@_needs_compiler
def test_the_served_shapes_compile_with_every_access_ordered(monkeypatch):
    shapes = [("masked", rows, _programs(monkeypatch, "masked", rows)) for rows in MASKED_ROWS]
    shapes += [("select_mask", rows, _programs(monkeypatch, "select_mask", rows))
               for rows in MASKED_ROWS]
    with concurrent.futures.ThreadPoolExecutor(len(shapes)) as pool:
        results = dict(zip(shapes, pool.map(lambda s: _compile(*s), shapes)))
    bad = {}
    for (kernel, rows, programs), cores in results.items():
        label = f"{kernel} rows={rows} programs={programs}"
        for core, found in cores:
            print(f"{label} {core}: {found.summary()}")
        dirty = [(core, found.summary(), found.unsync[:4], found.library_alias[:4],
                  found.stepped_dma[:4], found.early_waits[:4])
                 for core, found in cores if not found.clean]
        if dirty:
            bad[label] = dirty
    assert not bad, {"unordered accesses": bad}


def _arguments(kernel: str, rows: int) -> tuple:
    """``(jitted kernel, {parameter: shape and dtype or python value})`` at the served shape."""
    from vllm_neuron.functional.attention import mla_decode_masked as MM
    from vllm_neuron.functional.dsa import decode_select as DS
    if kernel == "masked":
        return MM.mla_decode_masked_kernel, {
            "q_hbm": ((rows, 1, LATENT), "bfloat16"),
            "bank_hbm": ((BANK_PAGES * PAGE, LATENT), "bfloat16"),
            "table_hbm": ((1, WINDOW_PAGES), "int32"), "pos_hbm": ((1,), "int32"),
            "written_hbm": ((rows, LATENT), "bfloat16"),
            "sel_hbm": ((rows, CANDIDATES), "float32"), "softmax_scale": 0.0625,
            "page_size": PAGE, "pool_size": POOL, "source_digest": MM.SOURCE_DIGEST}
    return DS.dsa_decode_select_mask_kernel, {
        "bounded_hbm": ((rows, CANDIDATES), "float32"), "select_k": SELECT_K,
        "fold_cap": int(DS.select_fold_cap(CANDIDATES)), "source_digest": DS.SOURCE_DIGEST}


def _compile_child(art: str, kernel: str, rows: int, programs: int):
    """Child process: compile one kernel to a NEFF with the ``lower_sync`` dump in ``art``
    (``test_decode_sbuf_order.py``'s child)."""
    import inspect
    from dataclasses import replace

    import nki.language as nl
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    from nki.language.buffers import shared_hbm
    from nki.language.tensor import NkiTensor

    jitted, arguments = _arguments(kernel, rows)
    func = jitted.func
    inputs = {}
    for name in inspect.signature(func).parameters:
        value = arguments[name]
        inputs[name] = (NkiTensor(name=name, shape=value[0], dtype=getattr(nl, value[1]),
                                  storage=None, buffer=shared_hbm)
                        if isinstance(value, tuple) else value)
    compiled = CompileKernel(func=func, lnc=programs, target="trn2", artifacts_dir=art)
    opts = replace(compiled._compile_opts(), output_path=os.path.join(art, "kernel.neff"),
                   neuronx_cc_args=("--internal-print-after=lower_sync",))
    bir = compile_to_bir(func, frontend=ParserFrontend(enable_backend_opt=False),
                         inputs=inputs, compile_opts=opts, enable_cache=False)
    compile_bir_to_neff(opts, bir, input_arrays=[],
                        argument_names=[s.name for s in bir.descriptor.input_specs],
                        output_arg_names=[s.name for s in bir.descriptor.output_specs])


if __name__ == "__main__":
    _compile_child(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:5]))
