# SPDX-License-Identifier: Apache-2.0
"""The wide ``T``-row scores kernel compiles to a NEFF at the 1M / DCP 8 served calls.

The simulator and the front-end compile (``test_decode_trow_wide_frontend_compile.py``)
stop before the backend, so they see neither the SB allocator nor the program's
instruction order. This compiles ``dsa_decode_scores_rows_wide_kernel`` through
neuronx-cc at the two calls the 1M / DCP 8 decode graph makes (32768 candidates on the
one-slot bank, the query rotated on chip, one program; one row at row 0's position and
four rows at each row's own) and, on each core's ``lower_sync`` dump, checks what
``test_decode_select_walk.py`` checks for the walk kernel:

* the SB allocator spills nothing and no tile meets a runtime reservation;
* the addresses the tiles cover fit ``SBUF_BYTES_PER_PARTITION``. The sum of every
  location, all live at once, is printed beside it: the score and key tiles of four
  rows exceed the bound in that sum, and the allocator reuses their addresses;
* every pair of instructions that touch the same bytes is ordered (:mod:`sbuf_order`), and
  the dependency checker's one-pass and loop-carried verdicts are CLEAN.

Each compile runs in a child process that sees no Neuron device node (``bwrap`` with a
fresh ``/dev`` on a device host).
"""

from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

from test.vllm_neuron.functional.dsa import sbuf_order
from test.vllm_neuron.functional.dsa.test_decode_select_walk import (
    _DROP,
    _NO_DEVICE,
    _depcheck,
    _needs_compiler,
    _sb_spills,
    _sbuf_allocation,
)
from vllm_neuron.utils.neuron_utils import SBUF_BYTES_PER_PARTITION

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[4]
HEADS, DIM, POOL = 32, 128, 4
#: The served axis of the 1M / DCP 8 form: 131072 tokens per rank over pools of 4.
CANDIDATES = 32768
#: Query rows of the served calls: a one-row step and a verify step at MTP k = 3.
SERVED_ROWS = (1, 4)


def _compile(rows: int):
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    with tempfile.TemporaryDirectory(prefix="dsa_trow_wide_backend_") as art:
        environment.update(
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NKI_COMPILE_CACHE_URL=f"{art}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, "-B", str(_HERE), art, str(rows)],
            cwd=art, env=environment, capture_output=True, text=True, timeout=1800,
            check=False)
        assert done.returncode == 0, (rows, done.stdout[-2000:], done.stderr[-3000:])
        assert f"module={_ROOT}/" in done.stdout, done.stdout[-2000:]
        assert pathlib.Path(art, "kernel.neff").is_file(), sorted(os.listdir(art))
        dumps = sorted(pathlib.Path(art).glob("sg00/bir_debug.*after-lower_sync*.json"))
        assert len(dumps) == 1, (rows, dumps)
        dump = str(dumps[0])
        return _sbuf_allocation(dump), sbuf_order.check_dump(dump), *_depcheck(dump), \
            _sb_spills(art)


@_needs_compiler
@pytest.mark.parametrize("rows", SERVED_ROWS, ids=[f"T{t}" for t in SERVED_ROWS])
def test_the_served_calls_compile_clean_with_every_access_ordered(rows):
    sbuf, order, one_pass, loop_carried, spills = _compile(rows)
    label = f"wide scores T={rows} candidates={CANDIDATES}"
    bound = SBUF_BYTES_PER_PARTITION
    print(f"{label}: SB allocator spills {spills}; SBUF {sbuf.covered} B per partition of "
          f"addresses covered, bound SBUF_BYTES_PER_PARTITION {bound}, max(value - bound) "
          f"{sbuf.covered - bound}; {sbuf.total} B if every location were live at once; "
          f"locations in a reservation {len(sbuf.in_reserved)}; widest locations "
          f"(B per partition, addr, name) {sbuf.widest}")
    print(f"{label}: {order.summary()}")
    print(f"{label}: {one_pass}")
    print(f"{label}: {loop_carried}")
    assert spills and max(spills) == 0, (label, spills)
    assert not sbuf.in_reserved, (label, sbuf.in_reserved[:4])
    assert sbuf.covered <= bound, (label, sbuf.covered, bound)
    assert order.clean, (label, order.summary(), order.unsync[:4], order.library_alias[:4],
                         order.stepped_dma[:4], order.early_waits[:4])
    assert one_pass.startswith("VERDICT: CLEAN"), (label, one_pass)
    assert loop_carried.startswith("VERDICT (loop-carried): CLEAN"), (label, loop_carried)


def _compile_child(art: str, rows: int):
    """Child process: compile the wide kernel to a NEFF with the ``lower_sync`` dump in
    ``art``, on the operands the 1M / DCP 8 decode graph hands it."""
    from dataclasses import replace

    import nki.language as nl
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    from nki.language.buffers import shared_hbm
    from nki.language.tensor import NkiTensor

    import vllm_neuron
    from vllm_neuron.functional.dsa import decode_trow_wide as TW

    print(f"module={pathlib.Path(vllm_neuron.__file__).resolve()}", flush=True)
    func = TW.dsa_decode_scores_rows_wide_kernel.func
    arguments = {"q_hbm": ((rows, HEADS, DIM), nl.bfloat16), "w_hbm": ((rows, HEADS), nl.float32),
                 "bank_hbm": ((1, CANDIDATES + 1, DIM), nl.bfloat16),
                 "slots_hbm": ((1, 1), nl.int32), "pos_hbm": ((rows, 1), nl.int32),
                 "pooled_hbm": ((rows, DIM), nl.bfloat16), "candidates": CANDIDATES,
                 "pool_size": POOL, "rows": rows, "rotate": True,
                 "source_digest": TW.SOURCE_DIGEST}
    inputs = {}
    for name in inspect.signature(func).parameters:
        value = arguments[name]
        inputs[name] = (NkiTensor(name=name, shape=value[0], dtype=value[1], storage=None,
                                  buffer=shared_hbm) if isinstance(value, tuple) else value)
    compiled = CompileKernel(func=func, lnc=1, target="trn2", artifacts_dir=art)
    opts = replace(compiled._compile_opts(), output_path=os.path.join(art, "kernel.neff"),
                   neuronx_cc_args=("--internal-print-after=lower_sync",))
    bir = compile_to_bir(func, frontend=ParserFrontend(enable_backend_opt=False),
                         inputs=inputs, compile_opts=opts, enable_cache=False)
    compile_bir_to_neff(opts, bir, input_arrays=[],
                        argument_names=[s.name for s in bir.descriptor.input_specs],
                        output_arg_names=[s.name for s in bir.descriptor.output_specs])


if __name__ == "__main__":
    _compile_child(sys.argv[1], int(sys.argv[2]))
