# SPDX-License-Identifier: Apache-2.0
"""Both DSA Hadamard kernels compile for the device at the served shapes.

The simulator runs a kernel body as plain Python, so it never meets the NKI front end, which
refuses call forms the simulator accepts, nor the backend compiler. Each case here builds
``_hadamard128_nki`` or ``_kpool_hadamard_nki`` as ``compile_nki`` does for the served graph --
``CompileKernel`` for the platform target, at the launch's LNC -- through the front end to BIR,
then through ``neuronx-cc`` to a NEFF.

The rotation's cases are the served prefill calls (``tokens * index_n_heads`` rows at 1024 and
2048 tokens), the decode calls at batch 1 and 64, and a ragged row count that reaches every
branch of the tiling; the pooling's are the served prefill calls (one pool per token) and a
ragged pool count. Each runs on one and on two programs, the launches the seam makes.

Nothing here opens a device: the target comes from ``NEURON_PLATFORM_TARGET_OVERRIDE``, which
``test/conftest.py`` pins, and each case checks that the process holds no ``/dev/neuron``
descriptor after its compile. A host without ``neuronx-cc`` fails these cases instead of
skipping them, because a skipped compile check reads as a pass.
"""

from __future__ import annotations

import dataclasses
import os
import shutil

import pytest
import torch
from libtorch_neuronx_lite.compile.platform import get_platform_target
from libtorch_neuronx_lite.nki.nki_compile import _convert_input
from nki.compiler.ncc_driver import compile_bir_to_neff
from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

from vllm_neuron.functional.dsa import kpool_hadamard as mod

#: ``index_n_heads`` on the target checkpoint: the query rotation takes one row per head.
INDEX_N_HEADS = 32

#: Tokens per served prefill chunk: 1024 at ``max_model_len`` 8192, 2048 at 65536.
PREFILL_TOKENS = (1024, 2048)

#: Decode batches at the ends of the served range.
DECODE_BATCHES = (1, 64)

#: SBUF partitions, the row count of one tile.
PARTITIONS = 128

#: A plain launch, and the LNC2 launch the seam makes once there are two rows to split.
PROGRAMS = (1, 2)


def _ragged_rows() -> int:
    """Per program: a whole DMA block, a short block of three tiles, and a 5-row partial tile."""
    return 2 * (PARTITIONS * mod._CHUNK_TILES + PARTITIONS * 3 + 5)


def _ragged_pools() -> int:
    """Per program: a whole block, a one-pool-per-partition block, and a 3-partition tail."""
    return 2 * (PARTITIONS * (mod._POOL_TILE_COLUMNS + 1) + 3)


ROTATION_ROWS = {f"prefill_t{t}": t * INDEX_N_HEADS for t in PREFILL_TOKENS}
ROTATION_ROWS.update({f"decode_b{b}": b * INDEX_N_HEADS for b in DECODE_BATCHES})
ROTATION_ROWS[f"ragged_r{_ragged_rows()}"] = _ragged_rows()

POOL_COUNTS = {f"prefill_t{t}": t for t in PREFILL_TOKENS}
POOL_COUNTS[f"ragged_p{_ragged_pools()}"] = _ragged_pools()


def _meta(shape, dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


def _open_device_nodes() -> int:
    held = [os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")]
    return len([path for path in held if path.startswith("/dev/neuron")])


def _compile_to_neff(kernel, arguments: dict, programs: int, tmp_path, monkeypatch) -> None:
    """Front end to BIR, then ``neuronx-cc`` to a NEFF, as the served graph's compile does."""
    # The compiler leaves log files in its working directory; keep them out of the worktree.
    monkeypatch.chdir(tmp_path)
    compiled = CompileKernel(func=kernel.func, lnc=programs, target=get_platform_target(),
                             artifacts_dir=str(tmp_path))
    neff = tmp_path / "kernel.neff"
    opts = dataclasses.replace(compiled._compile_opts(), output_path=str(neff))
    if shutil.which(opts.neuronx_cc_path) is None:
        pytest.fail(f"{opts.neuronx_cc_path} is not on PATH, so nothing can be compiled")
    inputs = {name: _convert_input(value, name) for name, value in arguments.items()}
    nir = compile_kernel_to_nir(
        compiled, inputs=inputs, compile_opts=opts,
        frontend=compiled._frontend_cls(enable_backend_opt=compiled._enable_backend_opt),
        enable_cache=False, trace_cache=None)
    compile_bir_to_neff(
        opts, nir, input_arrays=[],
        argument_names=[spec.name for spec in nir.descriptor.input_specs],
        output_arg_names=[spec.name for spec in nir.descriptor.output_specs])
    assert neff.stat().st_size > 0
    assert _open_device_nodes() == 0


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("case", ROTATION_ROWS)
def test_the_rotation_compiles_to_a_neff(case, programs, tmp_path, monkeypatch):
    n_rows = ROTATION_ROWS[case]
    arguments = {"x_hbm": _meta((n_rows, mod.INDEX_HEAD_DIM), torch.bfloat16), "n_rows": n_rows}
    _compile_to_neff(mod._hadamard128_nki, arguments, programs, tmp_path, monkeypatch)


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("case", POOL_COUNTS)
def test_the_pooling_compiles_to_a_neff(case, programs, tmp_path, monkeypatch):
    n_pools = POOL_COUNTS[case]
    pool_size = mod.DEFAULT_POOL_SIZE
    rows = (n_pools * pool_size, mod.INDEX_HEAD_DIM)
    arguments = {
        "slot_k_hbm": _meta(rows, torch.bfloat16),
        "slot_score_hbm": _meta(rows, torch.bfloat16),
        "ape_hbm": _meta((pool_size, mod.INDEX_HEAD_DIM), torch.float32),
        "n_pools": n_pools,
        "pool_size": pool_size,
    }
    _compile_to_neff(mod._kpool_hadamard_nki, arguments, programs, tmp_path, monkeypatch)
