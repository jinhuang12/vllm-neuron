# SPDX-License-Identifier: Apache-2.0
"""The score GEMM kernel compiles for the device at the served prefill shapes.

The simulator runs a kernel body as plain Python, so it never meets the NKI front end, which
refuses call forms the simulator accepts, nor the backend compiler. Each case here builds
``_score_gemm_nki`` as ``compile_nki`` does for the served graph -- ``CompileKernel`` for the
platform target, at the launch's LNC -- through the front end to BIR, then through ``neuronx-cc``
to a NEFF. The cases are the two served prefill calls, ``[tokens, cands]`` = ``[1024, 2048]`` and
``[2048, 16384]``, and a ragged shape past one score group, each on one and on two programs.

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

from vllm_neuron.functional.dsa import score_gemm as mod

#: ``[tokens, cands]`` of the served prefill calls: a 1024-token chunk at ``max_model_len``
#: 8192 and a 2048-token chunk at 65536, one candidate per pool of four keys.
SERVED = ((1024, 2048), (2048, 16384))

#: A plain launch, and the LNC2 launch the seam makes once there is a token tile per program.
PROGRAMS = (1, mod._LNC2_PROGRAMS)


def _ragged_shape() -> tuple[int, int, int]:
    """``(tokens, heads, cands)``: three token tiles, the last partial, and two score groups,
    the second holding a whole candidate tile and three columns of another."""
    return (2 * mod.TOKEN_TILE + 5, 5,
            mod._GROUP_TILES * mod.CAND_TILE + mod.CAND_TILE + 3)


SHAPES = {f"served_t{t}_c{c}": (t, mod.INDEX_N_HEADS, c) for t, c in SERVED}
SHAPES["ragged_t{}_h{}_c{}".format(*_ragged_shape())] = _ragged_shape()


def _arguments(tokens: int, heads: int, cands: int) -> dict[str, torch.Tensor]:
    """The kernel's operands as shape-and-dtype placeholders, named as the kernel names them."""
    def meta(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    return {"q_hbm": meta((tokens, heads, mod.INDEX_HEAD_DIM), torch.bfloat16),
            "k_hbm": meta((cands, mod.INDEX_HEAD_DIM), torch.bfloat16),
            "w_hbm": meta((tokens, heads), torch.float32)}


def _open_device_nodes() -> int:
    held = [os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")]
    return len([path for path in held if path.startswith("/dev/neuron")])


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_the_kernel_compiles_to_a_neff(shape, programs, tmp_path, monkeypatch):
    # The compiler leaves log files in its working directory; keep them out of the worktree.
    monkeypatch.chdir(tmp_path)
    compiled = CompileKernel(func=mod._score_gemm_nki.func, lnc=programs,
                             target=get_platform_target(), artifacts_dir=str(tmp_path))
    neff = tmp_path / "kernel.neff"
    opts = dataclasses.replace(compiled._compile_opts(), output_path=str(neff))
    if shutil.which(opts.neuronx_cc_path) is None:
        pytest.fail(f"{opts.neuronx_cc_path} is not on PATH, so nothing can be compiled")
    inputs = {name: _convert_input(value, name)
              for name, value in _arguments(*SHAPES[shape]).items()}
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
