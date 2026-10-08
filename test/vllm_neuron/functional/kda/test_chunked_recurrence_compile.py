# SPDX-License-Identifier: Apache-2.0
"""The KDA chunked-recurrence kernels compile for the device at the served shapes.

The simulator runs a kernel body as plain Python, so it never meets the NKI front
end, which refuses call forms the simulator accepts, nor the backend compiler.
These cases build each kernel as ``compile_nki`` does for the served graph --
``CompileKernel`` for the platform target, at the launch's LNC -- through the
front end to BIR, then through ``neuronx-cc`` to a NEFF, at the two served
prefill shapes. Nothing here opens a device: the target comes from
``NEURON_PLATFORM_TARGET_OVERRIDE``, which ``test/conftest.py`` pins.

A host without ``neuronx-cc`` fails these cases instead of skipping them, because
a skipped compile check reads as a pass.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest
import torch
from libtorch_neuronx_lite.compile.platform import get_platform_target
from libtorch_neuronx_lite.nki.nki_compile import _convert_input
from nki.compiler.ncc_driver import compile_bir_to_neff
from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

from vllm_neuron.functional.kda import chunked_recurrence as cr

#: The chunk width the KDA layer resolves and the key/value widths one TP=64 rank
#: sees, as in ``test_chunked_recurrence_packed``.
CHUNK = 8
KDIM = 128
VDIM = 128

#: The served prefill chunk sizes, in tokens; a call takes ``tokens // CHUNK``
#: chunks.
SERVED_PREFILL_TOKENS = (1024, 2048)

#: The served runtime's LNC, and the one-program launch every other runtime takes.
SERVED_LNC = cr.LNC2_PROGRAMS
ONE_PROGRAM = 1


def _meta(*shape):
    return torch.empty(shape, dtype=torch.float32, device="meta")


def _intra_arguments(n_chunks):
    consts = cr.chunk_constants(CHUNK, device="meta")
    return {
        "q_hbm": _meta(n_chunks, CHUNK, KDIM), "k_hbm": _meta(n_chunks, CHUNK, KDIM),
        "v_hbm": _meta(n_chunks, CHUNK, VDIM), "beta_hbm": _meta(n_chunks, CHUNK, 1),
        "gk_hbm": _meta(n_chunks, CHUNK, KDIM), "triu_hbm": consts.triu_ones,
        "eye_hbm": consts.eye, "mask_lower_hbm": consts.mask_lower,
        "last_row_hbm": consts.last_row,
    }


def _inter_arguments(n_chunks):
    consts = cr.inter_chunk_constants(CHUNK, KDIM, VDIM, device="meta")
    return {
        "kg_hbm": _meta(n_chunks, CHUNK, KDIM), "w_hbm": _meta(n_chunks, CHUNK, KDIM),
        "u_hbm": _meta(n_chunks, CHUNK, VDIM), "gk_hbm": _meta(n_chunks, CHUNK, KDIM),
        "q_hbm": _meta(n_chunks, CHUNK, KDIM), "aqk_hbm": _meta(n_chunks, CHUNK, CHUNK),
        "triu_hbm": consts.triu_ones, "last_col_hbm": consts.last_col,
        "state_init_hbm": consts.state_init,
    }


KERNELS = {
    "intra": (cr.kda_intra_chunk_kernel, _intra_arguments),
    "inter": (cr.kda_inter_chunk_kernel, _inter_arguments),
}

CASES = [
    (kernel, tokens, SERVED_LNC) for kernel in KERNELS for tokens in SERVED_PREFILL_TOKENS
] + [(kernel, SERVED_PREFILL_TOKENS[0], ONE_PROGRAM) for kernel in KERNELS]


@pytest.mark.parametrize("kernel,tokens,lnc", CASES)
def test_the_kernel_compiles_to_a_neff(kernel, tokens, lnc, tmp_path):
    entry, arguments = KERNELS[kernel]
    compiled = CompileKernel(
        func=entry.func, lnc=lnc, target=get_platform_target(),
        artifacts_dir=str(tmp_path),
    )
    neff = tmp_path / "kernel.neff"
    opts = dataclasses.replace(compiled._compile_opts(), output_path=str(neff))
    if shutil.which(opts.neuronx_cc_path) is None:
        pytest.fail(f"{opts.neuronx_cc_path} is not on PATH, so nothing can be compiled")
    inputs = {
        name: _convert_input(value, name)
        for name, value in arguments(tokens // CHUNK).items()
    }
    nir = compile_kernel_to_nir(
        compiled, inputs=inputs, compile_opts=opts,
        frontend=compiled._frontend_cls(enable_backend_opt=compiled._enable_backend_opt),
        enable_cache=False, trace_cache=None,
    )
    compile_bir_to_neff(
        opts, nir, input_arrays=[],
        argument_names=[spec.name for spec in nir.descriptor.input_specs],
        output_arg_names=[spec.name for spec in nir.descriptor.output_specs],
    )
    assert neff.stat().st_size > 0
