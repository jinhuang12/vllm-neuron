# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for the mHC combine kernel COMPILING, not merely simulating.

Acceptance command (the harness this suite already uses, this file substituted)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \
    python -m pytest \
      test/vllm_neuron/functional/mhc/test_hyper_connection_specializes_123.py \
      --timeout 900 -p no:cacheprovider -s -rA

What this file measures that the landed acceptance could not
-----------------------------------------------------------
The landed acceptance runs this kernel on the NKI SIMULATOR, which executes the
body as ordinary python. The server first captures a graph of the whole model and
hands every NKI seam in it to the NKI COMPILER to specialise. A body the
simulator runs happily can be one the compiler refuses, and this one was --
``failed to specialize NKI kernel: Collected 1 different diagnostics: - [x1]
error: unsupported expression``, at the first kernel call of every layer's
forward, on every rank, so no graph of the model could be captured at all.

Three items, NO ``parametrize``, so the collected count is derivable from this
file before it runs:

1. SPECIALISATION through the seam's own wrapper under the server's capture
   regime, at the shape the failing run reported. The module's kernel must
   capture AND the refused body -- kept below as a verbatim test-only copy --
   must still be refused in the same child, with the vendor's own text. Without
   that half, a venue which never reached the compiler reads green.

   THAT VENUE IS A CHILD PROCESS, and it has to be. ``NKI_SIMULATOR=1`` makes the
   wrapper interpret the kernel body instead of compiling it, and the venue is
   fixed at import; items 2 and 3 need the simulator, item 1 needs the compiler.
   So item 1 spawns one child under the pins the failing server run used
   (``VLLM_NEURON_CPU_COMPILE=1 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1``, the
   platform target, the simulator variables UNSET), builds META tensors, and
   captures through ``torch.compile`` with the graph-capture backend. The child
   prints its stage and the compiler's message; this file reads them.
2. BIT-IDENTITY against that refused body on the simulator venue, and exact
   identity against the module's torch oracle under the pass-through pattern
   where both are lossless. No tolerance is used or authored here.
3. THE ROUTE: one NKI dispatch per call, no torch fallback, off the module's own
   counters -- a body that answered in torch would satisfy item 2.

THE BASE ARM. These bytes also run against the tree this change was made on,
where the refusal is the EXPECTED reading: under ``GLM53F_123_EXPECT_BASE=1``
item 1 requires the module's kernel to be refused and to say why. Item 2 holds
there because the refused body IS that tree's body.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator

import vllm_neuron.functional.mhc.hyper_connection as HC_MODULE
from vllm_neuron.functional.mhc.hyper_connection import (
    MHC_STREAMS,
    PARTITION_MAX,
    dispatch_counters,
    hyper_connection_combine,
    hyper_connection_kernel,
    hyper_connection_torch_oracle,
    reset_dispatch_counters,
)

S = MHC_STREAMS  # 4, the target's `hc_mult`, read off the module rather than typed

#: The vendor's own two fragments, required of the base arm and of the copy.
WANT_REFUSAL = "failed to specialize NKI kernel"
WANT_DIAGNOSTIC = "unsupported expression"

#: Which tree these bytes run against; the file cannot tell, so the env names it.
EXPECT_BASE = os.environ.get("GLM53F_123_EXPECT_BASE") == "1"

#: The shape the failing capture reported: 2048 is the prefill bucket and 16 whole
#: row tiles at the model's hidden extent. One shape, because each arm is a compile.
SPECIALISE_ROWS = 2048
SPECIALISE_HIDDEN = 4096

#: Identity extents: a short last tile, one whole tile, and many whole tiles.
IDENTITY_SHAPES = (7, PARTITION_MAX, 2048)
IDENTITY_HIDDEN = 512


def _emit(tag: str, **values: object) -> None:
    """One row per reading, pipe-separated, prefix first."""
    fields = "|".join(f"{k}={v}" for k, v in values.items())
    print(f"HC123|{tag}|{fields}", flush=True)


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls, the vendor's own entry."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


# THE REFUSED BODY, copied verbatim from the tree this change was made on.
# TEST-ONLY: nothing shipped imports it. It keeps the LIST COMPREHENSION over the
# loaded stream tiles ON PURPOSE -- that is the construct the compiler counts and
# refuses -- and everything else verbatim, or item 2 compares against something else.
@nki.jit
def _refused_reference_kernel(x, residual, post_layer_mix, comb_res_mix):
    """The mHC combine as it was written before it was made to compile."""
    t_extent, s_extent, h_extent = residual.shape
    pmax = nl.tile_size.pmax
    n_tiles = (t_extent + pmax - 1) // pmax

    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )

    for t in range(n_tiles):
        rows = min(pmax, t_extent - t * pmax)
        off = t * pmax

        x_tile = nl.load(x[off : off + rows, 0:h_extent], dtype=nl.float32)

        streams = [
            nl.load(residual[off : off + rows, i, 0:h_extent], dtype=nl.float32)
            for i in range(s_extent)
        ]

        acc = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)
        term = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)

        for j in range(s_extent):
            post_j = nl.load(
                post_layer_mix[off : off + rows, j, 0:1], dtype=nl.float32
            )
            nisa.tensor_scalar(dst=acc, data=x_tile, op0=nl.multiply, operand0=post_j)

            for i in range(s_extent):
                w_ij = nl.load(
                    comb_res_mix[off : off + rows, i, j : j + 1], dtype=nl.float32
                )
                nisa.tensor_scalar(
                    dst=term, data=streams[i], op0=nl.multiply, operand0=w_ij
                )
                nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)

            nl.store(out[off : off + rows, j, 0:h_extent], value=acc)

    return out


def _inputs(rows: int, hidden: int, seed: int = 123):
    """The four tensors, fp32, deterministic by seed, as the landed suite builds them."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), generator=g, dtype=torch.float32)
    residual = torch.randn((rows, S, hidden), generator=g, dtype=torch.float32)
    post_layer_mix = torch.rand((rows, S, 1), generator=g, dtype=torch.float32)
    comb_res_mix = torch.softmax(
        torch.randn((rows, S, S), generator=g, dtype=torch.float32), dim=-1
    )
    return x, residual, post_layer_mix, comb_res_mix


def _pass_through_weights(rows: int):
    """``comb_res_mix = I_S`` and ``post_layer_mix = 0``, so ``out == residual``.

    Lossless on both sides in fp32 -- a multiply by exactly ``1.0`` and an add of
    exact ``0.0`` -- so kernel and oracle must agree to the bit here.
    """
    ident = torch.eye(S, dtype=torch.float32).expand(rows, S, S).contiguous()
    zero_post = torch.zeros((rows, S, 1), dtype=torch.float32)
    return zero_post, ident


#: THE COMPILING VENUE, and it is a CHILD PROCESS on purpose. ``NKI_SIMULATOR=1``
#: makes the wrapper interpret the body instead of compiling it, and the venue is
#: fixed at import, so the pins below cannot be set inside this process without
#: taking the other two items' venue away.
COMPILE_PINS = {
    "VLLM_NEURON_CPU_MODE": "1",
    "VLLM_NEURON_CPU_COMPILE": "1",
    "VLLM_NEURON_DISABLE_PARALLEL_TRACE": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
    "NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS": "4",
    "VLLM_NEURON_DISABLE_NKI_KERNELS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}
COMPILE_UNSET = (
    "NKI_SIMULATOR",
    "NKI_PRECISE_FP",
    "NEURON_VISIBLE_DEVICES",
    "NEURON_LIBTORCH_CACHE_ROOT",
)

_CHILD_SOURCE = '''
"""Two capture arms under the compiling pins. Prints rows; never raises."""
import os, pathlib, sys, traceback

import torch
import torch._dynamo

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

import vllm_neuron.functional.mhc.hyper_connection as HC

WORKDIR, ROWS, HIDDEN, STREAMS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])


def row(name, value):
    print(f"CHILDROW|{name}|{value}", flush=True)


row("MODULE_FILE", HC.__file__)
row("SIMULATOR_ENV", repr(os.environ.get("NKI_SIMULATOR")))
row("CPU_COMPILE_ENV", repr(os.environ.get("VLLM_NEURON_CPU_COMPILE")))

COPY_SOURCE = {copy_source!r}
exec(COPY_SOURCE, globals())


def options(workdir):
    """The production option string, as this campaign's own capture probe sets it."""
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    optlevel = None
    try:
        import dataclasses

        from vllm.config import VllmConfig

        for f in dataclasses.fields(VllmConfig):
            if f.name == "optimization_level":
                optlevel = getattr(f.default, "value", f.default)
    except Exception as exc:
        row("OPTLEVEL_ERROR", f"{{type(exc).__name__}}: {{exc}}")
    hlo = "--modular-flow-mac-threshold=10"
    if get_platform_target() not in ("trn3", "trn3pre"):
        hlo += " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
    return {{
        "alias_meta_to_neuron": True,
        "compiler_args": [
            "--auto-cast=none",
            "--verbose=35",
            f"-O{{optlevel}}",
            f"--internal-hlo2tensorizer-options={{hlo}}",
            "--internal-backend-options=--enable-verifier=true --enable-nested-dynamic-loop",
        ],
        "compiler_workdir": workdir,
    }}


def stage_of(message):
    """Name the stage. Specialisation precedes compilation, so it is tested first."""
    low = message.lower()
    if "failed to specialize" in low or "unsupported expression" in low:
        return "NKI_SPECIALIZE"
    if "failed to compile nki kernel" in low:
        return "NKI_COMPILE"
    if "failed verification" in low:
        return "MLIR_VERIFY"
    if "not on meta" in low:
        return "BACKEND_META"
    return "UNCLASSIFIED"


def meta(*shape):
    return torch.empty(shape, dtype=torch.float32, device="meta")


def arm(name, kernel, workdir):
    """One capture. A refusal IS the reading, and its stage is the point."""
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        x=meta(ROWS, HIDDEN),
        residual=meta(ROWS, STREAMS, HIDDEN),
        post_layer_mix=meta(ROWS, STREAMS, 1),
        comb_res_mix=meta(ROWS, STREAMS, STREAMS),
    )
    torch._dynamo.reset()
    wrapped = torch.compile(
        lambda **kw: wrap_nki(kernel)(**kw),
        backend="neuron_libtorch_graph_capture",
        fullgraph=True,
        options=options(workdir),
    )
    try:
        wrapped(**kwargs)
        row(f"{{name}}_STAGE", "RETURNED_WITHOUT_CAPTURECOMPLETE")
        row(f"{{name}}_MESSAGE", "")
    except CaptureComplete:
        row(f"{{name}}_STAGE", "CAPTURED")
        row(f"{{name}}_MESSAGE", "")
    except BaseException as exc:
        message = " ".join(str(exc).split())
        row(f"{{name}}_STAGE", stage_of(message))
        row(f"{{name}}_MESSAGE", message[:1600])
        row(f"{{name}}_TYPE", f"{{type(exc).__module__}}.{{type(exc).__name__}}")
        print("CHILDTRACE_BEGIN", flush=True)
        traceback.print_exc()
        print("CHILDTRACE_END", flush=True)


arm("MODULE", HC.hyper_connection_kernel, WORKDIR + "/module")
arm("COPY", globals()["_refused_reference_kernel"], WORKDIR + "/copy")
row("CHILD_DONE", "1")
'''


def _capture_arms(tmp_path) -> dict:
    """Run both capture arms in a child pinned to the venue that refused the old body."""
    script = tmp_path / "capture_arms.py"
    script.write_text(
        _CHILD_SOURCE.format(copy_source=inspect.getsource(_refused_reference_kernel))
    )
    env = {k: v for k, v in os.environ.items() if k not in COMPILE_UNSET}
    env.update(COMPILE_PINS)
    env["PYTHONPATH"] = os.getcwd()
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "cap"),
         str(SPECIALISE_ROWS), str(SPECIALISE_HIDDEN), str(S)],
        cwd=os.getcwd(), env=env, capture_output=True, text=True, timeout=1800,
    )
    rows = {}
    for line in (done.stdout + done.stderr).splitlines():
        if line.startswith("CHILDROW|"):
            _, name, value = line.split("|", 2)
            rows[name] = value
    rows["_rc"] = str(done.returncode)
    rows["_tail"] = " ".join((done.stdout + done.stderr).split())[-1200:]
    return rows


# --------------------------------------------------------------------------- #
# ITEM 1 -- the compiler specialises this body, and still refuses the old one.   #
# --------------------------------------------------------------------------- #
def test_the_kernel_specialises_under_the_capture_regime(tmp_path) -> None:
    """The module's kernel captures at the reported shape; the old body does not.

    The copy's refusal is what makes the first reading one: a venue that reached
    no compiler would report both bodies green, so the copy must fail in the same
    child, with the vendor's own two fragments, on BOTH trees.
    """
    rows = _capture_arms(tmp_path)
    module_stage = rows.get("MODULE_STAGE", "no_row")
    module_message = rows.get("MODULE_MESSAGE", "")
    copy_stage = rows.get("COPY_STAGE", "no_row")
    copy_message = rows.get("COPY_MESSAGE", "")
    print(f"HC123|I1_MODULE_VERBATIM|{SPECIALISE_ROWS}|{SPECIALISE_HIDDEN}|"
          f"{module_message}", flush=True)
    print(f"HC123|I1_COPY_VERBATIM|{SPECIALISE_ROWS}|{SPECIALISE_HIDDEN}|"
          f"{copy_message}", flush=True)
    _emit(
        "I1_CHILD",
        rc=rows["_rc"], done=rows.get("CHILD_DONE", "0"),
        simulator_env=rows.get("SIMULATOR_ENV", "no_row"),
        cpu_compile_env=rows.get("CPU_COMPILE_ENV", "no_row"),
        module_file_is_the_imported_one=int(
            rows.get("MODULE_FILE") == HC_MODULE.__file__
        ),
    )
    _emit(
        "I1_MODULE_KERNEL",
        rows=SPECIALISE_ROWS, streams=S, hidden=SPECIALISE_HIDDEN,
        expect_base=int(EXPECT_BASE), venue="nki_frontend_capture",
        stage=module_stage, raised=int(module_stage != "CAPTURED"),
        names_the_refusal=int(WANT_REFUSAL in module_message),
        names_the_diagnostic=int(WANT_DIAGNOSTIC in module_message),
    )
    _emit(
        "I1_REFUSED_COPY",
        rows=SPECIALISE_ROWS, hidden=SPECIALISE_HIDDEN, stage=copy_stage,
        raised=int(copy_stage != "CAPTURED"),
        names_the_refusal=int(WANT_REFUSAL in copy_message),
        names_the_diagnostic=int(WANT_DIAGNOSTIC in copy_message),
    )
    assert rows.get("CHILD_DONE") == "1", (
        f"the capture child did not finish: rc={rows['_rc']} {rows['_tail']}"
    )
    assert rows.get("SIMULATOR_ENV") == repr(None), (
        f"the child saw NKI_SIMULATOR={rows.get('SIMULATOR_ENV')}, so it would have "
        f"interpreted the body instead of compiling it"
    )
    assert copy_stage == "NKI_SPECIALIZE", (
        f"the replaced body reached {copy_stage}, so this venue asked the compiler "
        f"nothing about specialisation: {copy_message or rows['_tail']}"
    )
    assert WANT_REFUSAL in copy_message, copy_message
    assert WANT_DIAGNOSTIC in copy_message, copy_message
    if EXPECT_BASE:
        assert module_stage == "NKI_SPECIALIZE", (
            f"the base body reached {module_stage}: {module_message}"
        )
        assert WANT_REFUSAL in module_message, module_message
        assert WANT_DIAGNOSTIC in module_message, module_message
    else:
        assert module_stage == "CAPTURED", (
            f"the kernel reached {module_stage}: {module_message or rows['_tail']}"
        )


# --------------------------------------------------------------------------- #
# ITEM 2 -- identical numbers, to the bit, against two independent references.   #
# --------------------------------------------------------------------------- #
def test_the_body_is_bit_identical_to_the_one_it_replaced_and_to_the_oracle() -> None:
    """Equality, not closeness, on the simulator venue at three tile shapes.

    Against the refused body the claim is BIT-IDENTITY: the rewrite changed which
    python forms express the loop, not the numbers nor their order. Against the
    oracle it is the pass-through pattern, because on a random fixture a scalar
    accumulation and an ``einsum`` contraction round differently by construction.
    """
    for rows in IDENTITY_SHAPES:
        x, residual, post_layer_mix, comb_res_mix = _inputs(rows, IDENTITY_HIDDEN)

        reset_dispatch_counters()
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        nki_n, fallback_n = dispatch_counters()
        replaced = wrap_nki(_refused_reference_kernel)(
            x=x,
            residual=residual,
            post_layer_mix=post_layer_mix,
            comb_res_mix=comb_res_mix,
        )

        got32 = got.to(torch.float32)
        replaced32 = replaced.to(torch.float32)
        differing = int((got32 != replaced32).sum())
        _emit(
            "I2_EQUALS_THE_BODY_IT_REPLACED",
            rows=rows, hidden=IDENTITY_HIDDEN, entries=got32.numel(),
            nki_dispatch=nki_n, torch_fallback=fallback_n,
            differing_entries=differing, bit_exact=torch.equal(got32, replaced32),
            replaced_absmax=f"{float(replaced32.abs().max()):.6e}",
        )
        assert (nki_n, fallback_n) == (1, 0), (rows, nki_n, fallback_n)
        assert float(replaced32.abs().max()) > 0.0, f"rows={rows}: reference all zero"
        assert torch.equal(got32, replaced32), (
            f"rows={rows}: {differing} entries differ from the body this replaced"
        )

        zero_post, ident = _pass_through_weights(rows)
        reset_dispatch_counters()
        served = hyper_connection_combine(x, residual, zero_post, ident)
        oracle = hyper_connection_torch_oracle(x, residual, zero_post, ident)
        served32 = served.to(torch.float32)
        _emit(
            "I2_EQUALS_THE_ORACLE_EXACTLY",
            rows=rows, hidden=IDENTITY_HIDDEN,
            differing_from_oracle=int((served32 != oracle).sum()),
            differing_from_residual=int((served32 != residual).sum()),
            bit_exact_oracle=torch.equal(served32, oracle),
            bit_exact_residual=torch.equal(served32, residual),
            residual_absmax=f"{float(residual.abs().max()):.6e}",
            x_absmax=f"{float(x.abs().max()):.6e}",
        )
        assert float(residual.abs().max()) > 0.0, "the residual fixture is all zero"
        assert float(x.abs().max()) > 0.0, (
            "x is all zero, so an ignored post term could not have shown here"
        )
        assert torch.equal(served32, oracle), f"rows={rows}: the oracle disagrees"
        assert torch.equal(served32, residual), f"rows={rows}: not the residual"


# --------------------------------------------------------------------------- #
# ITEM 3 -- one NKI dispatch per call, and no torch fallback.                   #
# --------------------------------------------------------------------------- #
def test_the_route_is_one_nki_dispatch_per_call() -> None:
    """Two calls, read after each: ``1`` then ``2``, with the fallback at zero.

    A body walking the token axis from the HOST would read one dispatch per tile;
    one answering in torch would read a fallback.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(2048, 8)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        after_first = dispatch_counters()
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        after_second = dispatch_counters()
    tiles = -(-2048 // PARTITION_MAX)
    _emit(
        "I3_ROUTE",
        rows=2048, tiles=tiles, simulate_kernel_calls=sim.calls,
        after_first_call="/".join(str(v) for v in after_first),
        after_second_call="/".join(str(v) for v in after_second),
        a_host_tile_loop_would_read=tiles * 2,
    )
    assert after_first == (1, 0), after_first
    assert after_second == (2, 0), after_second
    assert sim.calls == 2, (
        f"nki.simulator.simulate_kernel ran {sim.calls} times for two calls, "
        f"declared 2; a host loop over tiles would read {tiles * 2}"
    )
