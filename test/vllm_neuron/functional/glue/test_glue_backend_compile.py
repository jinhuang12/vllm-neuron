# SPDX-License-Identifier: Apache-2.0
"""Every glue kernel passes neuronx-cc's backend verifier, and BIRSim agrees with torch.

The front-end compile test (``test_glue_nki_frontend_compile.py``) stops at the NKI front
end, so it cannot see a rule the backend verifier enforces: a bf16 PSUM slot that does
not start 4-byte aligned, a DMA transpose whose output row stride is not a multiple of
32 bytes, a reshaped view whose partition step is 0 at one token. Each of those passed
the simulator and the front end and stopped the device compile. This test compiles each
kernel through neuronx-cc (``nki``'s standalone path, which runs the same BIR verifier)
and runs it on BIRSim, the backend's instruction-level simulator, in a child process
that opens no device node:

* LNC1: compile, BIRSim, and the outputs against the torch expression they replace, at
  the tolerances the simulator tests state (BIRSim models the engines' arithmetic, so
  this also checks the activation and reciprocal engines' precision).
* LNC2: compile only (BIRSim runs one core).

Served dtypes and geometry: one rank's shapes at B in {1, 4, 64}, both transpose modes
of the two KDA kernels.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROW = "glue_backend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG")
_PIN = {"NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "PYTHONDONTWRITEBYTECODE": "1"}
BATCHES = (1, 4, 64)
#: (kernel, DMA transpose) cases; ``None`` where the kernel has no transpose switch.
CASES = (("mhc_pre", None), ("kda_projections", True), ("kda_projections", False),
         ("kda_output", True), ("kda_output", False), ("combine_bf16", None))
#: Tolerances, as the simulator tests state them (see each test module's docstring).
POST_ATOL = COMB_ATOL = 2e-5
PROJ_RTOL = 2e-6
ATTN_RTOL = 1e-5
#: The absolute term of a bf16 output's bound, ``CANCEL_RTOL * max|ref|``.
CANCEL_RTOL = 1e-5


def _child() -> None:
    import numpy as np
    import torch
    import ml_dtypes
    from types import SimpleNamespace

    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import StandaloneKernel

    from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
    from vllm_neuron.functional.mhc import hyper_connection

    bf = ml_dtypes.bfloat16
    hidden, streams, mix, width, rank = 4096, 4, 24, 128, 128

    def rnd(shape, dtype=bf, scale=1.0, seed=0):
        return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(dtype)

    def t(a):
        return torch.from_numpy(np.asarray(a).astype(np.float32))

    def step(*values):
        mag = torch.stack([v.abs() for v in values]).amax(0).clamp_min(2.0 ** -126)
        return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)

    def noop(compiled, inputs, outputs):
        return None

    def launch(kernel, lnc, args):
        if lnc > 1:
            k = kernel[lnc]._to_subclass(StandaloneKernel, _frontend_cls=resolve_frontend_cls(),
                                         _executor=noop)
        else:
            k = kernel._to_subclass(StandaloneKernel, _frontend_cls=resolve_frontend_cls(),
                                    _enable_simulation=True)
        return k(**args)

    def build(name, batch, dma):
        if name == "mhc_pre":
            args = dict(residual=rnd((batch, streams, hidden), scale=0.5),
                        fn=rnd((mix, streams * hidden), scale=(streams * hidden) ** -0.5, seed=1),
                        hc_scale=np.array([0.1, 0.2, 0.3], np.float32),
                        hc_base=rnd((mix,), np.float32, 0.5, 2))
            return mhc_pre.mhc_pre_kernel, args
        if name == "kda_projections":
            shapes = dict(q_w=(width, hidden), k_w=(width, hidden), v_w=(width, hidden),
                          b_w=(1, hidden), f_a_w=(rank, hidden), f_b_w=(width, rank),
                          g_a_w=(rank, hidden), g_b_w=(width, rank))
            args = {key: rnd(shape, scale=shape[1] ** -0.5, seed=i + 1)
                    for i, (key, shape) in enumerate(shapes.items())}
            args.update(x=rnd((batch, hidden)), DMA_TRANSPOSE=dma)
            return kda_projections.kda_projections_kernel, args
        if name == "kda_output":
            args = dict(core=rnd((batch, width), np.float32, 0.2),
                        out_gate=rnd((batch, width), np.float32, 2.0, 1),
                        o_norm=rnd((width,), seed=2), o_proj=rnd((hidden, width), scale=width ** -0.5, seed=3),
                        HEADS=1, EPS=1e-6, DMA_TRANSPOSE=dma)
            return kda_output.kda_output_kernel, args
        comb = np.abs(rnd((batch, streams, streams), np.float32, 0.3, 3))
        args = dict(x=rnd((batch, hidden), seed=1), residual=rnd((batch, streams, hidden), scale=0.5, seed=2),
                    post_layer_mix=np.abs(rnd((batch, streams, 1), np.float32, 1.0, 4)),
                    comb_res_mix=(comb / comb.sum(-1, keepdims=True)).astype(np.float32))
        return hyper_connection.hyper_connection_kernel, args

    def check(name, args, out):
        """``(ok, reading)`` of the BIRSim outputs against the torch expressions."""
        outs = [t(o) for o in (out if isinstance(out, tuple) else (out,))]
        if name == "mhc_pre":
            r, fn = t(args["residual"]).double(), t(args["fn"]).double()
            sc, base = t(args["hc_scale"]).double(), t(args["hc_base"]).double()
            batch = r.shape[0]
            flat = r.reshape(batch, -1)
            mixes = flat @ fn.t() * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            pre = torch.sigmoid(mixes[:, :streams] * sc[0] + base[:streams]) + 1e-6
            post = torch.sigmoid(mixes[:, streams:2 * streams] * sc[1]
                                 + base[streams:2 * streams]) * 2.0
            logits = mixes[:, 2 * streams:].reshape(batch, streams, streams) * sc[2] \
                + base[2 * streams:].reshape(1, streams, streams)
            comb = torch.softmax(logits, -1) + 1e-6
            x = ((pre.unsqueeze(-1) * r).sum(1)).float()
            d_post = float((outs[0].double().reshape(batch, streams) - post).abs().max())
            d_comb = float((outs[1].double() - comb).abs().max())
            bound = step(x, outs[2]) + CANCEL_RTOL * float(x.abs().max())
            x_ok = bool(((outs[2] - x).abs() <= bound).all())
            return (d_post <= POST_ATOL and d_comb <= COMB_ATOL and x_ok,
                    f"post={d_post:.1e} comb={d_comb:.1e} x_within_step={x_ok}")
        if name == "kda_projections":
            attn = SimpleNamespace(**{f"{key[:-2]}_proj_weight": t(v).to(torch.bfloat16)
                                      for key, v in args.items() if key.endswith("_w")})
            want = kda_projections.kda_projections_torch(t(args["x"]).to(torch.bfloat16), attn)
            rel = [float((g - w).abs().max() / w.abs().max()) for g, w in zip(outs, want)]
            return max(rel) <= PROJ_RTOL, "maxrel=" + ",".join(f"{v:.1e}" for v in rel)
        if name == "kda_output":
            attn = SimpleNamespace(o_proj_weight=t(args["o_proj"]).to(torch.bfloat16),
                                   o_norm_weight=t(args["o_norm"]).to(torch.bfloat16),
                                   num_kv_heads_per_rank=1, head_dim=width, rms_norm_eps=1e-6)
            want = kda_output.kda_gated_projection_torch(t(args["core"]), t(args["out_gate"]), attn)
            rel = float((outs[0] - want).abs().max() / want.abs().max())
            return rel <= ATTN_RTOL, f"maxrel={rel:.1e}"
        want = hyper_connection.hyper_connection_torch_oracle(
            t(args["x"]), t(args["residual"]), t(args["post_layer_mix"]), t(args["comb_res_mix"]))
        # One bf16 step, plus the cancellation term: a mixed stream that cancels carries
        # the fp32 error of its terms, not of its own magnitude.
        bound = step(want, outs[0]) + CANCEL_RTOL * float(want.abs().max())
        d = (outs[0] - want).abs()
        ok = bool((d <= bound).all())
        return ok, f"within_bound={ok} max|d|/max|ref|={float(d.max() / want.abs().max()):.1e}"

    for name, dma in CASES:
        for batch in BATCHES:
            for lnc in (1, 2):
                kernel, args = build(name, batch, dma)
                row = dict(kernel=name, dma=dma, B=batch, lnc=lnc)
                try:
                    out = launch(kernel, lnc, args)
                    row["compiled"] = True
                    if lnc == 1:
                        row["agrees"], row["reading"] = check(name, args, out)
                except Exception as refusal:  # the backend's refusal is the row's reading
                    row["compiled"] = False
                    row["reading"] = " ".join(str(refusal).split())[:1500]
                print(ROW + "|" + json.dumps(row), flush=True)
    fds = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            fds += "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}")
        except OSError:
            pass
    print(ROW + "|" + json.dumps(dict(neuron_fds=fds)), flush=True)


def test_every_glue_kernel_passes_the_backend_and_birsim(tmp_path):
    # neuronx-cc writes its intermediates (``<hash>/*.colz``, ``global_metric_store.json``)
    # into the working directory, so the child runs in a scratch one, not the tree.
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    venv_bin = str(pathlib.Path(sys.executable).parent)
    environment.update(_PIN, PYTHONPATH=str(_ROOT),
                       PATH=venv_bin + os.pathsep + environment.get("PATH", ""))
    done = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "child"],
                          cwd=tmp_path, env=environment, capture_output=True, text=True,
                          timeout=1500, check=False)
    rows = [json.loads(line.split("|", 1)[1]) for line in done.stdout.splitlines()
            if line.startswith(ROW + "|")]
    for row in rows:
        print(row)
    assert done.returncode == 0, done.stderr[-3000:]
    assert rows and rows[-1] == {"neuron_fds": 0}, rows[-1:]
    cases = rows[:-1]
    assert len(cases) == len(CASES) * len(BATCHES) * 2
    refused = [row for row in cases if not row["compiled"]]
    assert refused == [], refused
    disagree = [row for row in cases if row["lnc"] == 1 and not row["agrees"]]
    assert disagree == [], disagree


if __name__ == "__main__":
    _child()
