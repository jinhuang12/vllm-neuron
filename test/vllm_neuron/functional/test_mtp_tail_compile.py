# SPDX-License-Identifier: Apache-2.0
"""Both MTP tail kernels pass the NKI front end and neuronx-cc's backend, and BIRSim
agrees with the torch expressions they replace.

The simulator runs a kernel body as plain Python, so it sees neither a call form the
front end refuses nor a rule the backend verifier enforces (a bf16 PSUM slot that does
not start 4-byte aligned; a DMA transpose whose output row stride is not a multiple of
32 bytes). Two child processes, each pinning the platform target and opening no device
node (asserted: no ``/dev/neuron*`` descriptor is held at the end), so the test runs
the same with and without a device visible and never skips:

* front end: ``wrap_nki`` under ``FakeTensorMode`` for every served and tiny geometry,
  B in {1, 4, 64, 130} (130 spans two partition tiles), one and two programs; a body
  that reads an undefined name is the control that bodies are parsed;
* backend: ``nki``'s standalone path through neuronx-cc (the device compiler's own BIR
  verifier). LNC1 compiles, runs on BIRSim (the backend's instruction-level simulator,
  which models the engines' arithmetic) and checks the outputs against float64 within
  the derived bounds of ``test_mtp_tail_bounds.py``; LNC2 compiles only (BIRSim runs
  one core). Served geometry (H=4096, the TP=64 shards: 64 ``eh_proj`` rows, 2420 head
  rows) at B in {1, 4, 64}; the tiny geometry (H=512, whole ``eh_proj``, 64 head rows)
  at B in {1, 130}.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import nki
import nki.language as nl

ROW = "mtp_tail_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG")
_PIN = {"NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "PYTHONDONTWRITEBYTECODE": "1"}
#: ``(hidden, vocab, eh_proj rows, head rows)``: the served TP=64 rank and the tiny fixture.
SERVED = (4096, 64, 64, 154880 // 64)
TINY = (512, 64, 512, 64)
FRONTEND_BATCHES = (1, 4, 64, 130)
BACKEND_CASES = ((SERVED, 1), (SERVED, 4), (SERVED, 64), (TINY, 1), (TINY, 130))
EPS = 1e-5


@nki.jit
def body_that_reads_an_undefined_name(x_hbm):
    """Refused by a front end that parses bodies; accepted only by one that does not."""
    out = nl.ndarray(x_hbm.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    held = nl.ndarray((1, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
    nl.store(out[0:1, :], value=held * no_such_name_anywhere)  # noqa: F821
    return out


def _neuron_fds() -> int:
    fds = 0
    for handle in os.listdir("/proc/self/fd"):
        try:
            fds += "/dev/neuron" in os.readlink(f"/proc/self/fd/{handle}")
        except OSError:
            pass
    return fds


def _frontend_child() -> None:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.functional.mtp import tail_in, tail_out

    bf, i32 = torch.bfloat16, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def grid(entry, programs):
        call = wrap_nki(entry)
        return call[2] if programs == 2 else call

    built = []
    for hidden, vocab, rows, head_rows in (SERVED, TINY):
        for batch in FRONTEND_BATCHES:
            for programs in (1, 2):
                tag = f"h{hidden}_b{batch}_g{programs}"
                built.append((f"tail_in_{tag}", lambda h=hidden, v=vocab, r=rows, b=batch, g=programs: grid(
                    tail_in.mtp_tail_in_kernel, g)(
                    token_ids=fake((b,), i32), table=fake((v, h), bf), positions=fake((b,), i32),
                    previous=fake((b, h), bf), enorm=fake((h,), bf), hnorm=fake((h,), bf),
                    eh_proj_rows=fake((r, 2 * h), bf), EPS=EPS)))
                built.append((f"tail_out_{tag}", lambda h=hidden, s=head_rows, b=batch, g=programs: grid(
                    tail_out.mtp_tail_out_kernel, g)(
                    attended=fake((b, h), bf), ffn=fake((b, h), bf), gain=fake((h,), bf),
                    head_rows=fake((s, h), bf), EPS=EPS)))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|" + json.dumps(dict(tree=tail_in.__file__)), flush=True)
    for name, call in built:
        message = ""
        try:
            with FakeTensorMode():
                call()
        except BaseException as refusal:  # a refusal is a result here, not a test error
            message = " ".join(str(refusal).split())[:2000]
        print(ROW + "|" + json.dumps(dict(entry=name, refused=bool(message), neuron_fds=_neuron_fds(),
                                          diagnostic=message or "none")), flush=True)


def _backend_child() -> None:
    import ml_dtypes
    import numpy as np
    import torch

    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import StandaloneKernel

    from test.vllm_neuron.functional.test_mtp_tail_bounds import (
        assert_normed_within_one_flip, gemv_bound, gemv_bound_rounded_rows, rms64)
    from vllm_neuron.functional.mtp import tail_in, tail_out

    bf = ml_dtypes.bfloat16

    def rnd(shape, seed, scale=1.0, dtype=bf):
        return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(dtype)

    def t(a, dtype=torch.bfloat16):
        return torch.from_numpy(np.asarray(a).astype(np.float32)).to(dtype)

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

    def build(name, geometry, batch):
        hidden, vocab, rows, head_rows = geometry
        if name == "tail_in":
            ids = np.random.default_rng(10).integers(0, vocab, batch).astype(np.int32)
            positions = np.random.default_rng(11).integers(1, 4096, batch).astype(np.int32)
            positions[0] = 0
            args = dict(token_ids=ids, table=rnd((vocab, hidden), 12), positions=positions,
                        previous=rnd((batch, hidden), 13),
                        enorm=(1.0 + rnd((hidden,), 14, 0.05, np.float32)).astype(bf),
                        hnorm=(1.0 + rnd((hidden,), 15, 0.05, np.float32)).astype(bf),
                        eh_proj_rows=rnd((rows, 2 * hidden), 16, (2 * hidden) ** -0.5), EPS=EPS)
            return tail_in.mtp_tail_in_kernel, args
        args = dict(attended=rnd((batch, hidden), 20), ffn=rnd((batch, hidden), 21),
                    gain=(1.0 + rnd((hidden,), 22, 0.05, np.float32)).astype(bf),
                    head_rows=rnd((head_rows, hidden), 23, hidden ** -0.5), EPS=EPS)
        return tail_out.mtp_tail_out_kernel, args

    def check(name, geometry, args, out):
        """``(ok, reading)`` of the BIRSim outputs against float64 within the derived bounds."""
        hidden = geometry[0]
        if name == "tail_in":
            got = t(out[0] if isinstance(out, tuple) else out)
            table, rows_w = t(args["table"]), t(args["eh_proj_rows"])
            embeds = table[torch.from_numpy(args["token_ids"]).to(torch.int64)].double()
            masked = torch.from_numpy(args["positions"]).reshape(-1, 1) == 0
            embeds = torch.where(masked, torch.zeros_like(embeds), embeds)
            joined = torch.cat([rms64(embeds, t(args["enorm"]), EPS),
                                rms64(t(args["previous"]), t(args["hnorm"]), EPS)], dim=-1)
            exact = joined.to(torch.bfloat16).double() @ rows_w.double().t()
            bound = gemv_bound(got, joined, rows_w, hidden)
            diff = (got.double() - exact).abs()
            ok = bool((diff <= bound).all())
            return ok, f"within_bound={ok} max_excess={float((diff - bound).max()):.2e}"
        hidden_out, pair = t(out[0]), t(out[1], torch.float32)
        mixed = (t(args["attended"]).double() + t(args["ffn"]).double()).to(torch.bfloat16)
        exact_hidden = rms64(mixed, t(args["gain"]), EPS)
        try:
            assert_normed_within_one_flip(hidden_out, exact_hidden, hidden, "hidden")
            hidden_ok = True
        except AssertionError:
            hidden_ok = False
        head = t(args["head_rows"])
        exact = hidden_out.double() @ head.double().t()
        bound = gemv_bound_rounded_rows(exact, hidden_out, head)
        idx = pair[:, 1].to(torch.int64).reshape(-1, 1)
        at, bound_at = exact.gather(1, idx).reshape(-1), bound.gather(1, idx).reshape(-1)
        max_ok = bool(((pair[:, 0].double() - at).abs() <= bound_at).all())
        best, best_idx = exact.max(dim=-1)
        bound_best = bound.gather(1, best_idx.reshape(-1, 1)).reshape(-1)
        near_ok = bool((at >= best - bound_best - bound_at).all())
        top2 = exact.topk(2, dim=-1).values
        decisive = (top2[:, 0] - top2[:, 1]) > (bound_best + bound.gather(
            1, exact.topk(2, dim=-1).indices[:, 1:2]).reshape(-1))
        arg_ok = bool(torch.equal(idx.reshape(-1)[decisive], best_idx[decisive]))
        ok = hidden_ok and max_ok and near_ok and arg_ok
        return ok, (f"hidden_within_flip={hidden_ok} max_at_index={max_ok} near_max={near_ok} "
                    f"decisive_argmax={arg_ok} decisive_rows={int(decisive.sum())}/{len(decisive)}")

    for geometry, batch in BACKEND_CASES:
        for name in ("tail_in", "tail_out"):
            for lnc in (1, 2):
                kernel, args = build(name, geometry, batch)
                row = dict(kernel=name, H=geometry[0], B=batch, lnc=lnc)
                try:
                    out = launch(kernel, lnc, args)
                    row["compiled"] = True
                    if lnc == 1:
                        row["agrees"], row["reading"] = check(name, geometry, args, out)
                except Exception as refusal:  # the backend's refusal is the row's reading
                    row["compiled"] = False
                    row["reading"] = " ".join(str(refusal).split())[:1500]
                print(ROW + "|" + json.dumps(row), flush=True)
    print(ROW + "|" + json.dumps(dict(neuron_fds=_neuron_fds())), flush=True)


def _run_child(mode: str, scratch: pathlib.Path, timeout: int) -> list[dict]:
    # neuronx-cc writes its intermediates into the working directory, so the child runs
    # in a scratch one, not the tree; the venv's bin goes first on PATH for neuronx-cc.
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    venv_bin = str(pathlib.Path(sys.executable).parent)
    environment.update(_PIN, PYTHONPATH=str(_ROOT),
                       PATH=venv_bin + os.pathsep + environment.get("PATH", ""))
    done = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), mode],
                          cwd=scratch, env=environment, capture_output=True, text=True,
                          timeout=timeout, check=False)
    rows = [json.loads(line.split("|", 1)[1]) for line in done.stdout.splitlines()
            if line.startswith(ROW + "|")]
    for row in rows:
        print(row)
    assert done.returncode == 0, done.stderr[-3000:]
    return rows


def test_the_front_end_accepts_both_kernels_at_every_geometry(tmp_path) -> None:
    rows = _run_child("frontend", tmp_path, 1500)
    assert rows and rows[0].get("tree", "").startswith(str(_ROOT) + "/"), rows[:1]
    entries = rows[1:]
    assert {row["neuron_fds"] for row in entries} == {0}, entries
    control = [row for row in entries if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] is True, (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    kernels = [row for row in entries if row["entry"] != "undefined_name_body"]
    assert len(kernels) == 2 * 2 * len(FRONTEND_BATCHES) * 2
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in kernels if row["refused"]]
    assert refused == [], refused


def test_both_kernels_pass_the_backend_and_birsim_agrees(tmp_path) -> None:
    rows = _run_child("backend", tmp_path, 3000)
    assert rows and rows[-1] == {"neuron_fds": 0}, rows[-1:]
    cases = rows[:-1]
    assert len(cases) == len(BACKEND_CASES) * 2 * 2
    refused = [row for row in cases if not row["compiled"]]
    assert refused == [], refused
    disagree = [row for row in cases if row["lnc"] == 1 and not row["agrees"]]
    assert disagree == [], disagree


if __name__ == "__main__" and sys.argv[1:] == ["frontend"]:
    _frontend_child()
if __name__ == "__main__" and sys.argv[1:] == ["backend"]:
    _backend_child()
