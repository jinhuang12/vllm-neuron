# SPDX-License-Identifier: Apache-2.0
"""The DCP LSE merge kernel against an independent float64 reference of llama3's merge.

Under decode context parallelism every rank of a CP group returns, per (head, row), a
partial attention output normalised over the columns it owns and the log-sum-exp of those
columns (``mla_sparse.mla_sparse_attention_partial``). The merge combines the CP partials
of one head into the attention over every column, in the latent rank, and rounds once to
bf16. The reference below is written the way llama3's merge is written in
``attention_decode.py`` (``global_lse = logsumexp(all_lse)``, ``correction = exp(lse -
global_lse)``, ``sum``), in float64, so a shared mistake would have to be made twice in two
formulations.

The tolerance is derived, not measured: see :func:`merge_bound`. The simulator never runs
the backend verifier, so the kernel is also built to a NEFF by neuronx-cc on the CPU
(:func:`compile_in_a_child`).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pytest
import torch

from vllm_neuron.functional.attention import dcp_merge as DM

#: fp32 unit roundoff.
U32 = 2.0 ** -24
#: bf16 unit roundoff: 8 significand bits, round to nearest even.
U16 = 2.0 ** -8
#: Error of the simulator's ``exp`` (numpy float32) on a normal result, in units of U32
#: relative to the result. numpy documents its float32 exp at most 2.52 ulp (one ulp of v
#: is at most 2 * U32 * |v|), so 5.04 U32; measured 3.38 U32 over 4M samples in [-20, 0]
#: (``reports/dcp_item4-logs/exp_log_ulp.txt``). A result below :data:`TINY` is covered by
#: TINY instead.
EXP_U = 6.0
#: The smallest normal fp32. An exp whose result is below it is subnormal in the simulator
#: (relative error up to 1: measured on [-104, 20]) or flushed to zero on a device that
#: flushes; either way its absolute error is below TINY.
TINY = 2.0 ** -126
#: Latent rank of this checkpoint (kv_lora_rank).
LATENT = 512

#: The prefix of the row a compile child prints, and the environment it must not inherit.
COMPILE_ROW = "dcp_compile"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_LIBTORCH_CPU_MODE",
         "NEURON_RT_VISIBLE_CORES")
#: The merge's compile cases: ``(CP, H, R)`` at the served R = 2048 rows and L = 512.
MERGE_COMPILES = ((2, 1, 2048), (8, 1, 2048), (4, 2, 2048))
#: How a compile child starts: the plugin first, so its patches are in place when the
#: backends load (a test file may import ``libtorch_neuronx_lite`` at its top), then the test
#: file as ``__main__`` with ``sys.argv[1:]`` = ``["child", *case]``.
_CHILD_BOOT = ("import sys, runpy, vllm_neuron; "
               "runpy.run_path(sys.argv.pop(1), run_name='__main__')")


def gamma(n: int) -> float:
    """The classical bound on n fp32 roundings in a row: n u / (1 - n u)."""
    return n * U32 / (1.0 - n * U32)


def llama3_merge_reference(partials: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """float64 merge of ``[CP, H, R, L]`` partials and ``[CP, H, R]`` lses into ``[R, H, L]``."""
    p = partials.double()
    l = lse.double()
    global_lse = torch.logsumexp(l, dim=0)                  # [H, R]
    correction = torch.exp(l - global_lse)                  # [CP, H, R]
    merged = (p * correction.unsqueeze(-1)).sum(dim=0)      # [H, R, L]
    return merged.permute(1, 0, 2).contiguous()             # [R, H, L]


def merge_bound(partials: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Per-element bound on |kernel - reference|, derived from the kernel's arithmetic.

    The kernel computes, per (head, row): ``m = max_c lse_c`` (exact), ``w_c = exp(lse_c -
    m)`` (one fp32 subtraction, ``A u`` with ``A = |lse_c - m|``, and the exp, ``EXP_U u``),
    ``s = sum_c w_c`` (``gamma(CP - 1)``), ``1 / s`` and ``w_c / s`` (one rounding each),
    then ``y = sum_c (w_c / s) * p_c`` with one rounding per product and per add
    (``gamma(CP)``) and the cast of ``y`` to bf16 (``U16 |y|``). A weight that underflows
    is off by less than TINY absolute, and ``s >= 1``. To first order, with ``a_c`` the
    exact correction and ``S = sum_c a_c |p_c|``:

        |y - y*| <= eps * S + TINY * sum_c |p_c|,
        eps = 2 (A_max + 1 + EXP_U) u + gamma(CP - 1) + 2 u + gamma(CP)
        |bf16(y) - y*| <= U16 |y*| + (1 + U16) (eps S + TINY sum_c |p_c|)

    ``A_max`` is taken over the ranks that contribute (an empty rank's weight is exactly 0
    in both computations). Second-order terms are below 1e-10 relative and are not carried.
    The bf16 term is tight (an exact value just above a power of two rounds by almost
    ``U16 |y*|``), so this bound alone hides the float32 part: see
    :func:`assert_rounds_to_nearest_away_from_ties`.
    """
    exact = llama3_merge_reference(partials, lse)
    return U16 * exact.abs() + (1.0 + U16) * fp32_error_bound(partials, lse)


def fp32_error_bound(partials: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """``eps * S + TINY * sum_c |p_c|`` of :func:`merge_bound`: the bound before the bf16
    rounding, ``[R, H, L]``."""
    cp = int(partials.shape[0])
    l = lse.double()
    live = l > DM.EMPTY_LSE / 2
    m = l.max(dim=0).values
    spread = torch.where(live, (l - m).abs(), torch.zeros_like(l)).max(dim=0).values  # [H, R]
    eps = 2.0 * (spread + 1.0 + EXP_U) * U32 + gamma(max(cp - 1, 0)) + 2.0 * U32 + gamma(cp)
    correction = torch.exp(l - torch.logsumexp(l, dim=0))                      # [CP, H, R]
    s_abs = (partials.double().abs() * correction.unsqueeze(-1)).sum(dim=0)    # [H, R, L]
    floor = TINY * partials.double().abs().sum(dim=0)                           # [H, R, L]
    return (eps.unsqueeze(-1) * s_abs + floor).permute(1, 0, 2).contiguous()   # [R, H, L]


def make_partials(cp: int, heads: int, rows: int, latent: int = LATENT, seed: int = 7,
                  empty_fraction: float = 0.0):
    """CP partials and lses of the magnitudes the attention returns.

    A partial is a convex combination of latent rows, so its entries are of the cache's
    magnitude; an lse is ``scale * max score + ln(columns)``, a few units to a few tens.
    ``empty_fraction`` of the (rank, head, row) slots are empty: partial 0, lse EMPTY_LSE.
    """
    gen = torch.Generator().manual_seed(seed)
    partials = torch.randn(cp, heads, rows, latent, generator=gen) * 0.05
    lse = torch.rand(cp, heads, rows, generator=gen) * 20.0 - 5.0
    if empty_fraction > 0.0:
        empty = torch.rand(cp, heads, rows, generator=gen) < empty_fraction
        partials[empty] = 0.0
        lse[empty] = DM.EMPTY_LSE
    return partials, lse


def distance_to_a_bf16_tie(x: torch.Tensor) -> torch.Tensor:
    """Distance of each float64 ``x`` to the nearest bf16 rounding tie (a midpoint).

    In the binade ``[2**(e-1), 2**e)`` bf16 values are ``2**(e-8)`` apart, so the ties sit
    at odd multiples of ``2**(e-9)``; the tie just below ``2**(e-1)`` is ``2**(e-10)`` under
    it, because the spacing halves there. Zero has no binade and is returned as 0.
    """
    a = x.abs()
    _mantissa, e = torch.frexp(a)                        # a = mantissa * 2**e, mantissa in [0.5, 1)
    spacing = torch.ldexp(torch.ones_like(a), e - 8)
    t = a / spacing                                      # in [128, 256)
    inside = (t - t.floor() - 0.5).abs() * spacing
    below = (a - torch.ldexp(torch.ones_like(a), e - 1)) + spacing / 4
    return torch.where(a > 0, torch.minimum(inside, below), torch.zeros_like(a))


def assert_rounds_to_nearest_away_from_ties(got: torch.Tensor, partials: torch.Tensor,
                                            lse: torch.Tensor, covered: float = 0.99) -> None:
    """``got == RNE(exact)`` wherever ``exact`` is farther than ``eps * S`` from a bf16 tie.

    The kernel's float32 result ``y`` is within ``eps * S`` of the exact merge ``y*``. When no
    tie lies in that interval, ``y`` and ``y*`` round to the same bf16, so the kernel must
    match ``y*`` rounded to nearest even exactly. (``y*`` is rounded through float32 first;
    that moves it by at most ``u |y*| < eps * S``, so it crosses no tie either.) This is the
    fp32 part of :func:`merge_bound` checked on its own, where the bf16 half-ulp term would
    otherwise hide it. Where ``S = 0`` (a row empty on every rank) the bound is 0 and the
    kernel must return exactly 0, so those elements are checked too. ``covered`` is the
    share of elements that must be checked.
    """
    exact = llama3_merge_reference(partials, lse)
    bound = fp32_error_bound(partials, lse)
    clear = (distance_to_a_bf16_tie(exact) > bound) | (bound == 0)
    share = float(clear.double().mean())
    assert share >= covered, f"only {share:.5f} of the elements are clear of a tie"
    rne = exact.float().to(torch.bfloat16)
    wrong = (got != rne) & clear
    assert not bool(wrong.any()), f"{int(wrong.sum())} elements clear of a tie differ from RNE"


def assert_within_bound(got: torch.Tensor, partials: torch.Tensor, lse: torch.Tensor) -> None:
    want = llama3_merge_reference(partials, lse)
    bound = merge_bound(partials, lse)
    err = (got.double() - want).abs()
    worst = float((err / bound.clamp_min(1e-300)).max())
    assert torch.isfinite(got.float()).all()
    assert bool((err <= bound).all()), f"max |err| / bound = {worst}"


@pytest.mark.parametrize("cp", (1, 2, 4, 8))
@pytest.mark.parametrize("heads,rows", ((1, 1), (1, 64), (1, 256), (2, 200)))
def test_merge_matches_the_llama3_reference_within_the_derived_bound(cp, heads, rows):
    partials, lse = make_partials(cp, heads, rows, seed=100 * cp + rows)
    got = DM.dcp_lse_merge(partials, lse)
    assert got.dtype == torch.bfloat16
    assert tuple(got.shape) == (rows, heads, LATENT)
    assert_within_bound(got, partials, lse)


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_the_merge_is_the_exact_value_rounded_to_nearest_away_from_ties(cp):
    """The float32 error of the merge, apart from its one bf16 rounding, is inside eps * S."""
    partials, lse = make_partials(cp, 2, 256, seed=300 + cp, empty_fraction=0.2)
    got = DM.dcp_lse_merge(partials, lse)
    assert_rounds_to_nearest_away_from_ties(got, partials, lse)


def test_one_rank_is_the_partial_rounded_once():
    """CP = 1: the weight is exp(0) / 1 = 1 exactly, so the merge is a single bf16 rounding."""
    partials, lse = make_partials(1, 2, 64, seed=3)
    got = DM.dcp_lse_merge(partials, lse)
    torch.testing.assert_close(got, partials[0].permute(1, 0, 2).to(torch.bfloat16),
                               rtol=0.0, atol=0.0)


def test_a_row_empty_on_every_rank_merges_to_exact_zero_without_nan():
    """Every lse EMPTY_LSE: the max shift makes each weight exp(0) = 1, so 0 / CP, not NaN."""
    partials, lse = make_partials(4, 1, 128, seed=11)
    partials[:, :, 5:9] = 0.0
    lse[:, :, 5:9] = DM.EMPTY_LSE
    got = DM.dcp_lse_merge(partials, lse)
    assert not torch.isnan(got.float()).any()
    assert torch.count_nonzero(got[5:9]) == 0
    assert_within_bound(got, partials, lse)


@pytest.mark.parametrize("cp", (2, 8))
def test_empty_ranks_drop_out_of_the_merge(cp):
    """A rank that owns no column of a row has weight exactly 0, wherever it sits."""
    partials, lse = make_partials(cp, 2, 128, seed=21, empty_fraction=0.4)
    # One row where only the last rank holds anything: the merge is that rank's partial.
    partials[:-1, :, 0] = 0.0
    lse[:-1, :, 0] = DM.EMPTY_LSE
    got = DM.dcp_lse_merge(partials, lse)
    assert_within_bound(got, partials, lse)
    torch.testing.assert_close(got[0], partials[-1, :, 0].to(torch.bfloat16), rtol=0.0, atol=0.0)


def test_bfloat16_partials_merge_as_their_float32_widening():
    """The kernel reads a bf16 partial as its exact fp32 value; the arithmetic is unchanged."""
    partials, lse = make_partials(4, 1, 128, seed=31)
    narrow = partials.to(torch.bfloat16)
    got = DM.dcp_lse_merge(narrow, lse)
    wide = DM.dcp_lse_merge(narrow.float(), lse)
    torch.testing.assert_close(got, wide, rtol=0.0, atol=0.0)


def test_a_ragged_latent_is_tiled_without_padding():
    """A latent past one 512-wide column tile, with a ragged last tile."""
    partials, lse = make_partials(2, 1, 128, latent=640, seed=41)
    got = DM.dcp_lse_merge(partials, lse)
    assert tuple(got.shape) == (128, 1, 640)
    assert_within_bound(got, partials, lse)


@pytest.mark.parametrize("rows", (512, 200, 300))
def test_two_programs_are_bitwise_one_program(monkeypatch, rows):
    """The LNC2 grid splits whole row tiles between the programs; no arithmetic moves.

    200 rows are two tiles, the second ragged (72 rows), so the second program takes it.
    300 rows are three: the first program takes two and the second one, an uneven split
    fixed at trace time (``reports/dcp_item4.md`` section 14).
    """
    partials, lse = make_partials(4, 1, rows, seed=51, empty_fraction=0.2)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert DM.merge_programs(rows=rows) == 1
    one = DM.dcp_lse_merge(partials, lse)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert DM.merge_programs(rows=rows) == 2
    two = DM.dcp_lse_merge(partials, lse)
    torch.testing.assert_close(two, one, rtol=0.0, atol=0.0)
    assert DM.merge_programs(rows=DM.ROW_TILE) == 1   # one row tile has nothing to split


@pytest.mark.parametrize(
    "partials_shape,partials_dtype,lse_shape,lse_dtype,named",
    [
        ((2, 1, 8, 512), torch.float32, (2, 8), torch.float32, "lse"),
        ((2, 8, 512), torch.float32, (2, 1, 8), torch.float32, "partials"),
        ((2, 1, 8, 512), torch.float32, (2, 1, 9), torch.float32, "lse"),
        ((2, 1, 8, 512), torch.float32, (2, 1, 8), torch.bfloat16, "float32"),
        ((2, 1, 8, 512), torch.float16, (2, 1, 8), torch.float32, "float16"),
        ((129, 1, 8, 512), torch.float32, (129, 1, 8), torch.float32, "128"),
        ((2, 1, 0, 512), torch.float32, (2, 1, 0), torch.float32, "row"),
    ],
)
def test_the_seam_refuses_malformed_operands_by_name(partials_shape, partials_dtype, lse_shape,
                                                    lse_dtype, named):
    with pytest.raises(DM.DcpMergeError, match=named):
        DM.dcp_lse_merge(torch.zeros(partials_shape, dtype=partials_dtype),
                         torch.zeros(lse_shape, dtype=lse_dtype))


def compile_to_neff(fn, args, programs: int) -> None:
    """In a compile child: build ``fn(*args)`` to a NEFF and print one :data:`COMPILE_ROW` row.

    The whole compile runs on the CPU (Dynamo -> HLO -> neuronx-cc -> NEFF, trn2, LNC2), as
    ``test_mla_dense_window_cpu_compile.py`` runs it: ``build_executable`` is replaced, so the
    child never opens the Neuron runtime, and the target is pinned, so nothing asks the
    runtime for one. ``programs`` is the launch grid the seam chose, printed for the parent.
    """
    # The plugin was imported first, by the child's boot line (:data:`_CHILD_BOOT`).
    import vllm_neuron
    import libtorch_neuronx_lite.compile.backend as backend

    from test.vllm_neuron.functional.attention.neuron_device_nodes import (
        open_neuron_device_nodes,
    )

    class _Compiled(Exception):
        pass

    def _no_runtime(*, hlo_filename, neff_filename, **_):
        raise _Compiled(f"NEFF={neff_filename}")

    backend.build_executable = _no_runtime
    neff, message = "", ""
    started = time.monotonic()
    try:
        torch.compile(fn, backend="neuron_libtorch", fullgraph=True)(*args)
        message = "the compiled graph ran; build_executable was not replaced"
    except BaseException as caught:  # the compiler's failure is a result here
        text = " ".join(str(caught).split())
        if "NEFF=" in text:
            neff = text.split("NEFF=")[1].split()[0]
        else:
            message = text[:3000] or type(caught).__name__
    ok = bool(neff) and os.path.isfile(neff)
    print(f"{COMPILE_ROW}|module={vllm_neuron.__file__}|ok={ok}"
          f"|seconds={time.monotonic() - started:.1f}|programs={programs}"
          f"|device_nodes={len(open_neuron_device_nodes())}|neff={neff or 'none'}"
          f"|diagnostic={message or 'none'}", flush=True)


def _compiler_errors(scratch: str) -> str:
    """The distinct ``[NCC_...]`` errors in the compiler's own logs under ``scratch``."""
    found = []
    for log in pathlib.Path(scratch).rglob("log-neuron-cc.txt"):
        for hit in re.finditer(r"\[NCC_\w+\][^\n]{0,240}", log.read_text(errors="ignore")):
            if hit.group(0) not in found:
                found.append(hit.group(0))
    return " ~ ".join(found[:4]) or "no NCC error in the compiler logs"


def compile_in_a_child(test_file: str, *case: str) -> dict[str, str]:
    """Run ``test_file child *case`` in the CPU compile environment and return its row.

    Asserts the row came from this tree, built a NEFF, and opened no device node. A host
    without neuronx-cc fails here rather than skipping, so no suite passes uncompiled.
    """
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    scratch = tempfile.mkdtemp(prefix="dcp_compile_")
    try:
        environment.update(
            NEURON_LIBTORCH_CPU_COMPILE="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
            NEURON_LOGICAL_NC_CONFIG="2", NEURON_LIBTORCH_DISABLE_COMPILE_CACHE="1",
            NEURON_LIBTORCH_CACHE_ROOT=scratch, PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        assert shutil.which("neuronx-cc", path=environment["PATH"]), "neuronx-cc is not installed"
        done = subprocess.run([sys.executable, "-c", _CHILD_BOOT, test_file, "child", *case],
                              cwd=scratch, env=environment, capture_output=True, text=True,
                              timeout=1500, check=False)
        printed = [line for line in done.stdout.splitlines()
                   if line.startswith(COMPILE_ROW + "|")]
        assert printed, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
        print(printed[-1])
        # The diagnostic is the compiler's own text and may hold a pipe: cut it off first.
        head, _, diagnostic = printed[-1].partition("|diagnostic=")
        fields = dict(part.split("=", 1) for part in head.split("|")[1:])
        fields["diagnostic"] = diagnostic
        assert fields["module"].startswith(f"{_ROOT}/"), fields
        assert fields["ok"] == "True", f"{fields['diagnostic']} ~ {_compiler_errors(scratch)}"
        assert fields["device_nodes"] == "0", fields
        return fields
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _child(cp: int, heads: int, rows: int) -> None:
    meta = torch.device("meta")
    partials = torch.empty((cp, heads, rows, LATENT), dtype=torch.float32, device=meta)
    lse = torch.empty((cp, heads, rows), dtype=torch.float32, device=meta)
    compile_to_neff(DM.dcp_lse_merge, (partials, lse), DM.merge_programs(rows))


@pytest.mark.parametrize("cp,heads,rows", MERGE_COMPILES,
                         ids=[f"cp{c}-h{h}-r{r}" for c, h, r in MERGE_COMPILES])
def test_neuronx_cc_builds_the_merge_kernel(cp, heads, rows):
    """neuronx-cc builds the merge at the served rows on both cores of an LNC2 pair."""
    fields = compile_in_a_child(str(pathlib.Path(__file__).resolve()), str(cp), str(heads),
                                str(rows))
    assert fields["programs"] == "2", fields


def test_the_bound_is_tight_enough_to_see_a_dropped_correction():
    """The tolerances are not vacuous.

    An unweighted mean of the partials breaks the bound. A merge whose float32 value is off
    by 2**-16 relative, 256 U32 and well past eps * S, breaks the away-from-ties check.
    """
    partials, lse = make_partials(4, 1, 128, seed=61)
    exact = llama3_merge_reference(partials, lse)
    wrong = partials.mean(dim=0).permute(1, 0, 2).to(torch.bfloat16)
    assert bool(((wrong.double() - exact).abs() > merge_bound(partials, lse)).any())
    nudged = (exact * (1.0 + 2.0 ** -16)).float().to(torch.bfloat16)
    with pytest.raises(AssertionError, match="differ from RNE"):
        assert_rounds_to_nearest_away_from_ties(nudged, partials, lse)


# --------------------------------------------------------------------------- #
# The device benchmark (test/hardware/benchmark_dcp_merge.py), on the CPU
# --------------------------------------------------------------------------- #
_BENCHMARK = _ROOT / "test" / "hardware" / "benchmark_dcp_merge.py"


def _load_benchmark():
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_dcp_merge", _BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    """The benchmark module with an environment that passes every guard: a device is
    reported present (no device is opened), LNC2, a cache root, no simulator."""
    module = _load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark_dcp_merge.py", "--out",
                                      str(tmp_path / "bench.json")])
    monkeypatch.setattr(module, "neuron_devices", lambda: ["/dev/neuron0"])
    for key in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(tmp_path / "cache"))
    return module


@pytest.mark.parametrize("guard,match", (
    ("device", "no Neuron device is visible"),
    ("simulator", "unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR"),
    ("lnc", "NEURON_LOGICAL_NC_CONFIG must be 2"),
    ("cache", "set NEURON_LIBTORCH_CACHE_ROOT"),
))
def test_the_device_benchmark_refuses_an_unfit_environment_by_name(benchmark, monkeypatch,
                                                                    tmp_path, guard, match):
    """Each guard alone stops the run before anything is compiled or written."""
    if guard == "device":
        monkeypatch.setattr(benchmark, "neuron_devices", lambda: [])
    if guard == "simulator":
        monkeypatch.setenv("NKI_SIMULATOR", "1")
    if guard == "lnc":
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    if guard == "cache":
        monkeypatch.delenv("NEURON_LIBTORCH_CACHE_ROOT")
    with pytest.raises(SystemExit, match=match):
        benchmark.main()
    assert not (tmp_path / "bench.json").exists()


def test_the_device_benchmark_takes_no_core_selection_from_the_environment():
    """Core selection is the launcher's: the benchmark neither reads nor sets it."""
    assert "NEURON_RT_VISIBLE_CORES" not in _BENCHMARK.read_text()


def test_the_device_benchmark_s_roofline_counts_each_operand_byte_once():
    """Its bytes are the merge's operands as the seam takes them: the f32 partials and lses
    read once, the bf16 output written once (``reports/dcp_item4.md`` section 9 matches them
    to the compiled kernel's DMA bytes)."""
    module = _load_benchmark()
    for case in module.CASES:
        partials, lse = make_partials(case.cp, case.heads, 2)
        per_row = (partials[0, 0, 0].numel() * partials.element_size() * case.cp * case.heads
                   + lse.element_size() * case.cp * case.heads
                   + LATENT * case.heads * torch.tensor([], dtype=torch.bfloat16).element_size())
        assert module.roofline(case)["bytes"] == per_row * case.rows
        assert module.roofline(case)["binds"] == "bytes"


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    _child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
