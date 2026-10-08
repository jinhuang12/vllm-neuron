# SPDX-License-Identifier: Apache-2.0
"""The fused small-M MLP on both LNC2 cores, against the 0a08ff4 one-core kernel.

Under ``NEURON_LOGICAL_NC_CONFIG=2``, :func:`blockwise_fp8_mlp` launches
``blockwise_fp8_mlp_small_m_kernel`` on a ``[2]`` grid: program 0 runs the gate
projection, program 1 the up projection, the two exchange their activated
halves core to core, and each program runs the down projection for one half of
the hidden columns. The reference is ``blockwise_fp8_mlp`` of the 0a08ff4 file
(``test/hardware/baselines/dense_0a08ff4``), whose fused kernel runs on one core.

Shapes: the per-rank decode shapes at TP=64, hidden 4096, intermediate 128 (shared
expert) and 256 (dense MLP), M in {1, 4, 64} plus the chunk edges 33 and 127 and
the narrow/wide path edge 15/16. M=1 and M=4 run the narrow path, M=64 the wide
path.

Tolerance (round 1, ``test_blockwise_fp8_mlp_small_m.py``): elementwise
``|got - ref| <= 2e-3 * max|ref|``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

bw = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

BASELINE_FILE = (
    Path(__file__).resolve().parents[2]
    / "hardware" / "baselines" / "dense_0a08ff4" / "blockwise_fp8_mm.py"
)
HIDDEN = 4096
SWIGLU_LIMIT = 10.0
#: ``|got - ref| <= TOLERANCE * max|ref|``, the round-1 dense bound.
TOLERANCE = 2e-3
KERNEL = "blockwise_fp8_mlp_small_m_kernel"


@pytest.fixture(autouse=True)
def _simulator(monkeypatch, tmp_path):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    # The NKI driver writes compile artifacts into the working directory.
    monkeypatch.chdir(tmp_path)


def _baseline():
    name = "_dense_0a08ff4_blockwise_fp8_mm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASELINE_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _GridSpy:
    """Records the launch grid of every kernel ``bw`` hands to ``wrap_nki``."""

    def __init__(self):
        self.launches: list[tuple[str, tuple]] = []

    def __call__(self, kernel):
        real = wrap_nki(kernel)
        name = getattr(getattr(kernel, "func", kernel), "__name__", str(kernel))
        launches = self.launches

        class _Caller:
            def __getitem__(self, grid):
                launches.append((name, tuple(grid) if isinstance(grid, (tuple, list))
                                 else (grid,)))
                return real[grid]

            def __call__(self, *args, **kwargs):
                launches.append((name, ()))
                return real(*args, **kwargs)

        return _Caller()


@pytest.fixture
def grid_spy(monkeypatch):
    spy = _GridSpy()
    monkeypatch.setattr(bw, "wrap_nki", spy)
    return spy


def _operands(tokens: int, intermediate: int, seed: int, hidden: int = HIDDEN):
    """Decode-like operands, the generator of the round-1 tests.

    Distinct non-power-of-two block scales, sized so both clamps are active.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=g).to(torch.bfloat16)

    def weight(rows, cols):
        raw = torch.randn((rows, cols), generator=g) * 64
        return raw.clamp(-240, 240).to(torch.float8_e4m3fn)

    def scale(rows, cols, magnitude):
        grid = torch.rand((rows // 128, cols // 128), generator=g) + 0.37
        return (grid * magnitude).to(torch.float32)

    return (
        x,
        weight(hidden, intermediate),
        weight(hidden, intermediate),
        weight(intermediate, hidden),
        scale(hidden, intermediate, 3e-3),
        scale(hidden, intermediate, 3e-3),
        scale(intermediate, hidden, 1e-3),
    )


def _assert_agrees(got, ref):
    assert got.shape == ref.shape
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    bound = TOLERANCE * ref.abs().max().item()
    worst = (got - ref).abs().max().item()
    assert worst <= bound, f"max|got-ref|={worst:.3e} > {bound:.3e}"


def _reference(ops):
    """The 0a08ff4 fused MLP (one core) on the same operands."""
    return _baseline().blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)


@pytest.mark.parametrize("intermediate", [128, 256], ids=["shared", "dense"])
@pytest.mark.parametrize("tokens", [1, 4, 64])
def test_two_core_mlp_matches_the_0a08ff4_kernel(
    tokens, intermediate, monkeypatch, grid_spy
):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    ops = _operands(tokens, intermediate, seed=900 + tokens + intermediate)
    bw.reset_mlp_dispatch_counters()
    bw.reset_dispatch_counters()
    bw.reset_mlp_launch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    # One fused NKI dispatch, launched on a [2] grid; no three-call route and no
    # torch fallback.
    assert grid_spy.launches == [(KERNEL, (2,))]
    assert bw.mlp_dispatch_counters() == (1, 0)
    assert bw.mlp_launch_counters() == (0, 1)
    assert bw.dispatch_counters() == (0, 0)
    _assert_agrees(got, _reference(ops))
    # Both clamps are active somewhere, so the exchange of the clamped gate and
    # up halves is exercised on both sides.
    x, gate_w, up_w, _, gate_s, up_s, _ = ops
    gate = bw.blockwise_fp8_mm_torch_oracle(x, gate_w, gate_s)
    up = bw.blockwise_fp8_mm_torch_oracle(x, up_w, up_s)
    assert (gate > SWIGLU_LIMIT).any() and (up.abs() > SWIGLU_LIMIT).any()


@pytest.mark.parametrize("tokens", [33, 127])
def test_two_core_mlp_chunks_tokens(tokens, monkeypatch, grid_spy):
    # 33 and 127 rows: a short last token chunk on both programs.
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    ops = _operands(tokens, 256, seed=31 + tokens)
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert grid_spy.launches == [(KERNEL, (2,))]
    _assert_agrees(got, _reference(ops))


@pytest.mark.parametrize("tokens", [15, 16])
def test_two_core_mlp_at_the_wide_path_threshold(tokens, monkeypatch, grid_spy):
    # 15 rows take the narrow path, 16 the wide one (Tensor Engine transpose of
    # x, tokens-on-partitions down); both agree with the 0a08ff4 kernel.
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert bw._mlp_wide(tokens, HIDDEN // 128) is (tokens >= bw.MLP_WIDE_MIN_ROWS)
    ops = _operands(tokens, 128, seed=61 + tokens)
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert grid_spy.launches == [(KERNEL, (2,))]
    _assert_agrees(got, _reference(ops))


def test_wide_path_needs_whole_rows_of_the_supported_widths():
    assert bw.MLP_WIDE_MIN_ROWS == 16
    assert not bw._mlp_wide(15, 32)
    assert bw._mlp_wide(16, 32) and bw._mlp_wide(127, 32)
    # H = 256: two k-tiles do not split into the four transpose groups.
    assert not bw._mlp_wide(64, 2)
    # H = 8192: wider than the wide path is built and tested for.
    assert not bw._mlp_wide(64, 64)


@pytest.mark.parametrize(
    ("hidden", "tokens"), [(256, 5), (1024, 5), (1024, 33)], ids=["h256", "h1024", "h1024-wide"]
)
def test_two_core_mlp_off_the_decode_hidden(hidden, tokens, monkeypatch, grid_spy):
    # H = 256 leaves each program one output column per partition (the tiny
    # model's width); H = 1024 four. The scale-block masks differ from H = 4096,
    # and at H = 1024 33 rows run the wide path on eight k-tiles.
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    ops = _operands(tokens, 256, seed=hidden + tokens, hidden=hidden)
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert grid_spy.launches == [(KERNEL, (2,))]
    _assert_agrees(got, _reference(ops))


def test_two_core_rows_are_independent(monkeypatch):
    # Row r of a 4-row call equals the 1-row call on row r: neither the token
    # chunking nor the core exchange mixes rows.
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    x, *rest = _operands(4, 128, seed=77)
    batch = bw.blockwise_fp8_mlp(x, *rest, swiglu_limit=SWIGLU_LIMIT)
    for row in range(4):
        one = bw.blockwise_fp8_mlp(x[row:row + 1], *rest, swiglu_limit=SWIGLU_LIMIT)
        _assert_agrees(batch[row:row + 1], one)


@pytest.mark.parametrize("tokens", [4, 64], ids=["narrow", "wide"])
@pytest.mark.parametrize("config", [None, "1"], ids=["unset", "lnc1"])
def test_one_core_launch_without_lnc2(config, tokens, monkeypatch, grid_spy):
    # Without LNC2 there is one physical core per logical core: the kernel runs
    # on a one-program launch and computes the whole MLP there, on either path.
    if config is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", config)
    ops = _operands(tokens, 256, seed=5 + tokens)
    bw.reset_mlp_launch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert grid_spy.launches == [(KERNEL, ())]
    assert bw.mlp_launch_counters() == (1, 0)
    _assert_agrees(got, _reference(ops))


def test_launch_grid_follows_the_logical_core_config(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert bw.mlp_launch_grid(HIDDEN) == (2,)
    # One 128-row hidden block cannot be split into two column halves.
    assert bw.mlp_launch_grid(128) == ()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    assert bw.mlp_launch_grid(HIDDEN) == ()
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert bw.mlp_launch_grid(HIDDEN) == ()
