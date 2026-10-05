# SPDX-License-Identifier: Apache-2.0
"""The decode-path mHC kernels against their 5938748 versions, at real decode shapes.

One mHC site is one ``sinkhorn_normalise_blocks`` call (in ``mhc_pre``) and one
``hyper_connection_combine`` call (in ``mhc_post``). The target's decode shapes are
``hidden = 4096`` and ``hc_mult = 4``; ``B`` is the token axis, run at 1 (today's
step) and 4 (so nothing here leans on ``B == 1``).

The before-side is the unchanged source snapshot under
``test/hardware/baselines/mhc_5938748/`` (``git show 5938748:<path>``), run through
the same simulator. The new side runs through the module entry points, and every
numeric case asserts the NKI path served it: the module's dispatch counter reads
one, its torch-fallback counter reads zero, and ``nki.simulator.simulate_kernel``
ran once per kernel call.

Tolerances, stated once:

* combine: :data:`COMBINE_ATOL` / :data:`COMBINE_RTOL`. The new kernel performs the
  old one's fp32 operations in the old order (``post_j * x`` first, then
  ``+ comb_ij * residual_i`` for ``i = 0..S-1``, each product rounded before its
  add), so the two agree bit for bit; the bound is 0.
* Sinkhorn: :data:`SINKHORN_ATOL` / :data:`SINKHORN_RTOL`. At these shapes the new
  kernel carries the iteration as the two scaling vectors ``u = 1 / (K v)`` and
  ``v = 1 / (K^T u)`` and forms the entries ``u_i K_ij v_j`` once at the end, where
  the old one rescaled the entries on every pass. The algebra is the same, so only
  fp32 rounding differs; the iteration contracts, so the difference stays at a few
  fp32 ulps of the ``O(1/S)`` entries (about 1.2e-7 observed, against 2e-6).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

import nki.isa
import nki.language
import nki.simulator

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.mhc import hyper_connection as combine_mod
from vllm_neuron.functional.mhc import sinkhorn as sinkhorn_mod

#: Real decode shapes.
HIDDEN = 4096
S = sinkhorn_mod.MHC_STREAMS
ITERS = sinkhorn_mod.SINKHORN_ITERS
BATCHES = (1, 4)

COMBINE_ATOL = 0.0
COMBINE_RTOL = 0.0
SINKHORN_ATOL = 2e-6
SINKHORN_RTOL = 1e-5

#: The hc_eps the layer adds after the comb softmax (``text_config.hc_eps``).
HC_EPS = 1e-6

_REPO = pathlib.Path(__file__).resolve().parents[4]
_BASELINE_DIR = _REPO / "test" / "hardware" / "baselines" / "mhc_5938748"


def _load_snapshot(name: str):
    """Import one 5938748 source file under a private module name."""
    path = _BASELINE_DIR / f"{name}.py"
    assert path.is_file(), f"missing baseline snapshot {path}"
    module_name = f"_mhc_5938748_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


OLD_SINKHORN = _load_snapshot("sinkhorn")
OLD_COMBINE = _load_snapshot("hyper_connection")


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration."""

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


#: Engine instructions (not DMA) the census counts, by ``nki.isa`` / ``nki.language``
#: name. DMA (``dma_copy``, ``nl.load``, ``nl.store``) is counted separately.
_ENGINE_OPS_ISA = (
    "tensor_tensor",
    "tensor_scalar",
    "scalar_tensor_tensor",
    "tensor_reduce",
    "tensor_partition_reduce",
    "tensor_scalar_reduce",
    "tensor_scalar_cumulative",
    "tensor_copy",
    "tensor_copy_predicated",
    "reciprocal",
    "activation",
    "activation_reduce",
    "memset",
    "iota",
    "affine_select",
    "range_select",
    "nc_matmul",
    "nc_transpose",
    "nc_stream_shuffle",
    "tensor_tensor_scan",
    "select_reduce",
    "bn_stats",
    "bn_aggr",
    "max8",
)
_ENGINE_OPS_LANG = ("sum", "broadcast_to")
_DMA_OPS_ISA = ("dma_copy", "dma_transpose", "dma_compute")
_DMA_OPS_LANG = ("load", "store")


class _OpCensus:
    """Records every engine and DMA call a simulated kernel makes.

    The simulator executes the kernel body as Python, so wrapping the module
    attributes the body resolves at call time sees every instruction it issues.
    ``engine`` holds ``(name, partition_extent_of_dst)`` per engine op.
    """

    def __init__(self) -> None:
        self.engine: list[tuple[str, int]] = []
        self.dma: list[str] = []
        self._saved: list[tuple[object, str, object]] = []

    def _wrap(self, module, name: str, kind: str) -> None:
        real = getattr(module, name, None)
        if real is None:
            return
        census = self

        def recording(*args, **kwargs):
            if kind == "dma":
                census.dma.append(name)
            elif name != "broadcast_to":
                dst = kwargs.get("dst")
                if dst is None and args and name not in ("sum",):
                    dst = args[0]
                if dst is None and name == "sum":
                    data = kwargs.get("x", args[0] if args else None)
                    extent = int(data.shape[0]) if data is not None else -1
                else:
                    extent = int(dst.shape[0]) if dst is not None else -1
                census.engine.append((name, extent))
            return real(*args, **kwargs)

        self._saved.append((module, name, real))
        setattr(module, name, recording)

    def __enter__(self) -> "_OpCensus":
        for name in _ENGINE_OPS_ISA:
            self._wrap(nki.isa, name, "engine")
        for name in _ENGINE_OPS_LANG:
            self._wrap(nki.language, name, "engine")
        for name in _DMA_OPS_ISA:
            self._wrap(nki.isa, name, "dma")
        for name in _DMA_OPS_LANG:
            self._wrap(nki.language, name, "dma")
        return self

    def __exit__(self, *exc_info) -> None:
        for module, name, real in reversed(self._saved):
            setattr(module, name, real)
        self._saved.clear()


def _site_inputs(batch: int, seed: int = 5938748):
    """One decode site's tensors: fp32, as the layer hands them to both kernels.

    ``comb_start`` is the layer's own form, ``softmax(logits) + hc_eps``, with
    logits at a spread that leaves the matrix far from doubly stochastic, so 20
    iterations do real work. ``post`` spans the target's ``[0, 2]`` gate range.
    """
    g = torch.Generator().manual_seed(seed + batch)
    x = torch.randn((batch, HIDDEN), generator=g, dtype=torch.float32)
    residual = torch.randn((batch, S, HIDDEN), generator=g, dtype=torch.float32)
    post = 2.0 * torch.rand((batch, S, 1), generator=g, dtype=torch.float32)
    logits = 2.0 * torch.randn((batch, S, S), generator=g, dtype=torch.float32)
    comb_start = torch.softmax(logits, dim=-1) + HC_EPS
    return x, residual, post, comb_start


def _old_sinkhorn(comb_start: torch.Tensor) -> torch.Tensor:
    return wrap_nki(OLD_SINKHORN.sinkhorn_blocks_kernel)(
        affinity_blocks=comb_start, iters=ITERS
    ).to(torch.float32)


def _old_combine(x, residual, post, comb) -> torch.Tensor:
    return wrap_nki(OLD_COMBINE.hyper_connection_kernel)(
        x=x, residual=residual, post_layer_mix=post, comb_res_mix=comb
    ).to(torch.float32)


def _new_sinkhorn(comb_start: torch.Tensor) -> torch.Tensor:
    sinkhorn_mod.reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_mod.sinkhorn_normalise_blocks(comb_start, iters=ITERS)
    assert sinkhorn_mod.dispatch_counters() == (1, 0), sinkhorn_mod.dispatch_counters()
    assert sim.calls == 1, f"simulate_kernel ran {sim.calls} times, expected 1"
    return got.to(torch.float32)


def _new_combine(x, residual, post, comb) -> torch.Tensor:
    combine_mod.reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = combine_mod.hyper_connection_combine(x, residual, post, comb)
    assert combine_mod.dispatch_counters() == (1, 0), combine_mod.dispatch_counters()
    assert sim.calls == 1, f"simulate_kernel ran {sim.calls} times, expected 1"
    return got.to(torch.float32)


def test_the_snapshots_are_the_5938748_sources() -> None:
    """The before-side is the old code, not a copy of the new one."""
    assert OLD_SINKHORN.__file__ != sinkhorn_mod.__file__
    assert OLD_COMBINE.__file__ != combine_mod.__file__
    old_src = pathlib.Path(OLD_SINKHORN.__file__).read_text()
    assert "block_scalar_engine" in old_src, "the sinkhorn snapshot is not 5938748's"
    old_comb_src = pathlib.Path(OLD_COMBINE.__file__).read_text()
    assert "HIDDEN_TILE = 2048" in old_comb_src, "the combine snapshot is not 5938748's"


@pytest.mark.parametrize("batch", BATCHES)
def test_new_sinkhorn_matches_5938748_at_decode_shapes(batch: int) -> None:
    """``[B, 4, 4]`` blocks, 20 iterations: new equals old within the stated bound."""
    _, _, _, comb_start = _site_inputs(batch)
    old = _old_sinkhorn(comb_start)
    new = _new_sinkhorn(comb_start)

    assert tuple(new.shape) == (batch, S, S)
    assert float((old - comb_start).abs().max()) > 1e-3, (
        "the old kernel left its input almost unchanged, so this comparison is vacuous"
    )
    torch.testing.assert_close(new, old, rtol=SINKHORN_RTOL, atol=SINKHORN_ATOL)
    # On its own terms, independent of the old kernel: the last pass of every
    # iteration is the column pass, so column sums are 1 to fp32 rounding. Row
    # sums are still converging after 20 iterations at this logit spread (the
    # algorithm's property, shared by the old kernel), so they get a loose bound.
    torch.testing.assert_close(
        new.sum(dim=-2), torch.ones(batch, S), rtol=0.0, atol=1e-5
    )
    torch.testing.assert_close(
        new.sum(dim=-1), torch.ones(batch, S), rtol=0.0, atol=2e-2
    )


@pytest.mark.parametrize(
    "tokens",
    [
        sinkhorn_mod.SCALING_VECTORS_MAX_TOKENS,
        sinkhorn_mod.SCALING_VECTORS_MAX_TOKENS + 1,
        33,
        129,
    ],
)
def test_new_sinkhorn_matches_5938748_either_side_of_the_switch(tokens: int) -> None:
    """Both Sinkhorn forms match 5938748 within the stated bound.

    The last token count the scaling-vector form serves, the first the
    tokens-on-partitions form serves, and two partition-tile edges of the latter.
    """
    g = torch.Generator().manual_seed(tokens)
    logits = 2.0 * torch.randn((tokens, S, S), generator=g, dtype=torch.float32)
    comb_start = torch.softmax(logits, dim=-1) + HC_EPS
    old = _old_sinkhorn(comb_start)
    new = _new_sinkhorn(comb_start)
    torch.testing.assert_close(new, old, rtol=SINKHORN_RTOL, atol=SINKHORN_ATOL)


#: The two launch forms of the new combine: one program, and the two-program SPMD
#: launch the entry point takes under ``NEURON_LOGICAL_NC_CONFIG=2`` (serving).
PROGRAMS = (1, 2)


def _set_programs(monkeypatch: pytest.MonkeyPatch, programs: int) -> None:
    if programs == 2:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    else:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert combine_mod.launch_programs(HIDDEN) == programs


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("batch", BATCHES)
def test_new_combine_matches_5938748_at_decode_shapes(
    batch: int, programs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``x [B, 4096]``, ``residual [B, 4, 4096]``: new equals old within the bound."""
    _set_programs(monkeypatch, programs)
    x, residual, post, comb_start = _site_inputs(batch)
    comb = _old_sinkhorn(comb_start)
    old = _old_combine(x, residual, post, comb)
    new = _new_combine(x, residual, post, comb)

    assert tuple(new.shape) == (batch, S, HIDDEN)
    assert float(old.abs().max()) > 0.0
    torch.testing.assert_close(new, old, rtol=COMBINE_RTOL, atol=COMBINE_ATOL)


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("batch", BATCHES)
def test_one_new_site_matches_one_5938748_site(
    batch: int, programs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sinkhorn then combine, each side end to end on its own kernels."""
    _set_programs(monkeypatch, programs)
    x, residual, post, comb_start = _site_inputs(batch)
    old = _old_combine(x, residual, post, _old_sinkhorn(comb_start))
    new = _new_combine(x, residual, post, _new_sinkhorn(comb_start))
    # The Sinkhorn difference is the only one, and the combine is linear in the
    # mix with |residual| = O(1) per stream, so it moves the output by at most
    # S * SINKHORN_ATOL * max|residual| plus fp32 rounding of the sum.
    bound = S * SINKHORN_ATOL * float(residual.abs().max()) + 1e-5
    torch.testing.assert_close(new, old, rtol=0.0, atol=bound)


#: Engine ops the new Sinkhorn may issue outside its iteration loop: building the
#: block-diagonal ``K`` and its transpose, the two starting vectors, and forming the
#: entries at the end. A bound, so work cannot move out of the loop unseen.
SINKHORN_FIXED_OPS_MAX = 12


def _sinkhorn_engine_ops(kernel, comb_start: torch.Tensor, iters: int) -> list:
    """Every engine op one simulated Sinkhorn dispatch issues, in order."""
    with _OpCensus() as census, _SimulatorCounter() as sim:
        wrap_nki(kernel)(affinity_blocks=comb_start, iters=iters)
    assert sim.calls == 1, f"simulate_kernel ran {sim.calls} times, expected 1"
    return census.engine


@pytest.mark.parametrize("batch", BATCHES)
def test_sinkhorn_issues_at_most_four_engine_ops_per_iteration(batch: int) -> None:
    """One more iteration costs the new kernel at most 4 engine ops; 5938748 paid 31.

    Counted, not read off the source: each kernel runs at ``ITERS`` and at
    ``ITERS + 1`` iterations and the census difference is the per-iteration cost.
    """
    _, _, _, comb_start = _site_inputs(batch)
    new = sinkhorn_mod.sinkhorn_blocks_kernel
    old = OLD_SINKHORN.sinkhorn_blocks_kernel
    new_base = _sinkhorn_engine_ops(new, comb_start, ITERS)
    new_more = _sinkhorn_engine_ops(new, comb_start, ITERS + 1)
    old_base = _sinkhorn_engine_ops(old, comb_start, ITERS)
    old_more = _sinkhorn_engine_ops(old, comb_start, ITERS + 1)

    per_iteration = len(new_more) - len(new_base)
    old_per_iteration = len(old_more) - len(old_base)
    assert per_iteration <= 4, (
        f"B={batch}: one iteration costs {per_iteration} engine ops; "
        f"the extra ones: {new_more[len(new_base):]}"
    )
    assert [name for name, _ in new_more[-(4 + 4):-4]] == [
        "nc_matmul",
        "reciprocal",
        "nc_matmul",
        "reciprocal",
    ], new_more
    fixed = len(new_base) - per_iteration * ITERS
    assert 0 <= fixed <= SINKHORN_FIXED_OPS_MAX, (
        f"B={batch}: {fixed} engine ops outside the loop; ops: {new_base}"
    )
    # The baseline really is the 5938748 cost this compares against.
    assert old_per_iteration == 31, old_per_iteration


@pytest.mark.parametrize("programs", PROGRAMS)
@pytest.mark.parametrize("batch", BATCHES)
def test_combine_lays_hidden_across_all_128_partitions(
    batch: int, programs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every combine engine op runs on 128 partitions, not on B token lanes."""
    _set_programs(monkeypatch, programs)
    x, residual, post, comb_start = _site_inputs(batch)
    comb = _old_sinkhorn(comb_start)
    with _OpCensus() as census:
        _new_combine(x, residual, post, comb)
    assert census.engine, "the census saw no engine op, so it measured nothing"
    narrow = [op for op in census.engine if op[1] != sinkhorn_mod.PARTITION_MAX]
    assert not narrow, (
        f"B={batch}: {len(narrow)} of {len(census.engine)} combine engine ops ran on "
        f"fewer than {sinkhorn_mod.PARTITION_MAX} partitions: {sorted(set(narrow))}"
    )
    # Per program: 1 post term + 2 ops (multiply, add) per input stream, for the
    # whole token chunk at once, so the count does not grow with B at decode sizes.
    assert len(census.engine) == programs * (1 + 2 * S), (
        f"B={batch} programs={programs}: {len(census.engine)} engine ops; "
        f"ops: {census.engine}"
    )


@pytest.mark.parametrize(
    ("tokens", "hidden"),
    [
        (combine_mod.HIDDEN_ON_PARTITIONS_MAX_TOKENS, HIDDEN),
        (combine_mod.HIDDEN_ON_PARTITIONS_MAX_TOKENS + 1, HIDDEN),
        (combine_mod.HIDDEN_ON_PARTITIONS_MAX_TOKENS, 12288),
        (129, 4097),
        (7, 1023),
        (300, 256),
        (2, 7168),
    ],
)
def test_two_program_launch_matches_5938748_off_the_decode_shapes(
    tokens: int, hidden: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SPMD split matches 5938748 in both layouts, on either side of the switch.

    The first two cases are the last token count served with hidden on the
    partitions and the first served with tokens on them. ``4097`` splits
    2049 / 2048, so the two programs get unequal halves; ``7 x 1023`` leaves a
    short last partition in the hidden layout; ``300`` tokens span three token
    tiles in the token layout; ``32 x 12288`` is 6144 hidden per program, too wide
    for 32 tokens in one SBUF chunk, so the hidden layout walks two chunks.
    """
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert combine_mod.launch_programs(hidden) == 2
    g = torch.Generator().manual_seed(tokens * 7 + hidden)
    x = torch.randn((tokens, hidden), generator=g)
    residual = torch.randn((tokens, S, hidden), generator=g)
    post = 2.0 * torch.rand((tokens, S, 1), generator=g)
    comb = torch.softmax(torch.randn((tokens, S, S), generator=g), dim=-1)
    old = _old_combine(x, residual, post, comb)
    new = _new_combine(x, residual, post, comb)
    torch.testing.assert_close(new, old, rtol=COMBINE_RTOL, atol=COMBINE_ATOL)


def test_launch_programs_follows_the_lnc_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two programs under LNC2 for any hidden width of two partitions or more."""
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert combine_mod.launch_programs(HIDDEN) == 1
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    assert combine_mod.launch_programs(HIDDEN) == 1
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert combine_mod.launch_programs(HIDDEN) == 2
    assert combine_mod.launch_programs(2 * sinkhorn_mod.PARTITION_MAX) == 2
    assert combine_mod.launch_programs(2 * sinkhorn_mod.PARTITION_MAX - 1) == 1
