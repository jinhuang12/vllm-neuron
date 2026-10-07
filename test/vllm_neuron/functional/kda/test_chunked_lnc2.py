# SPDX-License-Identifier: Apache-2.0
"""The two-core KDA chunked recurrence against 342e93e's kernels, at the prefill shapes.

342e93e runs stages 1 to 3 one chunk at a time and stages 4 and 5 one chunk at a time on one
core: at the layer's chunk of 8 a 1024-token prefill is 128 iterations of ``[8, 128]`` tiles
in each kernel. This tree computes stages 1 to 3 for ``128 // chunk`` chunks per tile, as
block-diagonal ``[128, 128]`` products, and splits the chunks over both programs of an LNC2
launch; stages 4 and 5 hoist everything that does not read the carried state out of the
chunk loop, and split the value columns over the two programs (the state's value columns
are independent: the decay is per key channel).

The arithmetic per element is 342e93e's: every product contracts the same operands in the
same order, a block-diagonal product adds exact zeros, and every elementwise step is the
same fp32 operation. The contract is therefore equality (``torch.equal``: value equality,
which counts ``-0.0`` and ``+0.0`` as equal), not a tolerance.

The reference is the 342e93e snapshot (``test/hardware/baselines/prefill_cores_342e93e``),
run in the same simulator. Token counts: 1024 and 64 (the prefill chunk and a short
prompt), and 8, one chunk, the shortest prefill the chunked kernels see -- a 1-token
prefill takes no chunked call at all (``n_chunks = tokens // chunk`` is 0 and the layer
runs one decode step, ``model_fp8.py`` ``Glm5NextKDAAttention`` step 3).
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.kda import chunked_lnc2 as LN
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.hardware.baselines.prefill_cores_342e93e import load as load_342e93e

#: The layer's chunk at ``gate_lower_bound`` -5 (``_resolve_chunk_size``) and the real
#: per-rank head width: 64 heads of 128 at TP=64 is one head per rank.
CHUNK = 8
HEAD = 128
TOKEN_COUNTS = (8, 64, 1024)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"


def _programs(monkeypatch, n: int) -> None:
    if n == 2:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    else:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)


def _inputs(tokens: int, chunk: int, kdim: int, vdim: int, gate: float, seed: int):
    """One head's chunked operands and an entering state.

    ``gate`` bounds the per-token log decay: ``-gate <= gk <= 0``. At chunk 8 the layer's
    clamp range (5) keeps the chunk-local cumulative gate at most 40, inside the kernels'
    ``GATE_CUMSUM_ABS_LIMIT`` of 60.
    """
    gen = torch.Generator().manual_seed(seed)
    n = tokens // chunk
    q = torch.randn((n, chunk, kdim), generator=gen)
    k = torch.randn((n, chunk, kdim), generator=gen)
    v = torch.randn((n, chunk, vdim), generator=gen)
    beta = torch.rand((n, chunk), generator=gen) * 0.9 + 0.05
    gk = -torch.rand((n, chunk, kdim), generator=gen) * gate
    state = torch.randn((vdim, kdim), generator=gen) * 0.1
    return q, k, v, beta, gk, state


def _assert_equal(got, want, label: str) -> None:
    for name in want._fields:
        g, w = getattr(got, name), getattr(want, name)
        assert tuple(g.shape) == tuple(w.shape), f"{label} {name}: shape {g.shape} != {w.shape}"
        assert g.dtype == w.dtype
        assert torch.equal(g, w), (
            f"{label} {name}: {int((g != w).sum())} of {w.numel()} differ, max "
            f"|diff| {float((g - w).abs().max()):.3e}"
        )


def _intra_case(monkeypatch, programs, tokens, chunk, kdim, vdim, gate, seed):
    base = load_342e93e().chunked_recurrence
    q, k, v, beta, gk, _ = _inputs(tokens, chunk, kdim, vdim, gate, seed)
    want = base.kda_intra_chunk(q, k, v, beta, gk)
    _programs(monkeypatch, programs)
    got = LN.intra_chunk_lnc2(q, k, v, beta, gk)
    _assert_equal(got, want, f"intra T={tokens} C={chunk} programs={programs}")


def _inter_case(monkeypatch, programs, tokens, chunk, kdim, vdim, gate, seed, with_state):
    base = load_342e93e().chunked_recurrence
    q, k, v, beta, gk, state = _inputs(tokens, chunk, kdim, vdim, gate, seed)
    intra = base.kda_intra_chunk(q, k, v, beta, gk)
    entering = state if with_state else None
    want = base.kda_inter_chunk(intra.kg, intra.w, intra.u, gk, q, intra.aqk, state=entering)
    _programs(monkeypatch, programs)
    got = LN.inter_chunk_lnc2(intra.kg, intra.w, intra.u, gk, q, intra.aqk, entering)
    _assert_equal(got, want, f"inter T={tokens} C={chunk} programs={programs}")


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_intra_chunk_equals_342e93e_at_the_prefill_shape(monkeypatch, programs, tokens):
    _intra_case(monkeypatch, programs, tokens, CHUNK, HEAD, HEAD, 5.0, seed=tokens)


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_inter_chunk_equals_342e93e_at_the_prefill_shape(monkeypatch, programs, tokens):
    _inter_case(monkeypatch, programs, tokens, CHUNK, HEAD, HEAD, 5.0, seed=7 + tokens,
                with_state=True)


@pytest.mark.parametrize("programs", [1, 2])
def test_inter_chunk_equals_342e93e_from_a_zero_state(monkeypatch, programs):
    _inter_case(monkeypatch, programs, 64, CHUNK, HEAD, HEAD, 5.0, seed=3, with_state=False)


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_chained_pair_equals_342e93e(monkeypatch, programs, tokens):
    """Live intra into live inter, against 342e93e's pair: the layer's own composition."""
    base = load_342e93e().chunked_recurrence
    q, k, v, beta, gk, state = _inputs(tokens, CHUNK, HEAD, HEAD, 5.0, seed=11 + tokens)
    b_intra = base.kda_intra_chunk(q, k, v, beta, gk)
    want = base.kda_inter_chunk(b_intra.kg, b_intra.w, b_intra.u, gk, q, b_intra.aqk,
                                state=state)
    _programs(monkeypatch, programs)
    intra = LN.intra_chunk_lnc2(q, k, v, beta, gk)
    got = LN.inter_chunk_lnc2(intra.kg, intra.w, intra.u, gk, q, intra.aqk, state)
    _assert_equal(got, want, f"chain T={tokens} programs={programs}")


#: ``(chunk, n_chunks)`` beyond the layer's choice: a chunk that fills the tile alone, two
#: and four chunks a tile, a ragged final tile, and the narrowest chunk the kernels admit.
OTHER_WIDTHS = ((128, 2), (64, 3), (32, 5), (16, 9), (2, 70))


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("chunk,n_chunks", OTHER_WIDTHS)
def test_other_chunk_widths_equal_342e93e(monkeypatch, programs, chunk, n_chunks):
    _intra_case(monkeypatch, programs, chunk * n_chunks, chunk, 64, 64, 0.05,
                seed=chunk + n_chunks)
    _inter_case(monkeypatch, programs, chunk * n_chunks, chunk, 64, 64, 0.05,
                seed=chunk + 2 * n_chunks, with_state=True)


def test_program_counts(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert LN.intra_programs(128) == 1 and LN.inter_programs(128) == 1
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert LN.intra_programs(1) == 1
    assert LN.intra_programs(2) == 2
    assert LN.intra_programs(128) == 2
    assert LN.inter_programs(128) == 2
    assert LN.inter_programs(1) == 1
    assert LN.inter_programs(63) == 1


def _launch_grids(monkeypatch, call) -> list:
    seen = []
    real = LN.wrap_nki

    def spy(kernel):
        inner_call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                inner = inner_call[grid]
                return lambda *a, **kw: (seen.append((grid, a, kw)), inner(*a, **kw))[1]

            def __call__(self, *a, **kw):
                seen.append((1, a, kw))
                return inner_call(*a, **kw)
        return _Spy()

    monkeypatch.setattr(LN, "wrap_nki", spy)
    call()
    return seen


def test_lnc2_launches_are_two_program_grids_keyed_by_the_digest(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    q, k, v, beta, gk, state = _inputs(64, CHUNK, HEAD, HEAD, 5.0, seed=1)
    def both():
        intra = LN.intra_chunk_lnc2(q, k, v, beta, gk)
        LN.inter_chunk_lnc2(intra.kg, intra.w, intra.u, gk, q, intra.aqk, state)

    seen = _launch_grids(monkeypatch, both)
    assert [grid for grid, _, _ in seen] == [2, 2]
    for _, args, kwargs in seen:
        assert kwargs.get("source_digest", args[-1] if args else None) == LN.SOURCE_DIGEST


def test_one_chunk_is_one_intra_program(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    q, k, v, beta, gk, _ = _inputs(8, CHUNK, HEAD, HEAD, 5.0, seed=2)
    seen = _launch_grids(monkeypatch, lambda: LN.intra_chunk_lnc2(q, k, v, beta, gk))
    assert [grid for grid, _, _ in seen] == [1]


def _entry_pair(q, k, v, beta, gk, state):
    """``chunked_recurrence``'s two entry points, as ``Glm5NextKDAAttention`` calls them."""
    from vllm_neuron.functional.kda import chunked_recurrence as chunked
    intra = chunked.kda_intra_chunk(q, k, v, beta, gk)
    inter = chunked.kda_inter_chunk(intra.kg, intra.w, intra.u, gk, q, intra.aqk, state=state)
    return intra, inter


def _counted(monkeypatch, run):
    from vllm_neuron.functional.kda import chunked_recurrence as chunked
    chunked.reset_dispatch_counters()
    chunked.reset_inter_dispatch_counters()
    out = {}
    seen = _launch_grids(monkeypatch, lambda: out.update(pair=run()))
    return out["pair"], seen, (chunked.dispatch_counters(), chunked.inter_dispatch_counters())


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_entry_points_launch_the_two_core_kernels_on_lnc2(monkeypatch, tokens):
    """On LNC2 the layer's two calls launch this module's kernels, count one dispatch each,
    and return 342e93e's values."""
    base = load_342e93e().chunked_recurrence
    q, k, v, beta, gk, state = _inputs(tokens, CHUNK, HEAD, HEAD, 5.0, seed=23 + tokens)
    b_intra = base.kda_intra_chunk(q, k, v, beta, gk)
    b_inter = base.kda_inter_chunk(b_intra.kg, b_intra.w, b_intra.u, gk, q, b_intra.aqk,
                                   state=state)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.delenv(LN.CHUNKED_LNC2_ENV, raising=False)
    (intra, inter), seen, counters = _counted(
        monkeypatch, lambda: _entry_pair(q, k, v, beta, gk, state))
    one_chunk = tokens == CHUNK
    assert [grid for grid, _, _ in seen] == [1 if one_chunk else 2, 2]
    assert counters == ((1, 0), (1, 0))
    _assert_equal(intra, b_intra, f"entry intra T={tokens}")
    _assert_equal(inter, b_inter, f"entry inter T={tokens}")


@pytest.mark.parametrize("lnc,switch", [("2", "0"), (None, None)])
def test_the_entry_points_keep_the_one_core_kernels_when_switched_off_or_not_lnc2(
        monkeypatch, lnc, switch):
    """``VLLM_NEURON_KDA_CHUNKED_LNC2=0``, or a runtime that is not LNC2: the entry points
    launch 342e93e's kernels (no launch of this module's), counted as before."""
    q, k, v, beta, gk, state = _inputs(64, CHUNK, HEAD, HEAD, 5.0, seed=29)
    for name, value in (("NEURON_LOGICAL_NC_CONFIG", lnc), (LN.CHUNKED_LNC2_ENV, switch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert not LN.chunked_lnc2_enabled()
    _, seen, counters = _counted(monkeypatch, lambda: _entry_pair(q, k, v, beta, gk, state))
    assert seen == []
    assert counters == ((1, 0), (1, 0))
