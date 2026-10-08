# SPDX-License-Identifier: Apache-2.0
"""The packed-tile layout of the KDA chunked recurrence kernels.

Both kernels place ``MAX_TILE // C`` chunks of ``C`` tokens on the partitions of one
tile, so the production chunk width 16 fills a tile with 8 chunks. These cases
cover what the layout adds over the single-chunk cases in
``test_chunked_recurrence``: several full tiles, a partial tail tile, a chunk
width that does not divide the tile, and the two-program launch an LNC2 runtime
takes, which must reproduce the one-program launch bit for bit.

Every expected value is computed here, from the torch references or from the
one-program launch of the same kernel; no value is restated.
"""

from __future__ import annotations

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.kda import chunked_recurrence as cr

#: The production chunk width, the key/value width a TP=64 rank sees, and the
#: tolerances ``test_chunked_recurrence`` uses against the same references.
CHUNK = 16
KDIM = 128
VDIM = 128
RTOL = 1e-2
ATOL = 1e-5

#: The checkpoint's gate lower bound: a log gate per token in ``(LOWER, 0)``.
LOWER = -5.0


def _chunked_inputs(n_chunks: int, chunk: int, seed: int, vdim: int = VDIM):
    gen = torch.Generator().manual_seed(seed)
    shape_k = (n_chunks, chunk, KDIM)
    q = torch.randn(shape_k, generator=gen)
    k = torch.randn(shape_k, generator=gen)
    v = torch.randn((n_chunks, chunk, vdim), generator=gen)
    beta = torch.sigmoid(torch.randn((n_chunks, chunk), generator=gen))
    gk = LOWER * torch.sigmoid(torch.randn(shape_k, generator=gen) * 2)
    return q, k, v, beta, gk


def _tiles(n_chunks: int, chunk: int) -> tuple[int, int]:
    """``(full tiles, tail rows)`` of the packed layout, from the layout's own rule."""
    tile_rows = (cr.MAX_TILE // chunk) * chunk
    rows = n_chunks * chunk
    return rows // tile_rows, rows % tile_rows


def _intra_direct(q, k, v, beta, gk, grid=()):
    consts = cr.chunk_constants(int(q.shape[1]))
    call = wrap_nki(cr.kda_intra_chunk_kernel)
    if grid:
        call = call[grid]
    return call(
        q_hbm=q, k_hbm=k, v_hbm=v, beta_hbm=beta.unsqueeze(-1).contiguous(),
        gk_hbm=gk, triu_hbm=consts.triu_ones, eye_hbm=consts.eye,
        mask_lower_hbm=consts.mask_lower, last_row_hbm=consts.last_row,
    )


def _inter_direct(kg, w, u, gk, q, aqk, state, grid=()):
    _, chunk, kdim = kg.shape
    consts = cr.inter_chunk_constants(chunk, kdim, int(u.shape[2]))
    call = wrap_nki(cr.kda_inter_chunk_kernel)
    if grid:
        call = call[grid]
    return call(
        kg_hbm=kg, w_hbm=w, u_hbm=u, gk_hbm=gk, q_hbm=q, aqk_hbm=aqk,
        triu_hbm=consts.triu_ones, last_col_hbm=consts.last_col,
        state_init_hbm=state.contiguous(),
    )


@pytest.mark.parametrize("n_chunks", [20, 32])
def test_intra_chunk_packed_tiles_match_the_reference(n_chunks):
    """Several chunks per tile, with and without a partial tail tile."""
    full, tail = _tiles(n_chunks, CHUNK)
    assert full >= 1 and (tail > 0) == (n_chunks == 20)
    q, k, v, beta, gk = _chunked_inputs(n_chunks, CHUNK, seed=3100 + n_chunks)
    got = cr.kda_intra_chunk(q, k, v, beta, gk)
    expected = cr.kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    for field in cr.IntraChunkOutputs._fields:
        assert_close(
            getattr(got, field).float(), getattr(expected, field), rtol=RTOL,
            atol=ATOL, name=f"packed_intra[n_chunks={n_chunks}].{field}",
        )
    product = cr.rebuild_i_plus_a(k, beta, gk) @ got.a_inv.float()
    identity = torch.eye(CHUNK).expand(n_chunks, CHUNK, CHUNK)
    assert_close(product, identity, rtol=0.0, atol=ATOL, name="packed_intra.(I+A)@inv")


@pytest.mark.parametrize("chunk,n_chunks", [(CHUNK, 20), (24, 11)])
def test_inter_chunk_packed_tiles_match_the_sequential_scan(chunk, n_chunks):
    """The carry crosses tile boundaries, including a chunk width that leaves
    unused partitions at the bottom of every tile (24 does not divide 128)."""
    full, tail = _tiles(n_chunks, chunk)
    assert full >= 1 and tail > 0
    tokens = n_chunks * chunk
    gen = torch.Generator().manual_seed(4200 + chunk)
    flat = (
        torch.randn((tokens, KDIM), generator=gen),
        torch.randn((tokens, KDIM), generator=gen),
        torch.randn((tokens, VDIM), generator=gen),
        torch.sigmoid(torch.randn(tokens, generator=gen)),
        # Scaled so the chunk-local cumulative gate stays inside the limit at
        # either width: at most |LOWER| * CHUNK, as at the production width.
        LOWER * (CHUNK / chunk)
        * torch.sigmoid(torch.randn((tokens, KDIM), generator=gen) * 2),
    )
    q, k, v, beta, gk = flat
    # The intra-chunk kernel serves power-of-two widths only, so the inter-chunk
    # operands come from the torch reference, which serves any width.
    shape = (n_chunks, chunk, KDIM)
    intra = cr.kda_intra_chunk_torch_oracle(
        q.reshape(shape), k.reshape(shape), v.reshape(n_chunks, chunk, VDIM),
        beta.reshape(n_chunks, chunk), gk.reshape(shape),
    )
    cr.reset_inter_dispatch_counters()
    out = cr.kda_inter_chunk(
        intra.kg, intra.w, intra.u, gk.reshape(shape), q.reshape(shape), intra.aqk
    )
    assert cr.inter_dispatch_counters() == (1, 0)
    ref = cr.kda_inter_chunk_torch_oracle(
        intra.kg, intra.w, intra.u, gk.reshape(shape), q.reshape(shape), intra.aqk
    )
    for field in cr.InterChunkOutputs._fields:
        assert_close(
            getattr(out, field).float(), getattr(ref, field), rtol=RTOL, atol=ATOL,
            name=f"packed_inter[chunk={chunk}].{field}",
        )


def test_intra_chunk_grid_splits_whole_tiles_only_on_lnc2(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert cr.intra_chunk_grid(128, CHUNK) == ()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    tiles_per_program = 4
    whole = tiles_per_program * cr.MAX_TILE // CHUNK
    assert cr.intra_chunk_grid(2 * whole, CHUNK) == (2,)
    # An odd tile count, or a partial tile, keeps one program.
    assert cr.intra_chunk_grid(cr.MAX_TILE // CHUNK, CHUNK) == ()
    assert cr.intra_chunk_grid(2 * whole + 1, CHUNK) == ()


def test_inter_chunk_grid_splits_even_value_widths_only_on_lnc2(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert cr.inter_chunk_grid(VDIM) == ()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert cr.inter_chunk_grid(VDIM) == (2,)
    assert cr.inter_chunk_grid(VDIM - 1) == ()


def test_intra_chunk_two_program_launch_is_bit_equal_to_one_program():
    n_chunks = 2 * cr.MAX_TILE // CHUNK
    q, k, v, beta, gk = _chunked_inputs(n_chunks, CHUNK, seed=5100)
    one = _intra_direct(q, k, v, beta, gk)
    two = _intra_direct(q, k, v, beta, gk, grid=(2,))
    for field, a, b in zip(cr.IntraChunkOutputs._fields, one, two):
        assert torch.equal(a, b), f"intra {field}: the two launches differ"


def test_inter_chunk_two_program_launch_is_bit_equal_to_one_program():
    n_chunks = 2 * cr.MAX_TILE // CHUNK
    q, k, v, beta, gk = _chunked_inputs(n_chunks, CHUNK, seed=5200)
    intra = cr.kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    state = torch.randn((VDIM, KDIM), generator=torch.Generator().manual_seed(5201))
    args = (intra.kg, intra.w, intra.u, gk, q, intra.aqk, state)
    one = _inter_direct(*args)
    two = _inter_direct(*args, grid=(2,))
    for field, a, b in zip(cr.InterChunkOutputs._fields, one, two):
        assert torch.equal(a, b), f"inter {field}: the two launches differ"


def test_the_entry_points_take_the_two_program_launch_on_lnc2(monkeypatch):
    """Through the counted entry points, under LNC2: the same values as the
    one-program kernels, so the grid the entry chose served the call."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    n_chunks = 2 * cr.MAX_TILE // CHUNK
    assert cr.intra_chunk_grid(n_chunks, CHUNK) == (2,)
    q, k, v, beta, gk = _chunked_inputs(n_chunks, CHUNK, seed=5300)
    got = cr.kda_intra_chunk(q, k, v, beta, gk)
    one = _intra_direct(q, k, v, beta, gk)
    for field, a in zip(cr.IntraChunkOutputs._fields, one):
        assert torch.equal(getattr(got, field), a), f"intra entry {field}"
    out = cr.kda_inter_chunk(got.kg, got.w, got.u, gk, q, got.aqk)
    zero = torch.zeros((VDIM, KDIM))
    inter_one = _inter_direct(got.kg, got.w, got.u, gk, q, got.aqk, zero)
    for field, a in zip(cr.InterChunkOutputs._fields, inter_one):
        assert torch.equal(getattr(out, field), a), f"inter entry {field}"
