# SPDX-License-Identifier: Apache-2.0
"""KDA chunked recurrence for a prefill, on both cores of an LNC2 core.

:mod:`~vllm_neuron.functional.kda.chunked_recurrence` computes the recurrence one chunk at a
time on one core. At the layer's chunk of 8 (``Glm5NextKDAAttention._resolve_chunk_size`` at
``gate_lower_bound`` -5) a 1024-token prefill is 128 iterations of ``[8, 128]`` tiles in each
of its two kernels, so the cost is the instruction count: about 160 instructions per chunk,
2.4 ms per KDA layer on pcore 0 with pcore 1 idle (342e93e, trn2 slice benchmark). The two
kernels here compute the same values with far fewer, wider instructions, on both cores.

Stages 1 to 3 (:func:`intra_chunk_lnc2`)
    Every stage is chunk-local, so ``128 // chunk`` chunks share one ``[128, 128]`` tile: the
    tokens of those chunks sit on the partitions, and every chunk-local ``[C, C]`` product of
    342e93e becomes one block-diagonal ``[128, 128]`` product. The host-built constants are
    the block-diagonal forms of 342e93e's (the cumulative-sum ones matrix, the strictly lower
    mask, the last-row selector), so a product contracts the same operands in the same order
    and adds exact zeros for the other chunks; the inverse ``(I + A)**-1`` is the same
    doubling series with ``log2(C)`` stages, because a block-diagonal nilpotent ``N`` has
    ``N**C == 0``. The ``[C, C]`` outputs ``a_inv`` and ``aqk`` are read off the diagonal
    blocks by one matmul against a 0/1 column selector. The chunks split over the two
    programs of the launch grid, contiguous halves.

Stages 4 and 5 (:func:`inter_chunk_lnc2`)
    The state carry is sequential over chunks, but most of 342e93e's per-chunk work does not
    read the state: the chunk-local cumulative gate, the normalised and gated query, the
    transposes of ``w``, ``qg`` and ``Aqk``, and the per-chunk decay column. Those run once
    per 128-token tile. The chunk loop keeps only what reads the state: ``v_new = u - w @ h``,
    ``h' = h * decay + kg^T @ v_new``, and the output ``o = qg @ h + Aqk @ v_new``, which
    accumulates both products in one PSUM tile. The two programs split the value columns:
    the decay is per key channel, so the state's value columns never mix, and each program
    carries ``[K, V / 2]`` of the state with no exchange between the cores.

Both kernels take :data:`SOURCE_DIGEST` as their last argument, so an edit to this file or to
the ``chunked_recurrence`` helpers they emit through changes the compiled-kernel cache key.

``chunked_recurrence.kda_intra_chunk`` and ``kda_inter_chunk`` stay the entry points: they
validate and count as before, then launch these kernels when :func:`chunked_lnc2_enabled`
(an LNC2 runtime, unless ``VLLM_NEURON_KDA_CHUNKED_LNC2=0``), and 342e93e's one-core
kernels otherwise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda import chunked_recurrence as _chunked
from vllm_neuron.functional.kda.chunked_recurrence import (
    MAX_TILE,
    InterChunkOutputs,
    IntraChunkOutputs,
    _emit_l2_normalise,
    _emit_prepare,
    _emit_row_products,
    _emit_stage3,
    _emit_transpose,
    _psum,
    _sbuf,
    doubling_stages,
)

SOURCE_DIGEST = int(hashlib.sha256(
    Path(__file__).read_bytes() + Path(_chunked.__file__).read_bytes()
).hexdigest()[:7], 16)
"""The content digest of this file and of ``chunked_recurrence.py`` (whose helpers both kernels
emit through), handed to each kernel as a trace-time int: the compiled-kernel cache keys on a
kernel's own source and its arguments, and would otherwise not see a helper edit."""


#: Environment switch: ``0`` routes ``kda_intra_chunk`` and ``kda_inter_chunk`` back to
#: 342e93e's one-core kernels on an LNC2 runtime too.
CHUNKED_LNC2_ENV = "VLLM_NEURON_KDA_CHUNKED_LNC2"


def _lnc2() -> bool:
    return os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2"


def chunked_lnc2_enabled() -> bool:
    """Whether ``chunked_recurrence``'s two entry points launch these kernels: on an LNC2
    runtime, unless :data:`CHUNKED_LNC2_ENV` is ``0``."""
    return _lnc2() and os.environ.get(CHUNKED_LNC2_ENV, "1") != "0"


def intra_programs(n_chunks: int) -> int:
    """Programs stages 1 to 3 launch: both cores of an LNC2 core from two chunks on."""
    return 2 if _lnc2() and int(n_chunks) >= 2 else 1


def inter_programs(vdim: int) -> int:
    """Programs stages 4 and 5 launch: both cores of an LNC2 core when the value columns halve."""
    return 2 if _lnc2() and int(vdim) >= 2 and int(vdim) % 2 == 0 else 1


def chunks_per_tile(chunk: int) -> int:
    """Chunks one 128-partition tile holds."""
    return MAX_TILE // int(chunk)


# ---------------------------------------------------------------------------------------------
# Host-built constants
# ---------------------------------------------------------------------------------------------


def _block_of(rows: Tensor, chunk: int) -> Tensor:
    return torch.div(rows, chunk, rounding_mode="floor")


def grouped_intra_constants(chunk: int, *, device=None, dtype=torch.float32):
    """The ``[T, T]`` block-diagonal forms of 342e93e's ``chunk_constants``, ``T = 128 // C * C``,
    and the ``[T, C]`` column selector that reads the diagonal blocks off a ``[T, T]`` tile.

    * ``triu[s, t] = 1`` for ``s <= t`` in one chunk: as a stationary operand its transpose is
      the chunk-local lower-inclusive ones matrix, so ``triu^T @ gk`` is each chunk's own
      inclusive cumulative gate.
    * ``eye``, the ``T x T`` identity.
    * ``mask_lower[t, j] = 1`` for ``j < t`` in one chunk.
    * ``last_row[s, t] = 1`` for ``s`` the last row of ``t``'s chunk: its transpose repeats each
      chunk's last gate row down that chunk.
    * ``select[j, c] = 1`` for ``j % C == c``: ``X @ select`` is ``X``'s diagonal ``C x C``
      blocks stacked, for a block-diagonal ``X``.
    """
    tile = chunks_per_tile(chunk) * chunk
    idx = torch.arange(tile, device=device)
    rows, cols = idx.unsqueeze(1), idx.unsqueeze(0)
    same = _block_of(rows, chunk) == _block_of(cols, chunk)
    last_of_col = (_block_of(cols, chunk) + 1) * chunk - 1
    return (
        ((rows <= cols) & same).to(dtype),
        torch.eye(tile, device=device, dtype=dtype),
        ((cols < rows) & same).to(dtype),
        (rows == last_of_col).to(dtype).contiguous(),
        ((rows % chunk) == torch.arange(chunk, device=device).unsqueeze(0)).to(dtype),
    )


def grouped_inter_constants(chunk: int, *, device=None, dtype=torch.float32):
    """The block-diagonal cumulative-sum matrix of :func:`grouped_intra_constants` and the
    ``[T, 128 // C]`` last-row selector: column ``b`` picks row ``b * C + C - 1``, so
    ``gc^T @ last_sel`` lands every chunk's last cumulative-gate row as one ``[K, 1]`` column.
    """
    group = chunks_per_tile(chunk)
    tile = group * chunk
    idx = torch.arange(tile, device=device)
    rows, cols = idx.unsqueeze(1), idx.unsqueeze(0)
    same = _block_of(rows, chunk) == _block_of(cols, chunk)
    blocks = torch.arange(group, device=device).unsqueeze(0)
    return (
        ((rows <= cols) & same).to(dtype),
        (rows == blocks * chunk + chunk - 1).to(dtype).contiguous(),
    )


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _load_rows(dst, hbm, row0: int, rows: int, width: int, row_stride: int, col0: int = 0):
    """``rows`` rows of a token-major HBM tensor (``row_stride`` elements a row) into ``dst``."""
    nisa.dma_copy(dst=dst, src=hbm.ap(pattern=[[row_stride, rows], [1, width]],
                                      offset=row0 * row_stride + col0))


def _store_rows(hbm, src, row0: int, rows: int, width: int, row_stride: int, col0: int = 0):
    nisa.dma_copy(dst=hbm.ap(pattern=[[row_stride, rows], [1, width]],
                             offset=row0 * row_stride + col0), src=src)


def _emit_block_unit_lower_inverse(dst, a_sb, eye_sb, rows: int, stages: int):
    """``dst = (I + A)**-1`` for a block-diagonal strictly lower ``A``, ``stages`` doublings.

    ``chunked_recurrence._emit_unit_lower_inverse`` with the tile extent and the stage count
    separated: the tile holds several chunks, the series length is one chunk's.
    """
    n_sb = _sbuf(rows, rows)
    nt_sb = _sbuf(rows, rows)
    s_sb = _sbuf(rows, rows)
    nisa.tensor_scalar(dst=n_sb, data=a_sb, op0=nl.multiply, operand0=-1.0)
    _emit_transpose(nt_sb, n_sb, rows, rows)
    nisa.tensor_copy(dst=s_sb, src=eye_sb)

    ps_x = _psum(rows, rows)
    ps_y = _psum(rows, rows)
    tmp = _sbuf(rows, rows)
    for stage in range(stages):
        nisa.nc_matmul(dst=ps_x, stationary=nt_sb, moving=s_sb, accumulate=False)
        nisa.tensor_copy(dst=tmp, src=ps_x)
        nisa.tensor_tensor(dst=s_sb, data1=s_sb, data2=tmp, op=nl.add)
        if stage < stages - 1:
            nisa.nc_matmul(dst=ps_x, stationary=nt_sb, moving=n_sb, accumulate=False)
            nisa.nc_matmul(dst=ps_y, stationary=n_sb, moving=nt_sb, accumulate=False)
            nisa.tensor_copy(dst=n_sb, src=ps_x)
            nisa.tensor_copy(dst=nt_sb, src=ps_y)
    nisa.tensor_copy(dst=dst, src=s_sb)


def _emit_diagonal_blocks(dst, square_sb, select_sb, rows: int, chunk: int):
    """``dst [rows, C]`` = the diagonal ``C x C`` blocks of the block-diagonal ``square_sb``."""
    square_t = _sbuf(rows, rows)
    _emit_transpose(square_t, square_sb, rows, rows)
    ps = _psum(rows, chunk)
    nisa.nc_matmul(dst=ps, stationary=square_t, moving=select_sb, accumulate=False)
    nisa.tensor_copy(dst=dst, src=ps)


# ---------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------


@nki.jit
def _kda_intra_grouped_nki(q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm, triu_hbm, eye_hbm,
                           mask_lower_hbm, last_row_hbm, select_hbm, source_digest):
    """Stages 1 to 3 for every chunk, ``128 // C`` chunks a tile, chunks split over programs.

    ``q``, ``k`` and ``gk`` are ``[NC, C, K]``; ``v`` is ``[NC, C, V]``; ``beta`` is
    ``[NC, C, 1]``; the constants are :func:`grouped_intra_constants`'. Returns
    342e93e's five outputs in its shapes: ``w``, ``u``, ``kg`` ``[NC, C, *]`` and
    ``a_inv``, ``aqk`` ``[NC, C, C]``.

    The body is ``chunked_recurrence.kda_intra_chunk_kernel``'s with the chunk extent
    replaced by the tile's row count: the same helpers, the same operand order.
    """
    n_chunks, chunk, kdim = q_hbm.shape
    vdim = v_hbm.shape[2]
    scale = float(kdim) ** -0.5
    group = MAX_TILE // chunk
    tile = group * chunk
    stages = doubling_stages(chunk)

    w_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    u_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    kg_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    ainv_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)
    aqk_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    share = (n_chunks + n_prgs - 1) // n_prgs
    lo = prg * share
    hi = min(n_chunks, lo + share)

    triu_sb = _sbuf(tile, tile)
    eye_sb = _sbuf(tile, tile)
    mask_lower_sb = _sbuf(tile, tile)
    last_row_sb = _sbuf(tile, tile)
    select_sb = _sbuf(tile, chunk)
    causal_sb = _sbuf(tile, tile)
    nisa.dma_copy(dst=triu_sb, src=triu_hbm)
    nisa.dma_copy(dst=eye_sb, src=eye_hbm)
    nisa.dma_copy(dst=mask_lower_sb, src=mask_lower_hbm)
    nisa.dma_copy(dst=last_row_sb, src=last_row_hbm)
    nisa.dma_copy(dst=select_sb, src=select_hbm)
    nisa.tensor_tensor(dst=causal_sb, data1=mask_lower_sb, data2=eye_sb, op=nl.add)

    for c0 in range(lo, hi, group):
        nc = min(group, hi - c0)
        rows = nc * chunk
        row0 = c0 * chunk
        triu = triu_sb[0:rows, 0:rows]
        eye = eye_sb[0:rows, 0:rows]

        k_raw = _sbuf(rows, kdim)
        _load_rows(k_raw, k_hbm, row0, rows, kdim, kdim)
        gk_sb = _sbuf(rows, kdim)
        _load_rows(gk_sb, gk_hbm, row0, rows, kdim, kdim)
        k_sb = _sbuf(rows, kdim)
        gc_sb = _sbuf(rows, kdim)
        egc_sb = _sbuf(rows, kdim)
        _emit_prepare(k_sb, gc_sb, egc_sb, k_raw, gk_sb, triu, rows, kdim)

        q_raw = _sbuf(rows, kdim)
        _load_rows(q_raw, q_hbm, row0, rows, kdim, kdim)
        q_sb = _sbuf(rows, kdim)
        _emit_l2_normalise(q_sb, q_raw, rows, kdim)
        nisa.tensor_scalar(dst=q_sb, data=q_sb, op0=nl.multiply, operand0=scale)

        beta_sb = _sbuf(rows, 1)
        _load_rows(beta_sb, beta_hbm, row0, rows, 1, 1)

        neg_gc = _sbuf(rows, kdim)
        emgc_sb = _sbuf(rows, kdim)
        nisa.tensor_scalar(dst=neg_gc, data=gc_sb, op0=nl.multiply, operand0=-1.0)
        nisa.activation(dst=emgc_sb, data=neg_gc, op=nl.exp)

        kp_sb = _sbuf(rows, kdim)
        km_sb = _sbuf(rows, kdim)
        nisa.tensor_tensor(dst=kp_sb, data1=k_sb, data2=egc_sb, op=nl.multiply)
        nisa.tensor_tensor(dst=km_sb, data1=k_sb, data2=emgc_sb, op=nl.multiply)
        km_t_sb = _sbuf(kdim, rows)
        _emit_transpose(km_t_sb, km_sb, rows, kdim)

        kk_sb = _sbuf(rows, rows)
        a_sb = _sbuf(rows, rows)
        _emit_row_products(kk_sb, kp_sb, km_t_sb, rows, kdim)
        nisa.tensor_scalar(dst=a_sb, data=kk_sb, op0=nl.multiply, operand0=beta_sb)
        nisa.tensor_tensor(dst=a_sb, data1=a_sb, data2=mask_lower_sb[0:rows, 0:rows],
                           op=nl.multiply)

        qp_sb = _sbuf(rows, kdim)
        qk_sb = _sbuf(rows, rows)
        aqk_sb = _sbuf(rows, rows)
        nisa.tensor_tensor(dst=qp_sb, data1=q_sb, data2=egc_sb, op=nl.multiply)
        _emit_row_products(qk_sb, qp_sb, km_t_sb, rows, kdim)
        nisa.tensor_tensor(dst=aqk_sb, data1=qk_sb, data2=causal_sb[0:rows, 0:rows],
                           op=nl.multiply)

        a_inv_sb = _sbuf(rows, rows)
        _emit_block_unit_lower_inverse(a_inv_sb, a_sb, eye, rows, stages)

        v_sb = _sbuf(rows, vdim)
        _load_rows(v_sb, v_hbm, row0, rows, vdim, vdim)
        w_sb = _sbuf(rows, kdim)
        u_sb = _sbuf(rows, vdim)
        kg_sb = _sbuf(rows, kdim)
        _emit_stage3(w_sb, u_sb, kg_sb, k_sb, v_sb, beta_sb, gc_sb, egc_sb, a_inv_sb,
                     last_row_sb[0:rows, 0:rows], rows, kdim, vdim)

        ainv_c = _sbuf(rows, chunk)
        _emit_diagonal_blocks(ainv_c, a_inv_sb, select_sb[0:rows, 0:chunk], rows, chunk)
        aqk_c = _sbuf(rows, chunk)
        _emit_diagonal_blocks(aqk_c, aqk_sb, select_sb[0:rows, 0:chunk], rows, chunk)

        _store_rows(w_hbm, w_sb, row0, rows, kdim, kdim)
        _store_rows(u_hbm, u_sb, row0, rows, vdim, vdim)
        _store_rows(kg_hbm, kg_sb, row0, rows, kdim, kdim)
        _store_rows(ainv_hbm, ainv_c, row0, rows, chunk, chunk)
        _store_rows(aqk_hbm, aqk_c, row0, rows, chunk, chunk)

    return w_hbm, u_hbm, kg_hbm, ainv_hbm, aqk_hbm


@nki.jit
def _kda_inter_lnc2_nki(kg_hbm, w_hbm, u_hbm, gk_hbm, q_hbm, aqk_hbm, triu_hbm, last_sel_hbm,
                        state_init_hbm, source_digest):
    """Stages 4 and 5 for every chunk, value columns split over programs.

    Operands as ``chunked_recurrence.kda_inter_chunk_kernel``'s (``state_init`` ``[V, K]``),
    constants :func:`grouped_inter_constants`'. Returns ``o`` and ``v_new`` ``[NC, C, V]`` and
    the final state ``[V, K]``; program ``p`` writes value columns ``p * V / P .. (p + 1) * V / P
    - 1`` of all three.

    The state is carried transposed, ``[K, V / P]``, in two buffers used in turn, so a chunk's
    update writes the buffer the next chunk reads while the output products still read the
    entering one.
    """
    n_chunks, chunk, kdim = kg_hbm.shape
    vdim = u_hbm.shape[2]
    scale = float(kdim) ** -0.5
    group = MAX_TILE // chunk
    tile = group * chunk

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    width = vdim // n_prgs
    v0 = prg * width

    o_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    vnew_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    state_hbm = nl.ndarray((vdim, kdim), dtype=nl.float32, buffer=nl.shared_hbm)

    triu_sb = _sbuf(tile, tile)
    last_sel_sb = _sbuf(tile, group)
    nisa.dma_copy(dst=triu_sb, src=triu_hbm)
    nisa.dma_copy(dst=last_sel_sb, src=last_sel_hbm)

    entering_sb = _sbuf(width, kdim)
    _load_rows(entering_sb, state_init_hbm, v0, width, kdim, kdim)
    ht = [_sbuf(kdim, width), _sbuf(kdim, width)]
    _emit_transpose(ht[0], entering_sb, width, kdim)
    ps_v = [_psum(chunk, width), _psum(chunk, width)]
    ps_h = [_psum(kdim, width), _psum(kdim, width)]
    ps_o = [_psum(chunk, width), _psum(chunk, width)]
    cur = 0

    for c0 in range(0, n_chunks, group):
        nc = min(group, n_chunks - c0)
        rows = nc * chunk
        row0 = c0 * chunk

        # ---- everything that does not read the state, once per tile.
        gk_sb = _sbuf(rows, kdim)
        _load_rows(gk_sb, gk_hbm, row0, rows, kdim, kdim)
        gc_sb = _sbuf(rows, kdim)
        ps_gc = _psum(rows, kdim)
        nisa.nc_matmul(dst=ps_gc, stationary=triu_sb[0:rows, 0:rows], moving=gk_sb,
                       accumulate=False)
        nisa.tensor_copy(dst=gc_sb, src=ps_gc)
        egc_sb = _sbuf(rows, kdim)
        nisa.activation(dst=egc_sb, data=gc_sb, op=nl.exp)

        q_raw = _sbuf(rows, kdim)
        _load_rows(q_raw, q_hbm, row0, rows, kdim, kdim)
        q_sb = _sbuf(rows, kdim)
        _emit_l2_normalise(q_sb, q_raw, rows, kdim)
        nisa.tensor_scalar(dst=q_sb, data=q_sb, op0=nl.multiply, operand0=scale)
        qg_sb = _sbuf(rows, kdim)
        nisa.tensor_tensor(dst=qg_sb, data1=q_sb, data2=egc_sb, op=nl.multiply)
        qg_t = _sbuf(kdim, rows)
        _emit_transpose(qg_t, qg_sb, rows, kdim)

        w_sb = _sbuf(rows, kdim)
        _load_rows(w_sb, w_hbm, row0, rows, kdim, kdim)
        w_t = _sbuf(kdim, rows)
        _emit_transpose(w_t, w_sb, rows, kdim)

        aqk_sb = _sbuf(rows, chunk)
        _load_rows(aqk_sb, aqk_hbm, row0, rows, chunk, chunk)
        aqk_t = _sbuf(chunk, rows)
        _emit_transpose(aqk_t, aqk_sb, rows, chunk)

        # kg and u with the chunk's tokens on partitions 0 .. C - 1 and the chunk on the free
        # axis, so every per-chunk operand below starts at partition 0.
        kg3 = nl.ndarray((chunk, nc, kdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=kg3, src=kg_hbm.ap(
            pattern=[[kdim, chunk], [chunk * kdim, nc], [1, kdim]], offset=row0 * kdim))
        u3 = nl.ndarray((chunk, nc, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=u3, src=u_hbm.ap(
            pattern=[[vdim, chunk], [chunk * vdim, nc], [1, width]], offset=row0 * vdim + v0))

        # Every chunk's state decay exp(gc[C - 1]) as one [K, 1] column of a [K, nc] tile.
        ps_d = _psum(kdim, nc)
        nisa.nc_matmul(dst=ps_d, stationary=gc_sb, moving=last_sel_sb[0:rows, 0:nc],
                       accumulate=False)
        glast = _sbuf(kdim, nc)
        nisa.tensor_copy(dst=glast, src=ps_d)
        decay = _sbuf(kdim, nc)
        nisa.activation(dst=decay, data=glast, op=nl.exp)

        o3 = nl.ndarray((chunk, nc, width), dtype=nl.float32, buffer=nl.sbuf)
        vn3 = nl.ndarray((chunk, nc, width), dtype=nl.float32, buffer=nl.sbuf)

        # ---- the carry, one chunk at a time.
        for b in range(nc):
            h_now = ht[cur]
            h_next = ht[1 - cur]
            cols = slice(b * chunk, (b + 1) * chunk)
            pv = ps_v[b % 2]
            nisa.nc_matmul(dst=pv, stationary=w_t[:, cols], moving=h_now, accumulate=False)
            nisa.tensor_tensor(dst=vn3[:, b, :], data1=u3[:, b, :], data2=pv, op=nl.subtract)
            ph = ps_h[b % 2]
            nisa.nc_matmul(dst=ph, stationary=kg3[:, b, :], moving=vn3[:, b, :],
                           accumulate=False)
            nisa.scalar_tensor_tensor(dst=h_next, data=h_now, op0=nl.multiply,
                                      operand0=decay[:, b:b + 1], op1=nl.add, operand1=ph)
            po = ps_o[b % 2]
            nisa.nc_matmul(dst=po, stationary=qg_t[:, cols], moving=h_now, accumulate=False)
            nisa.nc_matmul(dst=po, stationary=aqk_t[:, cols], moving=vn3[:, b, :],
                           accumulate=True)
            nisa.tensor_copy(dst=o3[:, b, :], src=po)
            cur = 1 - cur

        out_pattern = [[vdim, chunk], [chunk * vdim, nc], [1, width]]
        nisa.dma_copy(dst=o_hbm.ap(pattern=out_pattern, offset=row0 * vdim + v0), src=o3)
        nisa.dma_copy(dst=vnew_hbm.ap(pattern=out_pattern, offset=row0 * vdim + v0), src=vn3)

    state_sb = _sbuf(width, kdim)
    _emit_transpose(state_sb, ht[cur], kdim, width)
    _store_rows(state_hbm, state_sb, v0, width, kdim, kdim)
    return o_hbm, state_hbm, vnew_hbm


# ---------------------------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------------------------


def intra_chunk_lnc2(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
                     ) -> IntraChunkOutputs:
    """Launch stages 1 to 3. Operands as ``chunked_recurrence.kda_intra_chunk`` takes them,
    already validated there (this launcher checks nothing and counts nothing)."""
    n_chunks, chunk = int(q.shape[0]), int(q.shape[1])
    triu, eye, mask_lower, last_row, select = grouped_intra_constants(
        chunk, device=q.device, dtype=torch.float32)
    call = wrap_nki(_kda_intra_grouped_nki)
    if intra_programs(n_chunks) == 2:
        call = call[2]
    w, u, kg, a_inv, aqk = call(
        q.float().contiguous(), k.float().contiguous(), v.float().contiguous(),
        beta.float().unsqueeze(-1).contiguous(), gk.float().contiguous(),
        triu, eye, mask_lower, last_row, select, SOURCE_DIGEST)
    return IntraChunkOutputs(w=w, u=u, kg=kg, a_inv=a_inv, aqk=aqk)


def inter_chunk_lnc2(kg: Tensor, w: Tensor, u: Tensor, gk: Tensor, q: Tensor, aqk: Tensor,
                     state: Tensor | None = None) -> InterChunkOutputs:
    """Launch stages 4 and 5. Operands as ``chunked_recurrence.kda_inter_chunk`` takes them,
    already validated there; ``state`` is the ``[V, K]`` entering state or ``None`` for zero."""
    chunk, kdim, vdim = int(kg.shape[1]), int(kg.shape[2]), int(u.shape[2])
    triu, last_sel = grouped_inter_constants(chunk, device=kg.device, dtype=torch.float32)
    entering = (torch.zeros(vdim, kdim, device=kg.device, dtype=torch.float32)
                if state is None else state.float().contiguous())
    call = wrap_nki(_kda_inter_lnc2_nki)
    if inter_programs(vdim) == 2:
        call = call[2]
    o, final_state, v_new = call(
        kg.float().contiguous(), w.float().contiguous(), u.float().contiguous(),
        gk.float().contiguous(), q.float().contiguous(), aqk.float().contiguous(),
        triu, last_sel, entering, SOURCE_DIGEST)
    return InterChunkOutputs(o=o, final_state=final_state, v_new=v_new)
