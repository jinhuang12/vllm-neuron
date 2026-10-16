# SPDX-License-Identifier: Apache-2.0
"""The exact path of the block-shared sparse MLA prefill kernel: one block by today's per-query gather.

The block-shared kernel (``mla_block_shared.py``) gathers one union of rows per block of ``block_queries``
queries. A block whose union does not fit the kernel's cap (its overflow flag, ``totals[b, 1]``, is 1) must not
read its cut union. Its queries take this path instead: the sparse kernel's low-precision bank-form body
(``mla_sparse._attention_body_row_tiled_lowp``) over the block's own index rows, which gathers each query's
selected rows one query at a time. The output bits are the sparse kernel's for every query of the block.

Bit-equality. Each query passes the instructions the sparse body runs for it, in the same order, on the same
operands: the same helpers (``mla_sparse``'s offsets, query staging, MM2 and bank table), the same query blocks of
``mla_sparse._queries_per_block`` rows, the same softmax groups of ``mla_sparse.SOFTMAX_ROWS`` queries with each
query in the same slot, and the same output transposes. The schedule is the sparse body's too: the same rings of
gathered-row buffers and offsets, the offsets computed the same groups ahead and the next query block's rows
loaded and staged one block ahead, so the software queue runs one query's gather after another while a group
finishes. Only where the rotation starts changes: the sparse body rotates over every query of its program, this
path over the queries of its one block. Neither changes a value. The sparse body's non-finite rule stays too: a
non-finite element in a row one query of a softmax group gathers reaches every query of that group, as
``0 * nan`` in MM1.

Branching on the flag. The flag picks the body on the device through a dynamic loop whose trip count is 0 or 1
(:func:`exact_trip_count`): ``nl.fori_loop(0, trips, body)``. Under LNC 2 both cores must run every dynamic loop
the same number of times (each of its basic blocks ends on a barrier of both cores), so the count is that of the
step, the blocks the cores run side by side, not of one block: a step with a flagged block runs this path for
every block of the step. The path is exact for any block, so that changes no answer. Every basic block also ends
on a drain of every engine, and the compiler schedules each block on its own, so the loops belong after the
program's last block-shared block: there the block-shared work keeps its overlap, and this path overwrites the
rows of the blocks it runs.
"""
from __future__ import annotations

import nki.isa as nisa
import nki.language as nl

from vllm_neuron.functional.attention import mla_sparse as _ms

#: Column of the union producer's ``totals`` that holds a block's overflow flag (``mla_block_shared_union``).
OVERFLOW_COLUMN = 1
#: int32 or fp32 elements in one 32-byte SBUF line: the width of the flag tiles.
_LINE = 8


def exact_trip_count(totals_hbm, first_block: int, blocks: int):
    """A register holding 1 when a block of the step has overflowed its cap, else 0.

    Args:
        totals_hbm: ``[S / block_queries, 2]`` int32, the union producer's totals; column
            :data:`OVERFLOW_COLUMN` is 1 where the block's union overflowed its cap.
        first_block: python int, the step's first block.
        blocks: python int, the blocks of the step: the programs of the launch (``nl.num_programs``), so every
            core of an LNC 2 pair reads the same flags and gets the same count. The chunk's blocks must be a whole
            number of steps, so that every core runs the same steps.

    The flags are read from HBM by one hardware-queue DMA and their maximum is taken in fp32 (exact for 0 and 1),
    then written as int32 and loaded into the register. A ``tensor_scalar`` over an int32 source lowers to the
    integer form, which takes no float operand, so the conversion reads the fp32 maximum.
    """
    width = totals_hbm.shape[1]
    raw = nl.ndarray((1, _LINE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw[0:1, 0:blocks],
                  src=totals_hbm.ap(pattern=[[width * blocks, 1], [width, blocks]],
                                    offset=first_block * width + OVERFLOW_COLUMN),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    flag = nl.ndarray((1, _LINE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=flag[0:1, 0:1], op=nl.maximum, data=raw[0:1, 0:blocks], axis=1)
    count = nl.ndarray((1, _LINE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=count[0:1, 0:1], data=flag[0:1, 0:1], op0=nl.multiply, operand0=1.0,
                       engine=nisa.engine.vector)
    trips = nisa.register_alloc()
    nisa.register_load(dst=trips, src=count[0:1, 0:1])
    return trips


def exact_block(q_lift_hbm, bank_hbm, topk_hbm, bank_rows, softmax_scale, page_size: int, block: int,
                block_queries: int, out_hbm):
    """Attention of block ``block``'s queries over their own index rows, as the sparse kernel computes it.

    Args:
        q_lift_hbm: ``[S, 1, L]`` bf16 or fp16, the absorbed queries; ``L`` a multiple of 128 within one MM2
            moving tile (:data:`mla_sparse.MOVING_MAX`).
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_lift_hbm``, already holding this step's
            rows (the sparse kernel's bank form).
        topk_hbm: ``[S, K]`` int32, each query's selected window rows, ``-1`` for a column with no token; ``K`` a
            multiple of 128. The sparse kernel's index rows at this call site.
        bank_rows: ``[128, pages]`` uint32, ``mla_sparse._lowp_bank_table(block_table, page_size)``: the bank row
            of each window page's first row on every partition. The caller builds it once per program.
        softmax_scale: python float, positive.
        page_size: python int, rows per page, a power of two.
        block: python int, the block: queries ``block * block_queries .. (block + 1) * block_queries - 1``.
        block_queries: python int, queries per block, a multiple of the sparse body's query block.
        out_hbm: ``[S, 1, L]`` fp32 in shared HBM, the block-shared kernel's output. This writes the block's rows,
            ``block_queries`` rows from row ``block * block_queries``, row-major: the rows the block-shared body
            writes for the block, in the same layout.

    The block runs on the calling program alone, with the sparse body's working set: its ring of gathered-row
    buffers, ``SOFTMAX_ROWS + mla_sparse.QUERY_BUFFERS_DEFAULT`` rounded up to whole groups, of ``(L / 128) * K``
    2-byte elements per partition (8 buffers, 136 KiB at K = 2,176, L = 512), plus the group's five
    ``[SOFTMAX_ROWS, K]`` 4-byte softmax rows (42.5 KiB per partition at K = 2,176).
    """
    seq, heads, latent = q_lift_hbm.shape
    topk = topk_hbm.shape[1]
    n_latent = latent // _ms.LATENT_TILE
    n_chunks = topk // _ms.KEY_CHUNK
    tiles = _ms._score_tiles(topk, _ms.MOVING_MAX)
    qpb = _ms._queries_per_block(seq, heads)
    rows_per_block = qpb * heads
    group = _ms._softmax_rows(qpb)
    dtype = bank_hbm.dtype
    first = block * block_queries
    query_blocks = block_queries // qpb

    # Two query blocks' rows and their tiles with the latent on partitions (column li * rows_per_block + q): query
    # block j uses set j % 2; its rows are copied two query blocks ahead and staged one ahead.
    q_rows = []
    q_stage = []
    for _ in range(2):
        q_rows.append(nl.ndarray((rows_per_block, latent), dtype=q_lift_hbm.dtype, buffer=nl.sbuf))
        q_stage.append(nl.ndarray((_ms.LATENT_TILE, _ms._aligned(n_latent * rows_per_block, _ms.STAGE_ALIGN)),
                                  dtype=q_lift_hbm.dtype, buffer=nl.sbuf))
    # MM1's stationaries: slot r carries query r of a group in column r and zeros elsewhere for the whole block.
    q_masked = []
    for _ in range(group):
        slot = nl.ndarray((_ms.LATENT_TILE, n_latent, _ms._aligned(group, _ms.STAGE_ALIGN)), dtype=dtype,
                          buffer=nl.sbuf)
        nisa.memset(dst=slot, value=0.0, engine=nisa.engine.vector)
        q_masked.append(slot)
    # The sparse body's rings. The block's n-th query gathers into rows_t[n % ring], so its gather waits for the
    # MM2 of query n - ring, which comes before the previous group's softmax. Its offsets are in
    # scratch[n % index_ring], computed after the MM1 of query n - ahead's group.
    ring = _ms._aligned(group + _ms.QUERY_BUFFERS_DEFAULT, group)
    ahead = _ms.OFFSET_GROUPS_AHEAD * group
    index_ring = ahead + group
    rows_t = []
    for _ in range(ring):
        rows_t.append(nl.ndarray((_ms.LATENT_TILE, 1, n_latent, topk), dtype=dtype, buffer=nl.sbuf))
    scratch = []
    for _ in range(index_ring):
        scratch.append(_ms._lowp_scratch(n_chunks, True))
    # MM2's two one-chunk stationaries (the bank form's order).
    c_pair = []
    for _ in range(2):
        c_pair.append(nl.ndarray((_ms.KEY_CHUNK, latent), dtype=dtype, buffer=nl.sbuf))
    # The group's softmax rows, one query per partition, reused along the chain as the sparse body reuses them.
    idx_rows = _ms._sbuf_i32(group, topk)
    valid_f = _ms._sbuf(group, topk)
    mask_bias = _ms._sbuf(group, topk)
    scores_m = _ms._sbuf(group, topk)
    p = _ms._sbuf(group, topk)
    neg_row_max = _ms._scalar(group)
    exp_bias = _ms._scalar(group)
    row_sum = _ms._scalar(group)
    recip = _ms._scalar(group)
    p_cols = n_chunks * group
    p_t = _ms._sbuf(_ms.KEY_CHUNK, _ms._aligned(p_cols))
    p_hi_f = _ms._sbuf(_ms.KEY_CHUNK, _ms._aligned(p_cols))
    p_hl = nl.ndarray((_ms.KEY_CHUNK, _ms._aligned(p_cols), 2), dtype=dtype, buffer=nl.sbuf)
    out_t = _ms._sbuf(_ms.LATENT_TILE, n_latent, _ms._aligned(qpb))
    out_rows = _ms._sbuf(qpb, latent)
    sentinel_bias = _ms._SENTINEL_EXP_FLOOR / softmax_scale

    # The first query block's rows and tiles, the second's rows, and the first `ahead` queries' offsets.
    _ms._lowp_load_queries(q_rows[0], q_lift_hbm, first, rows_per_block)
    if query_blocks > 1:
        _ms._lowp_load_queries(q_rows[1], q_lift_hbm, first + qpb, rows_per_block)
    _ms._lowp_stage_queries(q_stage[0], q_rows[0], n_latent, rows_per_block, q_lift_hbm.dtype)
    for m in range(min(ahead, block_queries)):
        _ms._lowp_offsets(scratch[m % index_ring], topk_hbm, first + m, bank_rows, page_size)

    for nb in range(query_blocks):
        n0 = nb * qpb
        q0 = first + n0
        q_tiles = q_stage[nb % 2]
        for g0 in range(0, qpb, group):
            scores_ps = []
            for ti in range(len(tiles)):
                scores_ps.append(_ms._psum(group, tiles[ti][1]))
            for r in range(group):
                qi = g0 + r
                c_t = rows_t[(n0 + qi) % ring]
                # The query's selected rows with the latent on partitions (the sparse body's gather).
                nisa.dma_transpose(
                    dst=c_t,
                    src=bank_hbm.ap(pattern=[[latent, topk], [0, 1], [_ms.LATENT_TILE, n_latent],
                                             [1, _ms.LATENT_TILE]],
                                    vector_offset=scratch[(n0 + qi) % index_ring][_ms._LP_OFFSET][:, 0:n_chunks],
                                    indirect_dim=0),
                    dge_mode=nisa.dge_mode.swdge)
                # MM1 per score tile into the group's PSUM rows.
                for li in range(n_latent):
                    col = li * rows_per_block + qi
                    nisa.tensor_copy(dst=q_masked[r][:, li, r:r + 1], src=q_tiles[:, col:col + 1],
                                     engine=nisa.engine.vector)
                for ti in range(len(tiles)):
                    ks = tiles[ti][0]
                    extent = tiles[ti][1]
                    for li in range(n_latent):
                        nisa.nc_matmul(dst=scores_ps[ti], stationary=q_masked[r][:, li, 0:group],
                                       moving=c_t[:, 0, li, ks:ks + extent], accumulate=(r > 0 or li > 0))
            # The offsets of the group `ahead` queries on, while this group finishes.
            for r in range(group):
                m = n0 + g0 + ahead + r
                if m < block_queries:
                    _ms._lowp_offsets(scratch[m % index_ring], topk_hbm, first + m, bank_rows, page_size)
            # After the query block's first MM1: the next query block's tiles from the rows copied a block ago,
            # then the rows of the query block after it into the set those tiles' block read.
            if g0 == 0 and nb + 1 < query_blocks:
                _ms._lowp_stage_queries(q_stage[(nb + 1) % 2], q_rows[(nb + 1) % 2], n_latent, rows_per_block,
                                        q_lift_hbm.dtype)
                if nb + 2 < query_blocks:
                    _ms._lowp_load_queries(q_rows[nb % 2], q_lift_hbm, q0 + 2 * qpb, rows_per_block)

            # The group's mask from its index rows: 1.0 where a column carries a token, its bias 0.0 there and
            # -sentinel_bias elsewhere.
            nisa.dma_copy(dst=idx_rows, src=topk_hbm.ap(pattern=[[topk, group], [1, topk]],
                                                        offset=(q0 + g0) * topk))
            nisa.tensor_scalar(dst=valid_f, data=idx_rows, op0=nl.greater_equal, operand0=0.0,
                               engine=nisa.engine.vector)
            nisa.tensor_scalar(dst=mask_bias, data=valid_f, op0=nl.add, operand0=-1.0,
                               engine=nisa.engine.scalar)
            nisa.tensor_scalar(dst=mask_bias, data=mask_bias, op0=nl.multiply, operand0=sentinel_bias,
                               engine=nisa.engine.scalar)
            for ti in range(len(tiles)):
                ks = tiles[ti][0]
                extent = tiles[ti][1]
                nisa.tensor_tensor(dst=scores_m[:, ks:ks + extent], data1=scores_ps[ti],
                                   data2=mask_bias[:, ks:ks + extent], op=nl.add)

            # Softmax over each whole row, normalised before MM2.
            nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_m, axis=1, negate=True)
            nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply, operand0=softmax_scale,
                               engine=nisa.engine.vector)
            nisa.activation(dst=p, op=nl.exp, data=scores_m, bias=exp_bias, scale=softmax_scale,
                            reduce_op=nl.add, reduce_res=row_sum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.tensor_tensor(dst=scores_m, data1=p, data2=valid_f, op=nl.multiply, engine=nisa.engine.vector)
            nisa.reciprocal(dst=recip, data=row_sum)
            nisa.tensor_scalar(dst=p, data=scores_m, op0=nl.multiply, operand0=recip, engine=nisa.engine.vector)

            # p onto the key partitions (one PE transpose per chunk), then its bf16 hi and lo halves.
            p_t_ps = _ms._psum(_ms.KEY_CHUNK, p_cols)
            for ck in range(n_chunks):
                nisa.nc_transpose(dst=p_t_ps[:, ck * group:(ck + 1) * group],
                                  data=p[:, ck * _ms.KEY_CHUNK:(ck + 1) * _ms.KEY_CHUNK],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=p_t[:, 0:p_cols], src=p_t_ps, engine=nisa.engine.vector)
            nisa.tensor_copy(dst=p_hl[:, 0:p_cols, 0], src=p_t[:, 0:p_cols], engine=nisa.engine.vector)
            nisa.tensor_copy(dst=p_hi_f[:, 0:p_cols], src=p_hl[:, 0:p_cols, 0], engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=p_hl[:, 0:p_cols, 1], data1=p_t[:, 0:p_cols], data2=p_hi_f[:, 0:p_cols],
                               op=nl.subtract, engine=nisa.engine.vector)

            # MM2 per query, a chunk at a time, latent on partitions.
            for r in range(group):
                _ms._lowp_mm2_per_chunk(out_t, c_pair, rows_t[(n0 + g0 + r) % ring], p_hl, g0 + r, r, group,
                                        n_chunks, n_latent, latent, dtype)

        # The query block's rows: one PE transpose per latent tile, one store.
        rows_out_ps = _ms._psum(qpb, latent)
        for li in range(n_latent):
            nisa.nc_transpose(dst=rows_out_ps[:, li * _ms.LATENT_TILE:(li + 1) * _ms.LATENT_TILE],
                              data=out_t[:, li, 0:qpb], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=out_rows, src=rows_out_ps, engine=nisa.engine.vector)
        nl.store(out_hbm.ap(pattern=[[heads * latent, qpb], [1, latent]], offset=q0 * heads * latent),
                 value=out_rows)
