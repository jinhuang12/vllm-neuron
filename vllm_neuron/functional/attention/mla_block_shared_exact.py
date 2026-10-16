# SPDX-License-Identifier: Apache-2.0
"""The exact path of the block-shared sparse MLA prefill attention: overflowed blocks by today's per-query gather.

The block-shared kernel (``mla_block_shared.py``) gathers one union of rows per block of ``block_queries``
queries. A block whose union does not fit the kernel's cap (the overflow flag ``totals[r, 1]`` of one of its rows
of 128 queries is 1) must not keep the rows it computed from its cut union. :func:`mla_block_shared_exact` runs
after the block-shared kernel and overwrites those blocks' rows of its output with the sparse kernel's
low-precision bank-form body (``mla_sparse._attention_body_row_tiled_lowp``) over the blocks' own index rows,
which gathers each query's selected rows one query at a time. The output bits are the sparse kernel's for every
query of such a block; every other row of the output is left as the block-shared kernel wrote it.

Bit-equality. Each query passes the instructions the sparse body runs for it, in the same order, on the same
operands: the same helpers (``mla_sparse``'s offsets and query loads; this module's copies of its bank table,
query staging and MM2, which differ only in where their tiles live), the same query blocks of
``mla_sparse._queries_per_block`` rows, the same softmax groups of ``mla_sparse.SOFTMAX_ROWS`` queries with each
query in the same slot, and the same output transposes. The schedule is the sparse body's too: the same rings of
gathered-row buffers and offsets, the offsets computed the same groups ahead and the next query block's rows
loaded and staged one block ahead, so the software queue runs one query's gather after another while a group
finishes. Only where the rotation starts changes: the sparse body rotates over every query of its program, this
path over the queries of its one block. Neither changes a value. The sparse body's non-finite rule stays too: a
non-finite element in a row one query of a softmax group gathers reaches every query of that group, as
``0 * nan`` in MM1.

Branching on the flag. The kernel takes the blocks in rounds of one block per program and runs each round in a
dynamic loop whose trip count is 0 or 1: ``nl.fori_loop(0, trips, body)``. Under LNC 2 both cores must run every
dynamic loop the same number of times (each of its basic blocks ends on a barrier of both cores), so the count is
the largest flag of the round, not of one block: a round with a flagged block runs this path for every block of
the round. The path is exact for any block, so that changes no answer. An untaken round costs its loop's entry and
exit.

Fixed addresses. Every SBUF tile comes from one ``SbufManager`` stack and every PSUM tile has a fixed bank.
neuronx-cc inlines a kernel with fixed addresses after its allocator and one without before it. In one graph with
a fixed-address kernel that has a device loop (the MoE kernels), a later device-loop kernel without fixed
addresses is inlined first, and the compile fails (NCC_IBIR735 with the verifier on, NCC_IGCA024 without it).
With fixed addresses both are inlined in one pass, in graph order. The PSUM banks follow the compiler's own
placement of the sparse body: MM1's score tiles on the first banks, and each query's MM2 turning its chunks
through one set of four banks while its sums sit in the other set, the sets alternating by query.
"""
from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest

#: Keys the compiled-kernel cache on this file: the cache keys a kernel by its own source, not its helpers'.
SOURCE_DIGEST = source_digest(__file__)
#: Queries per row of the union producer's ``totals`` (``mla_block_shared.mla_block_shared_union_kernel``).
TOTALS_ROW_QUERIES = 128
#: Column of ``totals`` that holds the overflow flag.
OVERFLOW_COLUMN = 1
#: int32 or fp32 elements in one 32-byte SBUF line: the width of the flag tiles and the most ``totals`` rows a
#: round reads.
_LINE = 8
#: Bytes every SBUF tile starts on: a whole 32-byte line.
_TILE_ALIGN = 32
#: PSUM banks. MM1's score tiles take the first banks, one tile per bank. The query tiles' staging and the
#: probability transposes take one bank each. Each query's MM2 turns its chunks through one set of four banks
#: and sums its latent tiles in the other set; the sets alternate by query. The output transposes reuse the
#: probability bank, which the query block's last group has read by then.
_SCORE_BANKS = 5
_STAGE_BANK = 5
_PROB_BANK = 6
_OUT_BANK = 6
_MM2_SETS = ((7, 0, 1, 2), (3, 4, 5, 6))
_MM2_BANKS = 4


class MlaBlockSharedExactError(ValueError):
    """A malformed call of :func:`mla_block_shared_exact`."""


def _tile(sbm, shape, dtype):
    """A tile of ``shape`` from ``sbm``'s current scope, on a whole SBUF line."""
    return sbm.alloc_stack(shape, dtype=dtype, align=_TILE_ALIGN)


def _scalar(sbm, parts: int):
    """A ``[parts, 1]`` float32 view of a tile whose row is one whole 32-byte line (``mla_sparse._scalar``)."""
    return _tile(sbm, (parts, _ms.DMA_TRANSPOSE_ALIGN), nl.float32)[:, 0:1]


def _bank(shape, dtype, bank: int):
    """A PSUM tile of ``shape`` at the start of PSUM bank ``bank``."""
    return nl.ndarray(shape, dtype=dtype, buffer=nl.psum, address=(0, bank * nl.tile_size.psum_bank_fmax_bytes))


def _bank_table(sbm, table_hbm, page_size: int):
    """``mla_sparse._lowp_bank_table`` with its tiles from ``sbm``: ``[128, pages]`` uint32 bank rows.

    The int32 copy of the table is freed once the bank rows are written.
    """
    pages = table_hbm.shape[0]
    firsts = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(pages)), nl.uint32)
    sbm.open_scope(name="bank_table")
    raw = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(pages)), nl.int32)
    nisa.dma_copy(dst=raw[:, 0:pages],
                  src=table_hbm.ap(pattern=[[0, _ms.KEY_CHUNK], [1, pages]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    nisa.tensor_scalar(dst=firsts[:, 0:pages], data=raw[:, 0:pages], op0=nl.maximum,
                       operand0=0.0, op1=nl.multiply, operand1=float(page_size),
                       engine=nisa.engine.vector)
    sbm.close_scope()
    return firsts[:, 0:pages]


def _offset_scratch(sbm, n_chunks: int):
    """``mla_sparse._lowp_scratch(n_chunks, True)`` with its tiles from ``sbm``: one query's index row and
    gather offsets, positions as ``mla_sparse._LP_*``."""
    ws = []
    for _ in range(_ms._LP_COUNT):
        ws.append(None)
    ws[_ms._LP_RAW] = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(n_chunks)), nl.int32)
    ws[_ms._LP_FILL] = _scalar(sbm, _ms.KEY_CHUNK)
    ws[_ms._LP_SHIFT] = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(n_chunks)), nl.int32)
    ws[_ms._LP_OFFSET] = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(n_chunks)), nl.uint32)
    ws[_ms._LP_PAGES] = _tile(sbm, (_ms.KEY_CHUNK, 3 * _ms._aligned(n_chunks)), nl.uint32)
    return ws


def _stage_queries(dst, rows, n_latent: int, block: int, dtype):
    """``mla_sparse._lowp_stage_queries`` through PSUM bank :data:`_STAGE_BANK`."""
    rows_t_ps = _bank((_ms.LATENT_TILE, n_latent * block), dtype, _STAGE_BANK)
    for li in range(n_latent):
        nisa.nc_transpose(dst=rows_t_ps[:, li * block:(li + 1) * block],
                          data=rows[0:block, li * _ms.LATENT_TILE:(li + 1) * _ms.LATENT_TILE],
                          engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=dst[:, 0:n_latent * block], src=rows_t_ps, engine=nisa.engine.vector)


def _mm2_per_chunk(out_t, c_pair, c_t, p_hl, qi: int, r: int, group: int, n_chunks: int, n_latent: int,
                   latent: int, dtype, turn_banks, sum_banks):
    """``mla_sparse._lowp_mm2_per_chunk``, its chunk ``ck`` turned in bank ``turn_banks[ck % 4]`` and its latent
    tile ``li`` summed in bank ``sum_banks[li]``."""
    pv_ps = []
    for li in range(n_latent):
        pv_ps.append(_bank((_ms.LATENT_TILE, 2), nl.float32, sum_banks[li]))
    for ck in range(n_chunks):
        rows_ps = _bank((_ms.KEY_CHUNK, latent), dtype, turn_banks[ck % _MM2_BANKS])
        for li in range(n_latent):
            nisa.nc_transpose(
                dst=rows_ps[:, li * _ms.LATENT_TILE:(li + 1) * _ms.LATENT_TILE],
                data=c_t[:, 0, li, ck * _ms.KEY_CHUNK:(ck + 1) * _ms.KEY_CHUNK],
                engine=nisa.engine.tensor)
        chunk = c_pair[ck % 2]
        nisa.tensor_copy(dst=chunk, src=rows_ps,
                         engine=nisa.engine.vector if ck % 2 == 0 else nisa.engine.scalar)
        col = ck * group + r
        for li in range(n_latent):
            nisa.nc_matmul(dst=pv_ps[li],
                           stationary=chunk[:, li * _ms.LATENT_TILE:(li + 1) * _ms.LATENT_TILE],
                           moving=p_hl[:, col, 0:2], accumulate=(ck > 0))
    for li in range(n_latent):
        nisa.tensor_reduce(dst=out_t[:, li, qi:qi + 1], op=nl.add, data=pv_ps[li], axis=1)


def _trip_count(sbm, totals_hbm, first_row: int, rows: int):
    """A register holding 1 when a block of the round has overflowed its cap, else 0.

    ``rows`` rows of ``totals_hbm`` from ``first_row``: every row of every block the round's programs run, so
    every core of an LNC 2 pair reads the same flags and gets the same count. The flags are read by one
    hardware-queue DMA and their maximum is taken in fp32 (exact for 0 and 1), then written as int32 and loaded
    into the register. A ``tensor_scalar`` over an int32 source lowers to the integer form, which takes no float
    operand, so the conversion reads the fp32 maximum.
    """
    width = totals_hbm.shape[1]
    raw = _tile(sbm, (1, _LINE), nl.int32)
    nisa.dma_copy(dst=raw[0:1, 0:rows],
                  src=totals_hbm.ap(pattern=[[width * rows, 1], [width, rows]],
                                    offset=first_row * width + OVERFLOW_COLUMN),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    flag = _tile(sbm, (1, _LINE), nl.float32)
    nisa.tensor_reduce(dst=flag[0:1, 0:1], op=nl.maximum, data=raw[0:1, 0:rows], axis=1)
    count = _tile(sbm, (1, _LINE), nl.int32)
    nisa.tensor_scalar(dst=count[0:1, 0:1], data=flag[0:1, 0:1], op0=nl.multiply, operand0=1.0,
                       engine=nisa.engine.vector)
    trips = nisa.register_alloc()
    nisa.register_load(dst=trips, src=count[0:1, 0:1])
    return trips


def _exact_block(sbm, q_lift_hbm, bank_hbm, topk_hbm, bank_rows, softmax_scale, page_size: int, block: int,
                 block_queries: int, out_hbm):
    """Attention of block ``block``'s queries over their own index rows, as the sparse kernel computes it.

    Writes the block's ``block_queries`` rows of ``out_hbm`` from row ``block * block_queries``, row-major. The
    working set is the sparse body's, in one scope of ``sbm`` freed at the end: its ring of gathered-row buffers,
    ``SOFTMAX_ROWS + mla_sparse.QUERY_BUFFERS_DEFAULT`` rounded up to whole groups, of ``(L / 128) * K`` 2-byte
    elements per partition (8 buffers, 136 KiB at K = 2,176, L = 512), the group's five ``[SOFTMAX_ROWS, K]``
    4-byte softmax rows (42.5 KiB per partition at K = 2,176), and the smaller tiles below.
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
    kernel_assert(block_queries % qpb == 0, "a block is whole query blocks of the sparse body")
    kernel_assert(len(tiles) <= _SCORE_BANKS and n_latent <= _MM2_BANKS, "the PSUM banks hold MM1 and MM2")
    first = block * block_queries
    query_blocks = block_queries // qpb
    sbm.open_scope(name="exact_block")

    # Two query blocks' rows and their tiles with the latent on partitions (column li * rows_per_block + q): query
    # block j uses set j % 2; its rows are copied two query blocks ahead and staged one ahead.
    q_rows = []
    q_stage = []
    for _ in range(2):
        q_rows.append(_tile(sbm, (rows_per_block, latent), q_lift_hbm.dtype))
        q_stage.append(_tile(sbm, (_ms.LATENT_TILE, _ms._aligned(n_latent * rows_per_block, _ms.STAGE_ALIGN)),
                             q_lift_hbm.dtype))
    # MM1's stationaries: slot r carries query r of a group in column r and zeros elsewhere for the whole block.
    q_masked = []
    for _ in range(group):
        slot = _tile(sbm, (_ms.LATENT_TILE, n_latent, _ms._aligned(group, _ms.STAGE_ALIGN)), dtype)
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
        rows_t.append(_tile(sbm, (_ms.LATENT_TILE, 1, n_latent, topk), dtype))
    scratch = []
    for _ in range(index_ring):
        scratch.append(_offset_scratch(sbm, n_chunks))
    # MM2's two one-chunk stationaries (the bank form's order).
    c_pair = []
    for _ in range(2):
        c_pair.append(_tile(sbm, (_ms.KEY_CHUNK, latent), dtype))
    # The group's softmax rows, one query per partition, reused along the chain as the sparse body reuses them.
    idx_rows = _tile(sbm, (group, topk), nl.int32)
    valid_f = _tile(sbm, (group, topk), nl.float32)
    mask_bias = _tile(sbm, (group, topk), nl.float32)
    scores_m = _tile(sbm, (group, topk), nl.float32)
    p = _tile(sbm, (group, topk), nl.float32)
    neg_row_max = _scalar(sbm, group)
    exp_bias = _scalar(sbm, group)
    row_sum = _scalar(sbm, group)
    recip = _scalar(sbm, group)
    p_cols = n_chunks * group
    p_t = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(p_cols)), nl.float32)
    p_hi_f = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(p_cols)), nl.float32)
    p_hl = _tile(sbm, (_ms.KEY_CHUNK, _ms._aligned(p_cols), 2), dtype)
    out_t = _tile(sbm, (_ms.LATENT_TILE, n_latent, _ms._aligned(qpb)), nl.float32)
    out_rows = _tile(sbm, (qpb, latent), nl.float32)
    sentinel_bias = _ms._SENTINEL_EXP_FLOOR / softmax_scale

    # The first query block's rows and tiles, the second's rows, and the first `ahead` queries' offsets.
    _ms._lowp_load_queries(q_rows[0], q_lift_hbm, first, rows_per_block)
    if query_blocks > 1:
        _ms._lowp_load_queries(q_rows[1], q_lift_hbm, first + qpb, rows_per_block)
    _stage_queries(q_stage[0], q_rows[0], n_latent, rows_per_block, q_lift_hbm.dtype)
    for m in range(min(ahead, block_queries)):
        _ms._lowp_offsets(scratch[m % index_ring], topk_hbm, first + m, bank_rows, page_size)

    for nb in range(query_blocks):
        n0 = nb * qpb
        q0 = first + n0
        q_tiles = q_stage[nb % 2]
        for g0 in range(0, qpb, group):
            scores_ps = []
            for ti in range(len(tiles)):
                scores_ps.append(_bank((group, tiles[ti][1]), nl.float32, ti))
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
                _stage_queries(q_stage[(nb + 1) % 2], q_rows[(nb + 1) % 2], n_latent, rows_per_block,
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
            p_t_ps = _bank((_ms.KEY_CHUNK, p_cols), nl.float32, _PROB_BANK)
            for ck in range(n_chunks):
                nisa.nc_transpose(dst=p_t_ps[:, ck * group:(ck + 1) * group],
                                  data=p[:, ck * _ms.KEY_CHUNK:(ck + 1) * _ms.KEY_CHUNK],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=p_t[:, 0:p_cols], src=p_t_ps, engine=nisa.engine.vector)
            nisa.tensor_copy(dst=p_hl[:, 0:p_cols, 0], src=p_t[:, 0:p_cols], engine=nisa.engine.vector)
            nisa.tensor_copy(dst=p_hi_f[:, 0:p_cols], src=p_hl[:, 0:p_cols, 0], engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=p_hl[:, 0:p_cols, 1], data1=p_t[:, 0:p_cols], data2=p_hi_f[:, 0:p_cols],
                               op=nl.subtract, engine=nisa.engine.vector)

            # MM2 per query, a chunk at a time, latent on partitions; the bank sets alternate by query.
            for r in range(group):
                j = n0 + g0 + r
                _mm2_per_chunk(out_t, c_pair, rows_t[j % ring], p_hl, g0 + r, r, group, n_chunks, n_latent,
                               latent, dtype, _MM2_SETS[j % 2], _MM2_SETS[(j + 1) % 2])

        # The query block's rows: one PE transpose per latent tile, one store.
        rows_out_ps = _bank((qpb, latent), nl.float32, _OUT_BANK)
        for li in range(n_latent):
            nisa.nc_transpose(dst=rows_out_ps[:, li * _ms.LATENT_TILE:(li + 1) * _ms.LATENT_TILE],
                              data=out_t[:, li, 0:qpb], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=out_rows, src=rows_out_ps, engine=nisa.engine.vector)
        nl.store(out_hbm.ap(pattern=[[heads * latent, qpb], [1, latent]], offset=q0 * heads * latent),
                 value=out_rows)
    sbm.close_scope()


@nki.jit
def mla_block_shared_exact_kernel(q_lift_hbm, bank_hbm, table_hbm, topk_hbm, totals_hbm, out_hbm, softmax_scale,
                                  page_size, block_queries, source_digest=0, staging_digest=0):
    """``out_hbm`` with every flagged round's blocks overwritten by the exact path, written in place.

    Args:
        q_lift_hbm: ``[S, 1, L]`` bf16 or fp16, the absorbed queries; ``L`` a multiple of 128, at most 512.
        bank_hbm: ``[slots, L]``, the whole latent bank in ``q_lift_hbm``'s dtype, already holding this step's
            rows (the sparse kernel's bank form).
        table_hbm: ``[pages, 1]`` int32, the window's block table; -1 pads read page 0.
        topk_hbm: ``[S, K]`` int32, each query's selected window rows, -1 for a column with no token; ``K`` a
            multiple of 128. The sparse kernel's index rows at this call site.
        totals_hbm: ``[S / 128, 2]`` int32, the union producer's totals; column :data:`OVERFLOW_COLUMN` is 1 where
            that row's block has a cut union.
        out_hbm: ``[S, 1, L]`` fp32, the block-shared kernel's output. The kernel writes the rows of every
            block of a flagged round, row-major, and returns this tensor.
        softmax_scale: python float, positive.
        page_size: python int, rows per page, a power of two.
        block_queries: python int, queries per block, a multiple of 128; the program count times its rows of
            ``totals_hbm`` is at most :data:`_LINE`.
        source_digest, staging_digest: :data:`SOURCE_DIGEST` and ``mla_sparse.SOURCE_DIGEST``; they only key the
            compiled-kernel cache.

    Blocks are taken in rounds of one block per program: round ``k`` is blocks ``k * n + p`` for programs
    ``p < n``, and its trip count reads their ``totals`` rows. Every round's count is read before the first loop.
    """
    seq = q_lift_hbm.shape[0]
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    kernel_assert(block_queries % TOTALS_ROW_QUERIES == 0 and seq % block_queries == 0,
                  "whole blocks of whole totals rows")
    n_blocks = seq // block_queries
    kernel_assert(n_blocks % n_prgs == 0, "whole rounds of one block per program")
    round_rows = n_prgs * (block_queries // TOTALS_ROW_QUERIES)
    kernel_assert(round_rows <= _LINE, "a round's flags fit one line")
    sbm = SbufManager(sb_lower_bound=0, sb_upper_bound=nl.tile_size.sbuf_fmax_bytes)
    sbm.open_scope(name="mla_block_shared_exact")
    bank_rows = _bank_table(sbm, table_hbm, page_size)
    trips = []
    for k in range(n_blocks // n_prgs):
        trips.append(_trip_count(sbm, totals_hbm, k * round_rows, round_rows))
    for k in range(len(trips)):
        block = k * n_prgs + prg

        def run_round(i):
            _exact_block(sbm, q_lift_hbm, bank_hbm, topk_hbm, bank_rows, softmax_scale, page_size, block,
                         block_queries, out_hbm)

        nl.fori_loop(0, trips[k], run_round)
    sbm.close_scope()
    return out_hbm


def mla_block_shared_exact(q_lift: Tensor, c_kv: Tensor, topk_indices: Tensor, totals: Tensor, out: Tensor,
                           softmax_scale: float, block_table_row: Tensor, page_size: int = 128,
                           block_queries: int = 128) -> Tensor:
    """``out`` with every overflowed block's rows recomputed by today's per-query gather.

    Args:
        q_lift: ``[S, 1, L]`` bf16, as the block-shared call takes it.
        c_kv: the whole latent bank ``[slots, L]`` bf16, already holding this step's rows.
        topk_indices: ``[S, K]`` int32 window rows, -1 = no token: the sparse call's index rows.
        totals: ``[S / 128, 2]`` int32, the union producer's totals.
        out: ``[S, 1, L]`` fp32, contiguous: the block-shared call's output. It is written in place.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32.
        page_size: rows per page, a power of two.
        block_queries: queries per block, a multiple of 128.

    Returns:
        The written ``out``. A round of blocks, one per program, whose ``totals`` rows carry no flag keeps the
        block-shared rows; a round with a flag gets the sparse kernel's rows for every one of its blocks.

    Launch: two programs (LNC 2) when the call has an even number of blocks, else one.

    Raises:
        MlaBlockSharedExactError: on shapes, dtypes or a layout this contract does not take.
    """
    seq = int(q_lift.shape[0])
    if (q_lift.ndim != 3 or tuple(out.shape) != tuple(q_lift.shape) or out.dtype != torch.float32
            or not out.is_contiguous()):
        raise MlaBlockSharedExactError(
            f"out must be a contiguous fp32 tensor of q_lift's shape {tuple(q_lift.shape)}; got "
            f"{tuple(out.shape)} {out.dtype}")
    if block_queries % TOTALS_ROW_QUERIES != 0 or seq % block_queries != 0:
        raise MlaBlockSharedExactError(
            f"{seq} queries are not whole blocks of {block_queries}, a multiple of {TOTALS_ROW_QUERIES}")
    if tuple(totals.shape) != (seq // TOTALS_ROW_QUERIES, 2) or tuple(topk_indices.shape[:1]) != (seq,):
        raise MlaBlockSharedExactError(
            f"totals must be [{seq // TOTALS_ROW_QUERIES}, 2] and topk_indices [{seq}, K]; got "
            f"{tuple(totals.shape)} and {tuple(topk_indices.shape)}")
    call = wrap_nki(mla_block_shared_exact_kernel)
    if lnc_pair() and (seq // block_queries) % 2 == 0:
        call = call[2]
    return call(
        q_lift.contiguous(), c_kv.contiguous(), block_table_row.contiguous().to(torch.int32),
        topk_indices.contiguous().to(torch.int32), totals.contiguous().to(torch.int32), out,
        float(softmax_scale), int(page_size), int(block_queries), SOURCE_DIGEST, _ms.SOURCE_DIGEST,
    )
