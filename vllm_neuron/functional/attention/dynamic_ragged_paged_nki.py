# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Experimental dynamic-query-length ragged paged attention for NeuronCore-v3.

This is a correctness-first NKI analogue of vLLM-TPU's RPA v3 MIXED path.
It accepts different q_len and kv_len values for every request in one launch.

Contract:
  q:          [Hq, 128, max_tokens] BF16, packed on the token axis.
  key_pages:  [num_pages, Hkv, 128, 128] BF16, K transposed per cache page.
  value_pages:[num_pages, Hkv, 128, 128] BF16, V in [page_token, head_dim].
  page_table: [num_requests, max_pages_per_request] int32 (physical page ID).
  cu_q_lens:  [num_requests + 1] int32, cumulative *active* query tokens.
  kv_lens:    [num_requests] int32, post-append KV lengths (old + new).
  output:     [Hq, 128, max_tokens] BF16; unused token slots are unspecified.

Precondition: current-step keys/values have ALREADY been written to the
paged cache. This kernel only reads that cache; it does not append K/V.
Each request owns writable KV pages (or uses a safe copy-on-write policy).

This first baseline processes one query at a time (Q_TILE=1). It implements
runtime q_len and paged causal attention correctly but is NOT expected to
compete with a multi-query-tile CTE kernel for prefill throughput.
"""
import nki
import nki.isa as nisa
import nki.language as nl

_HEAD_DIM = 128
_PAGE_SIZE = 128
_MASKED_SCORE = -1.0e30
_SCALE = 0.08838834764831845  # 1 / sqrt(128)


@nki.jit
def dynamic_ragged_paged_attention_nki(
    q_hbm, key_pages_hbm, value_pages_hbm, page_table_hbm,
    cu_q_lens_hbm, kv_lens_hbm,
):
    """Run causal ragged attention with runtime q_len and page-table lookup.

    The launch grid is (num_requests, num_q_heads); a program processes
    one request and one query head. GQA maps its query head to a KV head.
    """
    head = nl.program_id(1)
    req = nl.program_id(0)
    num_heads, dim, max_tokens = q_hbm.shape
    num_pages, num_kv_heads, key_dim, page_size = key_pages_hbm.shape
    max_reqs, pages_per_req = page_table_hbm.shape
    group_size = num_heads // num_kv_heads
    kv_head = head // group_size

    # All geometry is static. The loop bounds and HBM indices are not.
    output = nl.ndarray(q_hbm.shape, dtype=q_hbm.dtype, buffer=nl.shared_hbm)
    start = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    end = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    kv_len = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    q_len = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    old_len = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=start,
        src=cu_q_lens_hbm.ap(pattern=[[1, 1], [1, 1]], offset=req),
    )
    nisa.dma_copy(
        dst=end,
        src=cu_q_lens_hbm.ap(pattern=[[1, 1], [1, 1]], offset=req + 1),
    )
    nisa.dma_copy(
        dst=kv_len,
        src=kv_lens_hbm.ap(pattern=[[1, 1], [1, 1]], offset=req),
    )
    nisa.tensor_tensor(dst=q_len, data1=end, data2=start, op=nl.subtract)
    nisa.tensor_tensor(dst=old_len, data1=kv_len, data2=q_len, op=nl.subtract)

    q_start_reg = nisa.register_alloc()
    q_end_reg = nisa.register_alloc()
    nisa.register_load(dst=q_start_reg, src=start)
    nisa.register_load(dst=q_end_reg, src=end)

    key_offsets = nl.ndarray((1, _PAGE_SIZE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=key_offsets, pattern=[[1, _PAGE_SIZE]])
    key_offsets_f = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=key_offsets_f, src=key_offsets)

    def query_body(q_index):
        # q_index is a VirtualRegister containing the global packed token ID.
        q_index_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.register_store(dst=q_index_i, src=q_index)
        # Absolute limit includes this token, plus all older cached tokens.
        # q_limit = kv_len - (q_end - q_start) + (q_index - q_start) + 1.
        q_rel = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        q_limit = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_rel, data1=q_index_i, data2=start,
                           op=nl.subtract)
        nisa.tensor_tensor(dst=q_limit, data1=old_len, data2=q_rel,
                           op=nl.add)
        nisa.tensor_scalar(dst=q_limit, data=q_limit, op0=nl.add, operand0=1)

        # ceil(q_limit / 128). FP32 cast truncates after dividing; the
        # supported max context (< 2**24) keeps integer values exact.
        q_limit_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        page_count_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        page_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_limit_f, src=q_limit)
        nisa.tensor_scalar(dst=page_count_f, data=q_limit_f,
                           op0=nl.add, operand0=127.0,
                           op1=nl.multiply, operand1=1.0 / 128.0)
        # Do not rely on the FP32 -> int32 cast's rounding mode.
        # floor((q_limit + PAGE - 1) / PAGE) is exact for this range.
        pages_floor = nl.floor(page_count_f)
        nisa.tensor_copy(dst=page_count, src=pages_floor)
        pages_reg = nisa.register_alloc()
        nisa.register_load(dst=pages_reg, src=page_count)

        # Q uses [head, dim, packed_token] so the TensorE K axis is already
        # on SBUF partitions. Read a single runtime query without transpose.
        q_sb = nl.ndarray((_HEAD_DIM, 1), dtype=q_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_sb,
            src=q_hbm.ap(
                pattern=[[max_tokens, _HEAD_DIM], [1, 1]],
                offset=head * _HEAD_DIM * max_tokens,
                scalar_offset=q_index, indirect_dim=2),
        )

        # Streaming FlashAttention state. Kept in SBUF across the dynamic
        # page loop, so memory usage does not depend on context length.
        m = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        l = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((1, _HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=m, value=_MASKED_SCORE)
        nisa.memset(dst=l, value=0.0)
        nisa.memset(dst=acc, value=0.0)

        def kv_page_body(page_index):
            # Read the physical page ID from the request's page table.
            # Dynamic offset is along page_table_hbm's second dimension.
            page_id = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=page_id,
                src=page_table_hbm.ap(
                    pattern=[[1, 1], [1, 1]],
                    offset=req * pages_per_req,
                    scalar_offset=page_index, indirect_dim=1),
            )
            k = nl.ndarray((_HEAD_DIM, _PAGE_SIZE),
                           dtype=key_pages_hbm.dtype, buffer=nl.sbuf)
            v = nl.ndarray((_PAGE_SIZE, _HEAD_DIM),
                           dtype=value_pages_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k,
                src=key_pages_hbm.ap(
                    pattern=[[_PAGE_SIZE, _HEAD_DIM], [1, _PAGE_SIZE]],
                    offset=kv_head * _HEAD_DIM * _PAGE_SIZE,
                    scalar_offset=page_id, indirect_dim=0),
            )
            nisa.dma_copy(
                dst=v,
                src=value_pages_hbm.ap(
                    pattern=[[_HEAD_DIM, _PAGE_SIZE], [1, _HEAD_DIM]],
                    offset=kv_head * _PAGE_SIZE * _HEAD_DIM,
                    scalar_offset=page_id, indirect_dim=0),
            )

            # This page has min(128, q_limit - page_index*128) visible
            # keys. Invalid positions must be excluded from softmax.
            page_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            page_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            visible = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            visible_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            mask_f = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            mask = nl.ndarray((1, _PAGE_SIZE), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.register_store(dst=page_i, src=page_index)
            nisa.tensor_scalar(dst=page_offset, data=page_i,
                               op0=nl.multiply, operand0=_PAGE_SIZE)
            nisa.tensor_tensor(dst=visible, data1=q_limit, data2=page_offset,
                               op=nl.subtract)
            nisa.tensor_scalar(dst=visible, data=visible,
                               op0=nl.minimum, operand0=_PAGE_SIZE)
            nisa.tensor_copy(dst=visible_f, src=visible)
            nisa.tensor_scalar(dst=mask_f, data=key_offsets_f,
                               op0=nl.less, operand0=visible_f)
            nisa.tensor_copy(dst=mask, src=mask_f)

            # Q[1, D] @ K[D, page] -> scores[1, page].
            score_ps = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32,
                                  buffer=nl.psum)
            nisa.nc_matmul(dst=score_ps, stationary=q_sb, moving=k,
                           accumulate=False)
            scores = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32,
                                buffer=nl.sbuf)
            nisa.tensor_copy(dst=scores, src=score_ps)
            nisa.tensor_scalar(dst=scores, data=scores,
                               op0=nl.multiply, operand0=_SCALE)
            masked = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32,
                                buffer=nl.sbuf)
            nisa.memset(dst=masked, value=_MASKED_SCORE)
            nisa.tensor_copy_predicated(dst=masked, src=scores, predicate=mask)

            tile_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            m_next = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            m_diff = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            alpha = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            neg_m = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=masked,
                               op=nl.maximum, axis=1)
            nisa.tensor_tensor(dst=m_next, data1=m, data2=tile_max,
                               op=nl.maximum)
            nisa.tensor_tensor(dst=m_diff, data1=m, data2=m_next,
                               op=nl.subtract)
            nisa.activation(dst=alpha, data=m_diff, op=nl.exp)
            nisa.tensor_scalar(dst=neg_m, data=m_next, op0=nl.multiply,
                               operand0=-1.0)

            p_f32 = nl.ndarray((1, _PAGE_SIZE), dtype=nl.float32,
                               buffer=nl.sbuf)
            nisa.activation(dst=p_f32, data=masked, op=nl.exp, bias=neg_m)
            p_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=p_sum, data=p_f32, op=nl.add, axis=1)
            # Stable online softmax l = l*alpha + sum(p).
            nisa.tensor_scalar(dst=l, data=l, op0=nl.multiply,
                               operand0=alpha)
            nisa.tensor_tensor(dst=l, data1=l, data2=p_sum, op=nl.add)

            # TensorE's stationary axis must be the contraction dimension.
            # Transpose P[1,page] to P.T[page,1] (fp32 PSUM avoids odd
            # BF16 transpose writes), then convert to BF16 for PV.
            p_ps = nl.ndarray((_PAGE_SIZE, 1), dtype=nl.float32,
                              buffer=nl.psum)
            nisa.nc_transpose(dst=p_ps, data=p_f32)
            p_bf16 = nl.ndarray((_PAGE_SIZE, 1), dtype=q_hbm.dtype,
                                buffer=nl.sbuf)
            nisa.tensor_copy(dst=p_bf16, src=p_ps)
            pv_ps = nl.ndarray((1, _HEAD_DIM), dtype=nl.float32,
                               buffer=nl.psum)
            nisa.nc_matmul(dst=pv_ps, stationary=p_bf16,
                           moving=v, accumulate=False)
            pv = nl.ndarray((1, _HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pv, src=pv_ps)

            # Stable online softmax acc = acc*alpha + p@v.
            nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply,
                               operand0=alpha)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=pv, op=nl.add)
            nisa.tensor_copy(dst=m, src=m_next)

        nl.fori_loop(0, pages_reg, kv_page_body)

        inv_l = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((1, _HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_l, data=l)
        nisa.tensor_scalar(dst=result, data=acc, op0=nl.multiply,
                           operand0=inv_l)
        # Output returns to the same [head, dim, packed_token] layout.
        o_ps = nl.ndarray((_HEAD_DIM, 1), dtype=nl.float32, buffer=nl.psum)
        o_sb = nl.ndarray((_HEAD_DIM, 1), dtype=q_hbm.dtype, buffer=nl.sbuf)
        nisa.nc_transpose(dst=o_ps, data=result)
        nisa.tensor_copy(dst=o_sb, src=o_ps)
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[max_tokens, _HEAD_DIM], [1, 1]],
                offset=head * _HEAD_DIM * max_tokens,
                scalar_offset=q_index, indirect_dim=2),
            src=o_sb,
        )

    nl.fori_loop(q_start_reg, q_end_reg, query_body)
    return output


def launch_dynamic_ragged_paged_attention(
    q, key_pages, value_pages, page_table, cu_q_lens, kv_lens,
):
    """Dispatch the standalone kernel. Call only with device tensors.

    This does not integrate with vLLM's scheduler or update the KV cache.
    The caller must validate metadata and prepopulate each cache page.
    """
    if q.ndim != 3 or q.shape[1] != _HEAD_DIM:
        raise ValueError("q must have shape [Hq, 128, max_tokens]")
    if key_pages.ndim != 4 or key_pages.shape[2:] != (128, 128):
        raise ValueError("key_pages must have shape [pages, Hkv, 128, 128]")
    if value_pages.shape != key_pages.shape:
        raise ValueError("value_pages must match key_pages shape")
    if key_pages.shape[1] == 0 or q.shape[0] % key_pages.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if q.dtype != key_pages.dtype or key_pages.dtype != value_pages.dtype:
        raise ValueError("q and KV pages must have the same dtype")
    if page_table.ndim != 2 or cu_q_lens.ndim != 1 or kv_lens.ndim != 1:
        raise ValueError("invalid metadata tensor rank")
    if page_table.shape[0] != kv_lens.shape[0]:
        raise ValueError("page table and KV lens request count mismatch")
    if cu_q_lens.shape[0] != page_table.shape[0] + 1:
        raise ValueError("cu_q_lens must have one extra end offset")
    if page_table.shape[1] * _PAGE_SIZE >= (1 << 24):
        raise ValueError("max context must be below 2**24 for page-count conversion")
    return dynamic_ragged_paged_attention_nki[
        (page_table.shape[0], q.shape[0])
    ](q, key_pages, value_pages, page_table, cu_q_lens, kv_lens)
