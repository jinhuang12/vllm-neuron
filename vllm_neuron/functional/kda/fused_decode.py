# SPDX-License-Identifier: Apache-2.0
"""Kimi delta attention (KDA) decode step for one rank's heads, in one NKI kernel.

Each token runs the short convolution and SiLU over q, k and v, the q and k L2
norms, the forget gate, the gated delta-rule state step and the gated RMSNorm
of the output::

    q, k, v = silu(sum_s w[c, s] * [conv rows; x][s, c])   # 4 taps per channel
    q, k    = q / |q| / sqrt(K), k / |k|
    decay   = exp(lower_bound * sigmoid(exp(A_log) * (f_b(f_a) + dt_bias)))
    S       = decay * S;  S += k (sigmoid(b) * (v - S^T k))^T  # S is [K, V]
    core    = rmsnorm(S^T q) * o_norm_weight * sigmoid(g)

The two states live in the KV cache's page buffer, one page per slot: the conv
rows from the page's first byte, then the recurrent state. The kernel reads
each token's rows from its slot's page and writes the advanced rows back to the
same page, so the buffer is updated in place and returned. The tokens split
over the logical core's programs, so each program writes whole rows: its
tokens' pages and its tokens' runs of the output.

The output is [128, T, 3, 1], head dimension first: the input layout of the
output projection's decode kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

HEADS = 3
DIM = 128
WIDTH = HEADS * DIM
TAPS = 4
ROWS = TAPS - 1
CONV_WIDTH = 3 * WIDTH
COLUMNS = 4 * WIDTH + DIM + HEADS
GATE_COL = 3 * WIDTH
FA_COL = 4 * WIDTH
BETA_COL = 4 * WIDTH + DIM
L2_EPS = 1e-6
MAX_TOKENS = 32
TOKENS_PER_PASS = 4
CELLS_PER_BANK = 4
CONV_BYTES = ROWS * CONV_WIDTH * 2
STATE_BYTES = HEADS * DIM * DIM * 4
# Fixed-address loads use descriptors built at compile time, which keeps them off the
# GpSimd queue; a page's address comes from its slot, read at run time.
STATIC = nisa.dge_mode.none
BY_SLOT = nisa.dge_mode.hwdge


def kda_fused_decode(
    x: torch.Tensor,
    slots: torch.Tensor,
    pages: torch.Tensor,
    q_conv_weight: torch.Tensor,
    k_conv_weight: torch.Tensor,
    v_conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    f_b_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    *,
    lower_bound: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one decode step; returns the gated-norm output and the updated page buffer.

    Args:
        x: [T, 1667] bf16 projections per token, columns q, k, v, g (each 384),
            f_a (128) and the 3 beta logits.
        slots: [T] int32 page of each token. Slots are distinct, and every slot
            is a page of the buffer; a padding token takes a slot of its own.
        pages: [slots, page bytes] int8 page buffer; the second dimension is the
            row stride. A page holds the conv rows [3, 1152] bf16 from byte 0,
            oldest first, channels q | k | v, then the fp32 recurrent state
            [3, 128, 128] per head, [K, V]. The kernel writes no other byte.
        q_conv_weight, k_conv_weight, v_conv_weight: [384, 1, 4] fp32 taps.
        a_log: [3] fp32. dt_bias: [384] fp32. f_b_weight: [384, 128] bf16.
        o_norm_weight: [128] fp32.

    Returns:
        core [128, T, 3, 1] bf16, element [d, t, h] the channel d of head h of token t,
        and the page buffer with the token pages advanced.
    """
    tokens = x.shape[0]
    assert x.shape == (tokens, COLUMNS) and x.dtype == torch.bfloat16, f"x {tuple(x.shape)} {x.dtype}"
    assert 1 <= tokens <= MAX_TOKENS, f"{tokens} tokens; the kernel takes 1 to {MAX_TOKENS}"
    assert slots.shape == (tokens,) and slots.dtype == torch.int32, \
        f"slots {tuple(slots.shape)} {slots.dtype}"
    assert pages.dim() == 2 and pages.dtype == torch.int8, f"pages {tuple(pages.shape)} {pages.dtype}"
    assert pages.shape[1] >= CONV_BYTES + STATE_BYTES and pages.shape[1] % 4 == 0, \
        f"a {pages.shape[1]}-byte page cannot hold the states at fp32 alignment"
    for weight in (q_conv_weight, k_conv_weight, v_conv_weight):
        assert weight.shape == (WIDTH, 1, TAPS) and weight.dtype == torch.float32
    assert a_log.shape == (HEADS,) and dt_bias.shape == (WIDTH,) and o_norm_weight.shape == (DIM,)
    assert f_b_weight.shape == (WIDTH, DIM) and f_b_weight.dtype == torch.bfloat16
    if not can_run_kernel(x):
        raise RuntimeError("kda_fused_decode needs a Neuron device or the NKI simulator")
    kernel = wrap_nki(kda_fused_decode_kernel)[get_logical_nc_config()]
    return kernel(x=x, slots=slots.reshape(1, tokens), pages=pages,
                  q_w=q_conv_weight, k_w=k_conv_weight, v_w=v_conv_weight, a_log=a_log,
                  dt_bias=dt_bias, f_b=f_b_weight, o_w=o_norm_weight,
                  lower_bound=float(lower_bound), norm_eps=float(norm_eps))


@nki.jit
def kda_fused_decode_kernel(x, slots, pages, q_w, k_w, v_w, a_log, dt_bias, f_b, o_w,
                            lower_bound, norm_eps):
    """The decode step of :func:`kda_fused_decode`; the page buffer is written in place."""
    tokens = x.shape[0]
    kernel_assert(x.shape[1] == COLUMNS, f"x has {x.shape[1]} columns, expected {COLUMNS}")
    core = nl.ndarray((DIM, tokens, HEADS, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # One page per slot, read in each state's dtype: a slot step is one page.
    conv_pages = pages.view(nl.bfloat16)
    state_pages = pages.view(nl.float32)

    programs = nl.num_programs(0)
    program = nl.program_id(0)
    per = -(-tokens // programs)
    first = min(program * per, tokens)
    end = min(first + per, tokens)
    if end > first:
        consts = _load_constants(q_w, k_w, v_w, a_log, dt_bias, f_b, o_w)
        idx = nl.ndarray((1, tokens), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=slots, dge_mode=STATIC)
        for t0 in range(first, end, TOKENS_PER_PASS):
            nt = min(TOKENS_PER_PASS, end - t0)
            _step_tokens(x, idx, conv_pages, state_pages, core, consts, t0, nt, lower_bound, norm_eps)
    return core, pages


def _load_constants(q_w, k_w, v_w, a_log, dt_bias, f_b, o_w):
    """The weights, laid out for the step, and the matmul constants."""
    nh = HEADS
    taps = nl.ndarray((DIM, 3 * nh, 1, TAPS), dtype=nl.float32, buffer=nl.sbuf)
    for s, w in enumerate((q_w, k_w, v_w)):
        nisa.dma_copy(dst=taps[:, s * nh:(s + 1) * nh, 0, :],
                      src=w.ap(pattern=[[TAPS, DIM], [DIM * TAPS, nh], [1, TAPS]], offset=0),
                      dge_mode=STATIC)
    bias = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias, src=dt_bias.ap(pattern=[[1, DIM], [DIM, nh]], offset=0), dge_mode=STATIC)
    a_raw = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_raw, src=a_log.ap(pattern=[[0, DIM], [1, nh]], offset=0), dge_mode=STATIC)
    a_exp = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a_exp, op=nl.exp, data=a_raw, engine=nisa.engine.gpsimd)
    norm_w = nl.ndarray((DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=norm_w, src=o_w.ap(pattern=[[1, DIM], [1, 1]], offset=0), dge_mode=STATIC)

    # f_b rows of each head, turned so the contraction axis (rank) is on partitions.
    fb_rows = nl.ndarray((DIM, nh, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=fb_rows,
                  src=f_b.ap(pattern=[[DIM, DIM], [DIM * DIM, nh], [1, DIM]], offset=0),
                  dge_mode=STATIC)
    fb_turned = nl.ndarray((DIM, nh * DIM), dtype=nl.bfloat16, buffer=nl.psum)
    for h in range(nh):
        nisa.nc_transpose(dst=fb_turned[:, h * DIM:(h + 1) * DIM], data=fb_rows[:, h, :])
    fb_t = nl.ndarray((DIM, nh, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=fb_t.reshape((DIM, nh * DIM)), src=fb_turned)

    ones = nl.ndarray((DIM, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    eye = nl.shared_identity_matrix(n=DIM, dtype=nl.float32)
    neg_eye = nl.ndarray((DIM, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_eye, data=eye, op0=nl.multiply, operand0=-1.0)
    return taps, bias, a_exp, norm_w, fb_t, ones, neg_eye


def _rows(dtype, n):
    """Rows 0 .. n-1 of a full-partition SBUF tile, so every operand starts at partition 0."""
    return nl.ndarray((DIM, DIM), dtype=dtype, buffer=nl.sbuf)[0:n]


def _step_tokens(x, idx, conv_pages, state_pages, core, consts, t0, nt, lower_bound, norm_eps):
    """Step tokens t0 .. t0+nt-1 and store their output columns and page rows."""
    taps, bias, a_exp, norm_w, fb_t, ones, neg_eye = consts
    tokens = x.shape[0]
    nh = HEADS
    cells = nt * nh
    blocks = 3 * nh
    slot = [idx.ap(pattern=[[tokens, 1], [1, 1]], offset=t0 + u) for u in range(nt)]

    # Page rows by slot: per token its 3 conv rows then its own row, and its state.
    rows = nl.ndarray((DIM, 3, WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
    state = nl.ndarray((DIM, cells, DIM), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(nt):
        nisa.dma_copy(dst=rows[TAPS * u:TAPS * u + ROWS],
                      src=_conv_rows(conv_pages, slot[u]), dge_mode=BY_SLOT)
        nisa.dma_copy(dst=rows[TAPS * u + ROWS:TAPS * (u + 1)],
                      src=x.ap(pattern=[[COLUMNS, 1], [1, CONV_WIDTH]], offset=(t0 + u) * COLUMNS),
                      dge_mode=STATIC)
        nisa.dma_copy(dst=state[:, u * nh:(u + 1) * nh, :], src=_state_rows(state_pages, slot[u]),
                      dge_mode=BY_SLOT)

    # Token columns that do not depend on the pages.
    fa_t = nl.ndarray((DIM, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=fa_t, src=x.ap(pattern=[[1, DIM], [COLUMNS, nt]], offset=t0 * COLUMNS + FA_COL),
                  dge_mode=STATIC)
    beta_raw = nl.ndarray((DIM, cells), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_raw, src=x.ap(pattern=[[0, DIM], [COLUMNS, nt], [1, nh]],
                                         offset=t0 * COLUMNS + BETA_COL), dge_mode=STATIC)
    gate_rows = _rows(nl.bfloat16, cells)
    for u in range(nt):
        nisa.dma_copy(dst=gate_rows[u * nh:(u + 1) * nh],
                      src=x.ap(pattern=[[DIM, nh], [1, DIM]], offset=(t0 + u) * COLUMNS + GATE_COL),
                      dge_mode=STATIC)

    # Output gate times the norm weight, columns per cell.
    gate_cols = nl.ndarray((DIM, cells), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=gate_cols, data=gate_rows)
    gate_sig = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_sig, op=nl.sigmoid, data=gate_cols)
    gate_w = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gate_w, data=gate_sig, op0=nl.multiply, operand0=norm_w)

    # Forget gate: decay = exp(lower_bound * sigmoid(exp(A_log) * (f_b f_a + dt_bias))).
    # exp and rsqrt run on GpSimd, whose results are closer than Scalar's; their error
    # would compound in the carried state. Sigmoid and SiLU stay on Scalar.
    raw_gate = nl.ndarray((DIM, nh, nt), dtype=nl.float32, buffer=nl.psum)
    for h in range(nh):
        nisa.nc_matmul(dst=raw_gate[:, h, :], stationary=fb_t[:, h, :], moving=fa_t)
    biased = nl.ndarray((DIM, nt, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=biased, data1=raw_gate.permute((0, 2, 1)), data2=bias.broadcast(1, nt), op=nl.add)
    scaled = nl.ndarray((DIM, nt, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=biased, data2=a_exp.broadcast(1, nt), op=nl.multiply)
    squashed = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=squashed, op=nl.sigmoid, data=scaled.reshape((DIM, cells)))
    decay = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=decay, op=nl.exp, data=squashed, scale=float(lower_bound), engine=nisa.engine.gpsimd)
    beta = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta, op=nl.sigmoid, data=beta_raw)
    decayed = nl.ndarray((DIM, cells, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=decayed, data1=state, data2=decay.reshape((DIM, cells, 1)).broadcast(2, DIM),
                       op=nl.multiply)

    # Short convolution: channels on partitions, taps on the free axis.
    turned = nl.ndarray((DIM, blocks, TAPS * nt), dtype=nl.bfloat16, buffer=nl.psum)
    for s in range(3):
        for h in range(nh):
            nisa.nc_transpose(dst=turned[:, s * nh + h, :],
                              data=rows[0:TAPS * nt, s, h * DIM:(h + 1) * DIM])
    prod = nl.ndarray((DIM, blocks, nt, TAPS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=turned.reshape((DIM, blocks, nt, TAPS)), data2=taps.broadcast(2, nt),
                       op=nl.multiply)
    summed = nl.ndarray((DIM, blocks, nt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=summed, op=nl.add, data=prod, axis=(3,))
    act = nl.ndarray((DIM, 3, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=act.reshape((DIM, 3, nt, nh)).permute((0, 1, 3, 2)), op=nl.silu,
                    data=summed.reshape((DIM, 3, nh, nt)))

    # L2 norms of q and k over the head dimension, summed on every partition by a ones matmul.
    squares = nl.ndarray((DIM, 2 * cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=squares, data1=act[:, 0:2, :], data2=act[:, 0:2, :], op=nl.multiply)
    sums_ps = nl.ndarray((DIM, 2 * cells), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=sums_ps, stationary=ones, moving=squares)
    sums = nl.ndarray((DIM, 2 * cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sums, src=sums_ps)
    inv = nl.ndarray((DIM, 2 * cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=sums, bias=L2_EPS, engine=nisa.engine.gpsimd)
    qn = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=qn, data=act[:, 0, :], op0=nl.multiply, operand0=DIM ** -0.5,
                              op1=nl.multiply, operand1=inv[:, 0:cells])
    kn = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=kn, data1=act[:, 1, :], data2=inv[:, cells:2 * cells], op=nl.multiply)
    k_decayed = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_decayed, data1=kn, data2=decay, op=nl.multiply)
    neg_bk = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=neg_bk, data=beta, op0=nl.multiply, operand0=-1.0, op1=nl.multiply,
                              operand1=kn)

    # State step per cell. resid = (decay S)^T k - v lands on every partition: k decayed is
    # broadcast as the stationary, and v broadcast against -I adds -v. A cell's operands are
    # its own token's, so a NaN token stays in its cells.
    updated = nl.ndarray((DIM, cells, DIM), dtype=nl.float32, buffer=nl.sbuf)
    out_ps = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.psum)
    banks = [nl.ndarray((DIM, CELLS_PER_BANK, DIM), dtype=nl.float32, buffer=nl.psum)
             for _ in range(-(-cells // CELLS_PER_BANK))]
    for c in range(cells):
        resid = banks[c // CELLS_PER_BANK][:, c % CELLS_PER_BANK, :]
        nisa.nc_matmul(dst=resid, stationary=act[:, 2, c:c + 1].broadcast(1, DIM), moving=neg_eye,
                       accumulate=False)
        nisa.nc_matmul(dst=resid, stationary=k_decayed[:, c:c + 1].broadcast(1, DIM), moving=state[:, c, :],
                       accumulate=True)
        nisa.scalar_tensor_tensor(dst=updated[:, c, :], data=resid, op0=nl.multiply,
                                  operand0=neg_bk[:, c:c + 1], op1=nl.add, operand1=decayed[:, c, :])
        nisa.nc_matmul(dst=out_ps[:, c:c + 1], stationary=updated[:, c, :], moving=qn[:, c:c + 1],
                       accumulate=False)

    # Gated RMSNorm of the output, columns per cell, summed over the head dimension by a ones matmul.
    out_sq = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=out_sq, op=nl.square, data=out_ps)
    out_ss_ps = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_ss_ps, stationary=ones, moving=out_sq)
    out_ss = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_ss, src=out_ss_ps)
    rstd = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=out_ss, scale=1.0 / DIM, bias=float(norm_eps),
                    engine=nisa.engine.gpsimd)
    scale = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scale, data1=rstd, data2=gate_w, op=nl.multiply)
    y = nl.ndarray((DIM, cells), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=out_ps, data2=scale, op=nl.multiply)

    # Stores: the output columns (cell c is item 3 t0 + c of the [128, 3T] layout), then each
    # token's state and its 3 newest conv rows, into its page.
    nisa.dma_copy(dst=core.ap(pattern=[[HEADS * tokens, DIM], [1, cells]], offset=HEADS * t0), src=y,
                  dge_mode=STATIC)
    for u in range(nt):
        nisa.dma_copy(dst=_state_rows(state_pages, slot[u]), src=updated[:, u * nh:(u + 1) * nh, :],
                      dge_mode=BY_SLOT)
        nisa.dma_copy(dst=_conv_rows(conv_pages, slot[u]), src=rows[TAPS * u + 1:TAPS * (u + 1)],
                      dge_mode=BY_SLOT)


def _conv_rows(conv_pages, slot):
    """The 3 conv rows [3, 1152] of the slot's page, from byte 0."""
    return conv_pages.ap(pattern=[[CONV_WIDTH, ROWS], [1, CONV_WIDTH]], offset=0,
                         scalar_offset=slot, indirect_dim=0)


def _state_rows(state_pages, slot):
    """The slot's recurrent state as [K, heads, V], right after its conv rows in the page."""
    return state_pages.ap(pattern=[[DIM, DIM], [DIM * DIM, HEADS], [1, DIM]], offset=CONV_BYTES // 4,
                          scalar_offset=slot, indirect_dim=0)
