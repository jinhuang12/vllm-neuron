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

The kernel reads each token's conv rows and recurrent state from the banks at
the token's slot, and writes the advanced rows back to the same slot, so the
banks are updated in place and returned. The (token, head) items split over
the logical core's programs; a token shared by two programs has its rows
written by each program for its own heads only.
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
CELLS_PER_PASS = 12
CELLS_PER_BANK = 4
# Fixed-address loads use descriptors built at compile time, which keeps them off the
# GpSimd queue; a bank row's address comes from its slot, read at run time.
STATIC = nisa.dge_mode.none
BY_SLOT = nisa.dge_mode.hwdge


def kda_fused_decode(
    x: torch.Tensor,
    slots: torch.Tensor,
    conv_bank: torch.Tensor,
    state_bank: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one decode step; returns the gated-norm output and the two updated banks.

    Args:
        x: [T, 1667] bf16 projections per token, columns q, k, v, g (each 384),
            f_a (128) and the 3 beta logits.
        slots: [T] int32 bank slot of each token. Slots are distinct, and every
            slot is inside the banks; a padding token takes a slot of its own.
        conv_bank: [slots, 3, 1152] bf16 conv rows, oldest first, channels q | k | v.
        state_bank: [slots, 3, 128, 128] fp32 recurrent state per head, [K, V].
        q_conv_weight, k_conv_weight, v_conv_weight: [384, 1, 4] fp32 taps.
        a_log: [3] fp32. dt_bias: [384] fp32. f_b_weight: [384, 128] bf16.
        o_norm_weight: [128] fp32.

    Returns:
        core [T, 384] bf16, and conv_bank and state_bank with the token slots advanced.
    """
    tokens = x.shape[0]
    assert x.shape == (tokens, COLUMNS) and x.dtype == torch.bfloat16, f"x {tuple(x.shape)} {x.dtype}"
    assert 1 <= tokens <= MAX_TOKENS, f"{tokens} tokens; the kernel takes 1 to {MAX_TOKENS}"
    assert slots.shape == (tokens,) and slots.dtype == torch.int32, \
        f"slots {tuple(slots.shape)} {slots.dtype}"
    assert conv_bank.shape[1:] == (ROWS, CONV_WIDTH) and conv_bank.dtype == torch.bfloat16
    assert state_bank.shape[1:] == (HEADS, DIM, DIM) and state_bank.dtype == torch.float32
    assert conv_bank.shape[0] == state_bank.shape[0]
    for weight in (q_conv_weight, k_conv_weight, v_conv_weight):
        assert weight.shape == (WIDTH, 1, TAPS) and weight.dtype == torch.float32
    assert a_log.shape == (HEADS,) and dt_bias.shape == (WIDTH,) and o_norm_weight.shape == (DIM,)
    assert f_b_weight.shape == (WIDTH, DIM) and f_b_weight.dtype == torch.bfloat16
    if not can_run_kernel(x):
        raise RuntimeError("kda_fused_decode needs a Neuron device or the NKI simulator")
    kernel = wrap_nki(kda_fused_decode_kernel)[get_logical_nc_config()]
    return kernel(x=x, slots=slots.reshape(1, tokens), conv_bank=conv_bank, state_bank=state_bank,
                  q_w=q_conv_weight, k_w=k_conv_weight, v_w=v_conv_weight, a_log=a_log,
                  dt_bias=dt_bias, f_b=f_b_weight, o_w=o_norm_weight,
                  lower_bound=float(lower_bound), norm_eps=float(norm_eps))


@nki.jit
def kda_fused_decode_kernel(x, slots, conv_bank, state_bank, q_w, k_w, v_w, a_log, dt_bias, f_b,
                            o_w, lower_bound, norm_eps):
    """The decode step of :func:`kda_fused_decode`; the banks are written in place."""
    tokens = x.shape[0]
    kernel_assert(x.shape[1] == COLUMNS, f"x has {x.shape[1]} columns, expected {COLUMNS}")
    core = nl.ndarray((tokens, WIDTH), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    programs = nl.num_programs(0)
    program = nl.program_id(0)
    per = -(-HEADS * tokens // programs)
    lo = min(program * per, HEADS * tokens)
    hi = min(lo + per, HEADS * tokens)
    if hi > lo:
        first, last = lo // HEADS, (hi - 1) // HEADS
        if first == last:
            ha, hb = lo % HEADS, (hi - 1) % HEADS + 1
        else:
            ha, hb = 0, HEADS
        consts = _load_constants(q_w, k_w, v_w, a_log, dt_bias, f_b, o_w, ha, hb)
        idx = nl.ndarray((1, tokens), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=slots, dge_mode=STATIC)
        step = max(1, CELLS_PER_PASS // (hb - ha))
        for t0 in range(first, last + 1, step):
            nt = min(step, last + 1 - t0)
            _step_tokens(x, idx, conv_bank, state_bank, core, consts, t0, nt, ha, hb, lo, hi,
                         lower_bound, norm_eps)
    return core, conv_bank, state_bank


def _load_constants(q_w, k_w, v_w, a_log, dt_bias, f_b, o_w, ha, hb):
    """Weights of heads ha .. hb-1, laid out for the step, and the matmul constants."""
    nh = hb - ha
    taps = nl.ndarray((DIM, 3 * nh, 1, TAPS), dtype=nl.float32, buffer=nl.sbuf)
    for s, w in enumerate((q_w, k_w, v_w)):
        nisa.dma_copy(dst=taps[:, s * nh:(s + 1) * nh, 0, :],
                      src=w.ap(pattern=[[TAPS, DIM], [DIM * TAPS, nh], [1, TAPS]], offset=ha * DIM * TAPS),
                      dge_mode=STATIC)
    bias = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias, src=dt_bias.ap(pattern=[[1, DIM], [DIM, nh]], offset=ha * DIM), dge_mode=STATIC)
    a_raw = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_raw, src=a_log.ap(pattern=[[0, DIM], [1, nh]], offset=ha), dge_mode=STATIC)
    a_exp = nl.ndarray((DIM, 1, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a_exp, op=nl.exp, data=a_raw, engine=nisa.engine.gpsimd)
    norm_w = nl.ndarray((DIM, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=norm_w, src=o_w.ap(pattern=[[0, DIM], [1, DIM]], offset=0), dge_mode=STATIC)

    # f_b rows of each head, turned so the contraction axis (rank) is on partitions.
    fb_rows = nl.ndarray((DIM, nh, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=fb_rows,
                  src=f_b.ap(pattern=[[DIM, DIM], [DIM * DIM, nh], [1, DIM]], offset=ha * DIM * DIM),
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


def _rows(dtype, n, width=DIM):
    """Rows 0 .. n-1 of a full-partition SBUF tile, so every operand starts at partition 0."""
    return nl.ndarray((DIM, width), dtype=dtype, buffer=nl.sbuf)[0:n]


def _step_tokens(x, idx, conv_bank, state_bank, core, consts, t0, nt, ha, hb, lo, hi,
                 lower_bound, norm_eps):
    """Step tokens t0 .. t0+nt-1 on heads ha .. hb-1; store the items in [lo, hi)."""
    taps, bias, a_exp, norm_w, fb_t, ones, neg_eye = consts
    tokens = x.shape[0]
    nh = hb - ha
    cells = nt * nh
    blocks = 3 * nh
    slot = [idx.ap(pattern=[[tokens, 1], [1, 1]], offset=t0 + u) for u in range(nt)]

    # Bank rows by slot: per token its 3 conv rows then its own row, and its state.
    rows = nl.ndarray((DIM, 3, WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
    state = nl.ndarray((DIM, cells, DIM), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(nt):
        nisa.dma_copy(dst=rows[TAPS * u:TAPS * u + ROWS],
                      src=conv_bank.ap(pattern=[[CONV_WIDTH, ROWS], [1, CONV_WIDTH]], offset=0,
                                       scalar_offset=slot[u], indirect_dim=0),
                      dge_mode=BY_SLOT)
        nisa.dma_copy(dst=rows[TAPS * u + ROWS:TAPS * (u + 1)],
                      src=x.ap(pattern=[[COLUMNS, 1], [1, CONV_WIDTH]], offset=(t0 + u) * COLUMNS),
                      dge_mode=STATIC)
        nisa.dma_copy(dst=state[:, u * nh:(u + 1) * nh, :],
                      src=state_bank.ap(pattern=[[DIM, DIM], [DIM * DIM, nh], [1, DIM]], offset=ha * DIM * DIM,
                                        scalar_offset=slot[u], indirect_dim=0),
                      dge_mode=BY_SLOT)

    # Token columns that do not depend on the banks.
    fa_t = nl.ndarray((DIM, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=fa_t, src=x.ap(pattern=[[1, DIM], [COLUMNS, nt]], offset=t0 * COLUMNS + FA_COL),
                  dge_mode=STATIC)
    beta_raw = nl.ndarray((DIM, cells), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_raw, src=x.ap(pattern=[[0, DIM], [COLUMNS, nt], [1, nh]],
                                         offset=t0 * COLUMNS + BETA_COL + ha), dge_mode=STATIC)
    gate_rows = _rows(nl.bfloat16, cells)
    for u in range(nt):
        nisa.dma_copy(dst=gate_rows[u * nh:(u + 1) * nh],
                      src=x.ap(pattern=[[DIM, nh], [1, DIM]], offset=(t0 + u) * COLUMNS + GATE_COL + ha * DIM),
                      dge_mode=STATIC)

    # Output gate times the norm weight, rows per cell.
    gate_sig = _rows(nl.float32, cells)
    nisa.activation(dst=gate_sig, op=nl.sigmoid, data=gate_rows)
    gate_w = _rows(nl.float32, cells)
    nisa.tensor_tensor(dst=gate_w, data1=gate_sig, data2=norm_w[0:cells], op=nl.multiply)

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
                              data=rows[0:TAPS * nt, s, (ha + h) * DIM:(ha + h + 1) * DIM])
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

    # Gated RMSNorm of the output, rows per cell.
    out_cols = nl.ndarray((DIM, cells), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_cols, src=out_ps)
    out_rows = nl.ndarray((cells, DIM), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=out_rows, data=out_cols)
    squared = _rows(nl.float32, cells)
    sq_sum = _rows(nl.float32, cells, 1)
    nisa.activation_reduce(dst=squared, op=nl.square, data=out_rows, reduce_op=nl.add, reduce_res=sq_sum)
    rstd = _rows(nl.float32, cells, 1)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=sq_sum, scale=1.0 / DIM, bias=float(norm_eps),
                    engine=nisa.engine.gpsimd)
    y = _rows(nl.bfloat16, cells)
    nisa.scalar_tensor_tensor(dst=y, data=out_rows, op0=nl.multiply, operand0=rstd, op1=nl.multiply,
                              operand1=gate_w)

    # Stores: the output rows, then each token's state and conv rows for its own heads only.
    item = [HEADS * (t0 + c // nh) + ha + c % nh for c in range(cells)]
    own = [c for c in range(cells) if lo <= item[c] < hi]
    c0, c1 = own[0], own[-1] + 1
    nisa.dma_copy(dst=core.ap(pattern=[[DIM, c1 - c0], [1, DIM]], offset=item[c0] * DIM), src=y[c0:c1])
    for u in range(nt):
        t = t0 + u
        hs, he = max(ha, lo - HEADS * t), min(hb, hi - HEADS * t)
        if he <= hs:
            continue
        cs = u * nh + hs - ha
        nisa.dma_copy(dst=state_bank.ap(pattern=[[DIM, DIM], [DIM * DIM, he - hs], [1, DIM]],
                                        offset=hs * DIM * DIM, scalar_offset=slot[u], indirect_dim=0),
                      src=updated[:, cs:cs + he - hs, :], dge_mode=BY_SLOT)
        nisa.dma_copy(dst=conv_bank.ap(pattern=[[CONV_WIDTH, ROWS], [WIDTH, 3], [1, (he - hs) * DIM]],
                                       offset=hs * DIM, scalar_offset=slot[u], indirect_dim=0),
                      src=rows[TAPS * u + 1:TAPS * (u + 1), :, hs * DIM:he * DIM], dge_mode=BY_SLOT)
