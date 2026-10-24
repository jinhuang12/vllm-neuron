# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the KDA prefill recurrence, fp32 throughout.

Derived from nkilib ``kda_prefill`` (Copyright Amazon.com, Inc. or its
affiliates, Apache-2.0), whose chunked algebra it keeps: 128-token tiles of
eight 16-token sub-chunks, the decay folded into the operands, a four-factor
Neumann inverse and the WY form. It differs in three ways. Every matmul operand
is fp32. q, k and v arrive as the short conv stores them, bf16 channel-major
``[H * D, T]``. A token with a non-finite q, k, v, g or beta, or one the caller
marks invalid, is padding: before any product its normed q and k, v and beta
are zero and its gate is exactly 0, so it adds nothing to the state. From a
non-finite token to the end of the sequence the output rows, and the final
state, are NaN. Causal masks apply by select, never by multiplication.

Tiles suffixed ``_T`` are channel-major ``[D, C]``, unsuffixed ``[C, D]`` tiles
token-major, and tiles suffixed ``3`` sub-chunk-major ``[SUB, NSUB, D]``.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

L2_EPS = 1.0e-6
CHUNK = 128
SUB_CHUNK = 16
# The sub-chunk bounds k * exp(-gcum) by exp(|lower_bound| * SUB_CHUNK), 5.5e34
# at lower_bound -5. A 32-token sub-chunk would overflow fp32.
NEUMANN = 4
GROUP_MAX = 6


@nki.jit
def kda_prefill_kernel(q_ref, k_ref, v_ref, g_ref, beta_ref, a_log_ref, dt_bias_ref, valid_ref,
                       strict_ref, incl_ref, eye_ref, out, state_ref, scale_ref, lower_bound_ref):
    """KDA prefill over one sequence: fills ``out`` ``[H, T, D]`` and updates ``state_ref`` ``[H, D, D]`` in place.

    q_ref, k_ref, v_ref: bf16 ``[H * D, T]``, raw q and k (before the L2 norm), and v.
    g_ref: fp32 ``[H, T, D]``, raw gate; the kernel applies
    ``lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``. beta_ref: fp32 ``[H, T, 1]``
    logits. a_log_ref ``[H, 1]``, dt_bias_ref ``[H, D]``. valid_ref: fp32 ``[T, 1]``, 1
    for a real token and 0 for padding. Each head checks its own inputs. state_ref: fp32 key-major ``[H, D, D]``, the
    seed in and the final state out. scale_ref, lower_bound_ref: fp32 ``[1, 1]``.
    """
    nv, T, d = g_ref.shape
    C, SUB = CHUNK, SUB_CHUNK
    NSUB = C // SUB
    kernel_assert(d <= 128, f"head dim must be at most 128, got {d}")
    kernel_assert(tuple(q_ref.shape) == (nv * d, T) and tuple(k_ref.shape) == (nv * d, T),
                  f"q and k must be [{nv * d}, {T}]")
    kernel_assert(tuple(v_ref.shape) == (nv * d, T), f"v must be [{nv * d}, {T}]")
    kernel_assert(T % C == 0 and T >= C, f"T must be a positive multiple of {C}, got {T}")

    if nl.program_ndim() != 0:
        n_progs, prog_id = nl.num_programs(axes=0), nl.program_id(axis=0)
    else:
        n_progs, prog_id = 1, 0
    per_core = nv // n_progs
    h_start = prog_id * per_core
    h_end = nv if prog_id == n_progs - 1 else h_start + per_core

    f32 = nl.float32

    def sbuf(shape, dtype=f32):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def psum(shape):
        return nl.ndarray(shape, dtype=f32, buffer=nl.psum)

    # Loop-invariant constants. A predicate is 1 where a value is replaced.
    eye = sbuf((C, C))
    nisa.dma_copy(dst=eye, src=eye_ref[0:C, 0:C])
    off_strict, off_incl = sbuf((C, C), nl.uint8), sbuf((C, C), nl.uint8)
    for mask_ref, off in ((strict_ref, off_strict), (incl_ref, off_incl)):
        mask = sbuf((C, C))
        nisa.dma_copy(dst=mask, src=mask_ref[0:C, 0:C])
        nisa.tensor_scalar(dst=off, data=mask, op0=nl.equal, operand0=0.0)
    ones_col, ones_dd, ones_cc, zeros_cd = sbuf((d, 1)), sbuf((d, d)), sbuf((C, C)), sbuf((C, d))
    nisa.memset(dst=ones_col, value=1.0)
    nisa.memset(dst=ones_dd, value=1.0)
    nisa.memset(dst=ones_cc, value=1.0)
    nisa.memset(dst=zeros_cd, value=0.0)
    # upto[k, m] = 1 for k <= m, so upto.T @ x is the running sum of x over tokens.
    col_minus_row = sbuf((C, C), nl.int32)
    nisa.iota(dst=col_minus_row, pattern=[[1, C]], channel_multiplier=-1)
    upto = sbuf((C, C))
    nisa.tensor_scalar(dst=upto, data=col_minus_row, op0=nl.greater_equal, operand0=0.0)
    ones_scan, zero_init = sbuf((d, SUB)), sbuf((d, 1))
    nisa.memset(dst=ones_scan, value=1.0)
    nisa.memset(dst=zero_init, value=0.0)
    scale_t, lower_t = sbuf((1, 1)), sbuf((1, 1))
    nisa.dma_copy(dst=scale_t, src=scale_ref[0:1, 0:1])
    nisa.dma_copy(dst=lower_t, src=lower_bound_ref[0:1, 0:1])
    scale_bc, lower_bc = nl.broadcast_to(scale_t, (d, 1)), nl.broadcast_to(lower_t, (d, 1))

    # Heads in a group step through the serial chain in lockstep, so one head's
    # latency hides behind the others' work.
    group = max(n for n in range(1, GROUP_MAX + 1) if (h_end - h_start) % n == 0)

    def head_setup(h):
        dt_row = sbuf((1, d))
        nisa.dma_copy(dst=dt_row, src=dt_bias_ref[h:h + 1, 0:d])
        dt_col_p = psum((d, 1))
        nisa.nc_transpose(dst=dt_col_p, data=dt_row, engine=nisa.tensor_engine)
        dt_col = sbuf((d, 1))
        nisa.tensor_copy(dst=dt_col, src=dt_col_p)
        a_log = sbuf((1, 1))
        nisa.dma_copy(dst=a_log, src=a_log_ref[h:h + 1, 0:1])
        a_exp = sbuf((1, 1))
        nisa.activation(dst=a_exp, op=nl.exp, data=a_log)
        S = sbuf((d, d))
        nisa.dma_copy(dst=S, src=state_ref[h, 0:d, 0:d])
        # The count of non-finite tokens in earlier chunks, on every partition.
        seen = sbuf((C, 1))
        nisa.memset(dst=seen, value=0.0)
        return {"dt_col": dt_col, "a_exp": nl.broadcast_to(a_exp, (d, 1)), "seen": seen}, S

    def flags(qb, kb, vb, g_in, beta_in, valid_in):
        """Mark padding tokens; zero g_in's non-finite entries in place."""
        # x * 0 is 0 for a finite x and NaN otherwise.
        z = sbuf((d, C))
        nisa.tensor_scalar(dst=z, data=qb, op0=nl.multiply, operand0=0.0)
        for xb in (kb, vb):
            z_next = sbuf((d, C))
            nisa.scalar_tensor_tensor(dst=z_next, data=xb, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=z)
            z = z_next
        bad_T = sbuf((d, C))
        nisa.tensor_scalar(dst=bad_T, data=z, op0=nl.not_equal, operand0=0.0)
        # The count of non-finite q, k, v channels per token: a sum over partitions.
        bad_p = psum((C, 1))
        nisa.nc_matmul(dst=bad_p, stationary=bad_T, moving=ones_col)
        g_bad = sbuf((C, d), nl.uint8)
        nisa.tensor_scalar(dst=g_bad, data=g_in, op0=nl.multiply, operand0=0.0, op1=nl.not_equal, operand1=0.0)
        nisa.tensor_copy_predicated(dst=g_in, src=0.0, predicate=g_bad)
        g_any = sbuf((C, 1))
        nisa.tensor_reduce(dst=g_any, op=nl.maximum, data=g_bad, axis=1)
        tok = sbuf((C, 1))
        nisa.scalar_tensor_tensor(dst=tok, data=beta_in, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=bad_p)
        nonfin = sbuf((C, 1))
        nisa.tensor_scalar(dst=nonfin, data=tok, op0=nl.not_equal, operand0=0.0, op1=nl.maximum, operand1=g_any)
        pad = sbuf((C, 1))
        nisa.tensor_scalar(dst=pad, data=valid_in, op0=nl.less, operand0=0.5, op1=nl.maximum, operand1=nonfin)
        # The channel-major copy of the padding flag: a transpose of its row broadcast.
        pad_bc = sbuf((C, d))
        nisa.tensor_scalar(dst=pad_bc, data=zeros_cd, op0=nl.add, operand0=pad)
        pad_T_p = psum((d, C))
        nisa.nc_transpose(dst=pad_T_p, data=pad_bc, engine=nisa.tensor_engine)
        pad_T, pad_u8 = sbuf((d, C), nl.uint8), sbuf((C, 1), nl.uint8)
        nisa.tensor_copy(dst=pad_T, src=pad_T_p)
        nisa.tensor_copy(dst=pad_u8, src=pad)
        return pad_T, pad_u8, nonfin

    def poisoned(hc, nonfin):
        """Rows from the sequence's first non-finite token on, as a ``[C, D]`` predicate."""
        run_p = psum((C, 1))
        nisa.nc_matmul(dst=run_p, stationary=upto, moving=nonfin)
        rows = sbuf((C, 1))
        nisa.tensor_scalar(dst=rows, data=run_p, op0=nl.add, operand0=hc["seen"], op1=nl.greater, operand1=0.5)
        rows_bc = sbuf((C, d), nl.uint8)
        nisa.tensor_scalar(dst=rows_bc, data=zeros_cd, op0=nl.add, operand0=rows)
        total_p = psum((C, 1))
        nisa.nc_matmul(dst=total_p, stationary=ones_cc, moving=nonfin)
        seen = sbuf((C, 1))
        nisa.tensor_scalar(dst=seen, data=total_p, op0=nl.add, operand0=hc["seen"])
        hc["seen"] = seen
        return rows_bc

    def normed(xb, pad_T, scale):
        """x L2-normed over channels, times ``scale``, in fp32; zero for padding tokens."""
        sq = sbuf((d, C))
        nisa.tensor_tensor(dst=sq, data1=xb, data2=xb, op=nl.multiply)
        nisa.tensor_copy_predicated(dst=sq, src=0.0, predicate=pad_T)
        # ones.T @ x^2 puts each token's sum of squares on every partition.
        ss_p = psum((d, C))
        nisa.nc_matmul(dst=ss_p, stationary=ones_dd, moving=sq)
        rn = sbuf((d, C))
        nisa.activation(dst=rn, op=nl.rsqrt, data=ss_p, bias=L2_EPS)
        x_n = sbuf((d, C))
        if scale is None:
            nisa.tensor_tensor(dst=x_n, data1=xb, data2=rn, op=nl.multiply)
        else:
            nisa.scalar_tensor_tensor(dst=x_n, data=xb, op0=nl.multiply, operand0=scale, op1=nl.multiply, operand1=rn)
        nisa.tensor_copy_predicated(dst=x_n, src=0.0, predicate=pad_T)
        return x_n

    def transposed(x_T):
        x_p = psum((C, d))
        nisa.nc_transpose(dst=x_p, data=x_T, engine=nisa.tensor_engine)
        return x_p

    def preprocess(h, t0, hc):
        """Everything in a chunk that does not read S."""
        rows = slice(h * d, (h + 1) * d)
        qb, kb, vb = sbuf((d, C), nl.bfloat16), sbuf((d, C), nl.bfloat16), sbuf((d, C), nl.bfloat16)
        nisa.dma_copy(dst=qb, src=q_ref[rows, t0:t0 + C])
        nisa.dma_copy(dst=kb, src=k_ref[rows, t0:t0 + C])
        nisa.dma_copy(dst=vb, src=v_ref[rows, t0:t0 + C])
        g_in, beta_in, valid_in = sbuf((C, d)), sbuf((C, 1)), sbuf((C, 1))
        nisa.dma_copy(dst=g_in, src=g_ref[h, t0:t0 + C, 0:d])
        nisa.dma_copy(dst=beta_in, src=beta_ref[h, t0:t0 + C, 0:1])
        nisa.dma_copy(dst=valid_in, src=valid_ref[t0:t0 + C, 0:1])
        pad_T, pad_u8, nonfin = flags(qb, kb, vb, g_in, beta_in, valid_in)
        poison = poisoned(hc, nonfin)

        q_T = normed(qb, pad_T, scale_bc)
        k_T = normed(kb, pad_T, None)
        v_T = sbuf((d, C))
        nisa.tensor_copy(dst=v_T, src=vb)
        nisa.tensor_copy_predicated(dst=v_T, src=0.0, predicate=pad_T)
        beta = sbuf((C, 1))
        nisa.activation(dst=beta, op=nl.sigmoid, data=beta_in)
        nisa.tensor_copy_predicated(dst=beta, src=0.0, predicate=pad_u8)

        # gate = lower_bound * sigmoid(exp(a_log) * (g + dt_bias)), at most 0; exactly 0 for padding.
        g_T_p = psum((d, C))
        nisa.nc_transpose(dst=g_T_p, data=g_in, engine=nisa.tensor_engine)
        g_arg = sbuf((d, C))
        nisa.tensor_scalar(dst=g_arg, data=g_T_p, op0=nl.add, operand0=hc["dt_col"], op1=nl.multiply,
                           operand1=hc["a_exp"])
        g_sig = sbuf((d, C))
        nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_arg)
        gate = sbuf((d, C))
        nisa.tensor_scalar(dst=gate, data=g_sig, op0=nl.multiply, operand0=lower_bc)
        nisa.tensor_copy_predicated(dst=gate, src=0.0, predicate=pad_T)

        # Cumulative gate within each 16-token sub-chunk: result[i] = 1 * result[i-1] + gate[i].
        gcum = sbuf((d, C))
        for b in range(NSUB):
            o = b * SUB
            nisa.tensor_tensor_scan(dst=gcum[0:d, o:o + SUB], data0=ones_scan, data1=gate[0:d, o:o + SUB],
                                    initial=zero_init, op0=nl.multiply, op1=nl.add)
        exp_gcum, exp_neg = sbuf((d, C)), sbuf((d, C))
        nisa.activation(dst=exp_gcum, op=nl.exp, data=gcum)
        nisa.activation(dst=exp_neg, op=nl.exp, data=gcum, scale=-1.0)
        k_dec_T, k_inv_T, q_dec_T = sbuf((d, C)), sbuf((d, C)), sbuf((d, C))
        nisa.tensor_tensor(dst=k_dec_T, data1=k_T, data2=exp_gcum, op=nl.multiply)
        nisa.tensor_tensor(dst=k_inv_T, data1=k_T, data2=exp_neg, op=nl.multiply)
        nisa.tensor_tensor(dst=q_dec_T, data1=q_T, data2=exp_gcum, op=nl.multiply)

        # k_restored = k * exp(g_total - gcum), with a non-positive exponent; g_total is
        # each band's last gcum column, read SUB times through a stride-0 pattern.
        gtot = sbuf((d, C))
        nisa.tensor_copy(dst=gtot.ap(pattern=[[C, d], [SUB, NSUB], [1, SUB]]),
                         src=gcum.ap(pattern=[[C, d], [SUB, NSUB], [0, SUB]], offset=SUB - 1))
        gdiff = sbuf((d, C))
        nisa.tensor_tensor(dst=gdiff, data1=gcum, data2=gtot, op=nl.subtract)
        decay = sbuf((d, C))
        nisa.activation(dst=decay, op=nl.exp, data=gdiff, scale=-1.0)
        k_rest_T = sbuf((d, C))
        nisa.tensor_tensor(dst=k_rest_T, data1=k_T, data2=decay, op=nl.multiply)

        # L = beta * (k_dec @ k_inv.T) and Mqk = q_dec @ k_inv.T, each kept on its block-diagonal by select.
        L_p = psum((C, C))
        nisa.nc_matmul(dst=L_p, stationary=k_dec_T, moving=k_inv_T)
        L = sbuf((C, C))
        nisa.tensor_scalar(dst=L, data=L_p, op0=nl.multiply, operand0=beta)
        nisa.tensor_copy_predicated(dst=L, src=0.0, predicate=off_strict)
        M_p = psum((C, C))
        nisa.nc_matmul(dst=M_p, stationary=q_dec_T, moving=k_inv_T)
        M = sbuf((C, C))
        nisa.tensor_copy(dst=M, src=M_p)
        nisa.tensor_copy_predicated(dst=M, src=0.0, predicate=off_incl)
        M_T_p = psum((C, C))
        nisa.nc_transpose(dst=M_T_p, data=M, engine=nisa.tensor_engine)
        M_T = sbuf((C, C))
        nisa.tensor_copy(dst=M_T, src=M_T_p, engine=nisa.scalar_engine)

        # INV_T = ((I + L)^-1).T = ((I - L)(I + L^2)(I + L^4)(I + L^8)).T, exact: L^16 = 0 in each block.
        L_T_p = psum((C, C))
        nisa.nc_transpose(dst=L_T_p, data=L, engine=nisa.tensor_engine)
        L_T = sbuf((C, C))
        nisa.tensor_copy(dst=L_T, src=L_T_p, engine=nisa.scalar_engine)
        INV_T = sbuf((C, C))
        nisa.tensor_tensor(dst=INV_T, data1=eye, data2=L_T, op=nl.subtract)
        Lp_p, Lp_T_p = psum((C, C)), psum((C, C))
        nisa.nc_matmul(dst=Lp_p, stationary=L_T, moving=L)
        nisa.nc_matmul(dst=Lp_T_p, stationary=L, moving=L_T)
        Lp, Lp_T = sbuf((C, C)), sbuf((C, C))
        nisa.tensor_copy(dst=Lp, src=Lp_p, engine=nisa.scalar_engine)
        nisa.tensor_copy(dst=Lp_T, src=Lp_T_p, engine=nisa.scalar_engine)
        for step in range(NEUMANN - 1):
            ipl = sbuf((C, C))
            nisa.tensor_tensor(dst=ipl, data1=eye, data2=Lp, op=nl.add)
            inv_p = psum((C, C))
            nisa.nc_matmul(dst=inv_p, stationary=ipl, moving=INV_T)
            nisa.tensor_copy(dst=INV_T, src=inv_p, engine=nisa.scalar_engine)
            if step < NEUMANN - 2:
                lpt_p, lp_p = psum((C, C)), psum((C, C))
                nisa.nc_matmul(dst=lpt_p, stationary=Lp, moving=Lp_T)
                nisa.nc_matmul(dst=lp_p, stationary=Lp_T, moving=Lp)
                nisa.tensor_copy(dst=Lp, src=lp_p, engine=nisa.scalar_engine)
                nisa.tensor_copy(dst=Lp_T, src=lpt_p, engine=nisa.scalar_engine)

        # WY factors: u = INV @ (beta v), w = INV @ (beta k_dec); w comes out channel-major.
        vbeta = sbuf((C, d))
        nisa.tensor_scalar(dst=vbeta, data=transposed(v_T), op0=nl.multiply, operand0=beta)
        kbeta = sbuf((C, d))
        nisa.tensor_scalar(dst=kbeta, data=transposed(k_dec_T), op0=nl.multiply, operand0=beta)
        w_T_p = psum((d, C))
        nisa.nc_matmul(dst=w_T_p, stationary=kbeta, moving=INV_T)
        w_T = sbuf((d, C))
        nisa.tensor_copy(dst=w_T, src=w_T_p)

        # Sub-chunk-major staging of k_restored and u for the state update, four bands per PSUM tile.
        half = NSUB // 2
        kr3, u3 = sbuf((SUB, NSUB, d)), sbuf((SUB, NSUB, d))
        for part in range(2):
            kr3_p = nl.ndarray((SUB, half, d), dtype=f32, buffer=nl.psum)
            u3_p = nl.ndarray((SUB, half, d), dtype=f32, buffer=nl.psum)
            for bb in range(half):
                o = (part * half + bb) * SUB
                nisa.nc_transpose(dst=kr3_p[0:SUB, bb, 0:d], data=k_rest_T[0:d, o:o + SUB],
                                  engine=nisa.tensor_engine)
                nisa.nc_matmul(dst=u3_p[0:SUB, bb, 0:d], stationary=INV_T[0:C, o:o + SUB], moving=vbeta)
            nisa.tensor_copy(dst=kr3[0:SUB, part * half:(part + 1) * half, 0:d], src=kr3_p)
            nisa.tensor_copy(dst=u3[0:SUB, part * half:(part + 1) * half, 0:d], src=u3_p)
        return {"q_dec_T": q_dec_T, "exp_gcum": exp_gcum, "w_T": w_T, "u3": u3, "kr3": kr3, "M_T": M_T,
                "poison": poison, "U3": sbuf((SUB, NSUB, d)), "U_T": sbuf((d, C)), "oi_T": sbuf((d, C))}

    def band_step(P, S, b):
        """One serial step: U_b = u_b - w_b @ S; out_inter_b = q_dec_b @ S; S = S exp(g_total_b) + k_rest_b.T @ U_b."""
        o = b * SUB
        wS_p = nl.ndarray((SUB, d), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=wS_p, stationary=P["w_T"][0:d, o:o + SUB], moving=S)
        nisa.tensor_tensor(dst=P["U3"][0:SUB, b, 0:d], data1=P["u3"][0:SUB, b, 0:d], data2=wS_p, op=nl.subtract)
        U_T_p = nl.ndarray((d, SUB), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(dst=U_T_p, data=P["U3"][0:SUB, b, 0:d], engine=nisa.tensor_engine)
        nisa.tensor_copy(dst=P["U_T"][0:d, o:o + SUB], src=U_T_p)
        oi_p = nl.ndarray((d, SUB), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=oi_p, stationary=S, moving=P["q_dec_T"][0:d, o:o + SUB])
        nisa.tensor_copy(dst=P["oi_T"][0:d, o:o + SUB], src=oi_p)
        dS_p = psum((d, d))
        nisa.nc_matmul(dst=dS_p, stationary=P["kr3"][0:SUB, b, 0:d], moving=P["U3"][0:SUB, b, 0:d])
        S_new = sbuf((d, d))
        nisa.scalar_tensor_tensor(dst=S_new, data=S, op0=nl.multiply, operand0=P["exp_gcum"][0:d, o + SUB - 1:o + SUB],
                                  op1=nl.add, operand1=dS_p)
        return S_new

    def epilogue(h, t0, P):
        """out = out_inter + Mqk @ U for the chunk, NaN from a non-finite token on."""
        U = sbuf((C, d))
        nisa.tensor_copy(dst=U, src=transposed(P["U_T"]))
        intra_p = psum((C, d))
        nisa.nc_matmul(dst=intra_p, stationary=P["M_T"], moving=U)
        inter = sbuf((C, d))
        nisa.tensor_copy(dst=inter, src=transposed(P["oi_T"]))
        out_c = sbuf((C, d))
        nisa.tensor_tensor(dst=out_c, data1=inter, data2=intra_p, op=nl.add)
        nisa.tensor_copy_predicated(dst=out_c, src=float("nan"), predicate=P["poison"])
        nisa.dma_copy(dst=out[h, t0:t0 + C, 0:d], src=out_c)

    for hg in range(0, h_end - h_start, group):
        heads = [h_start + hg + i for i in range(group)]
        setups = [head_setup(h) for h in heads]
        hc, S = [s[0] for s in setups], [s[1] for s in setups]
        for c in range(T // C):
            t0 = c * C
            P = [preprocess(heads[i], t0, hc[i]) for i in range(group)]
            # Bands interleave across the group's heads.
            for b in range(NSUB):
                for i in range(group):
                    S[i] = band_step(P[i], S[i], b)
            for i in range(group):
                epilogue(heads[i], t0, P[i])
        for i in range(group):
            # A sequence with a non-finite token ends in a NaN state.
            lost = sbuf((d, d), nl.uint8)
            nisa.tensor_scalar(dst=lost, data=zeros_cd[0:d, 0:d], op0=nl.add, operand0=hc[i]["seen"][0:d, 0:1],
                               op1=nl.greater, operand1=0.5)
            nisa.tensor_copy_predicated(dst=S[i], src=float("nan"), predicate=lost)
            nisa.dma_copy(dst=state_ref[heads[i], 0:d, 0:d], src=S[i])
    return out, state_ref
