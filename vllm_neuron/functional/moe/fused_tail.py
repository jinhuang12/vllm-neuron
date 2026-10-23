# SPDX-License-Identifier: Apache-2.0
"""Fused MoE tail at decode: shared-expert SiTU and down projection, latent up
projection and the latent sum of squares, written as one fp32 pack per token.

A pack row is ``[up (H) | shared (H) | sum of squares (1)]``, each a partial over this
rank's K shard. The tail all-reduce sums the packs across ranks, and the consumer
applies the latent norm scale ``rsqrt(ss / D + eps)`` to the up part before adding the
shared part. The latent up weights arrive with the latent norm weight folded in.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config

_COLUMN_TILE = 512


def moe_fused_tail(
    combined: torch.Tensor,
    shared_gate_up: torch.Tensor,
    latent_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    """Return the fp32 tail pack ``[T, 2H + 1]`` of one MoE block at decode.

    Args:
        combined: ``[T, K]`` fp32 combined latent rows of this rank's K shard.
        shared_gate_up: ``[T, 2I]`` bf16 shared-expert gate (first I) and up outputs.
        latent_up_weight: ``[K, H]`` bf16 latent up weights, norm weight folded in.
        shared_down_weight: ``[I, H]`` bf16 shared-expert down weights.
        beta: SiTU gate beta.
        linear_beta: SiTU linear beta of the up half.

    Returns:
        ``[T, 2H + 1]`` fp32: the up partial, the shared partial and the sum of
        squares of ``combined``.
    """
    tokens, latent = combined.shape
    hidden = latent_up_weight.shape[1]
    shared = shared_down_weight.shape[0]
    assert combined.dtype == torch.float32, f"combined must be fp32, got {combined.dtype}"
    assert shared_gate_up.dtype == torch.bfloat16, (
        f"shared_gate_up must be bf16, got {shared_gate_up.dtype}"
    )
    assert latent_up_weight.dtype == shared_down_weight.dtype == torch.bfloat16, (
        "the tail weights must be bf16"
    )
    assert tuple(shared_gate_up.shape) == (tokens, 2 * shared), (
        f"shared_gate_up {tuple(shared_gate_up.shape)} must be [{tokens}, {2 * shared}]"
    )
    assert tuple(latent_up_weight.shape) == (latent, hidden), (
        f"latent_up_weight {tuple(latent_up_weight.shape)} must be [{latent}, {hidden}]"
    )
    assert shared_down_weight.shape[1] == hidden, (
        f"shared_down_weight {tuple(shared_down_weight.shape)} must have {hidden} columns"
    )
    return wrap_nki(_moe_fused_tail_kernel)[get_logical_nc_config()](
        combined=combined,
        shared_gate_up=shared_gate_up,
        latent_up_weight=latent_up_weight,
        shared_down_weight=shared_down_weight,
        beta=beta,
        linear_beta=linear_beta,
    )


@nki.jit
def _moe_fused_tail_kernel(
    combined, shared_gate_up, latent_up_weight, shared_down_weight, beta, linear_beta
):
    """Each core writes the pack columns of its half of H; core 0 writes the sum of squares."""
    T, KL = combined.shape
    _, H = latent_up_weight.shape
    I = shared_down_weight.shape[0]
    n_prog = nl.num_programs(axes=0)
    prog = nl.program_id(axis=0)
    HC = H // n_prog
    h0 = prog * HC
    kernel_assert(T <= nl.tile_size.pmax, f"Expected at most 128 tokens, got {T}")
    kernel_assert(KL <= nl.tile_size.pmax, f"Expected a K shard of at most 128, got {KL}")
    kernel_assert(
        HC % _COLUMN_TILE == 0,
        f"Expected H divisible by {_COLUMN_TILE * n_prog}, got {H}",
    )
    kernel_assert(beta > 0 and linear_beta > 0, "Expected positive SiTU betas")

    pack = nl.ndarray((T, 2 * H + 1), dtype=nl.float32, buffer=nl.shared_hbm)
    k_tiles = [(k0, min(nl.tile_size.pmax, I - k0)) for k0 in range(0, I, nl.tile_size.pmax)]

    # Each weight tile of this core's half moves in one DMA, issued before the activations.
    up_w = nl.ndarray((KL, HC), dtype=latent_up_weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=up_w, src=latent_up_weight[0:KL, h0:h0 + HC])
    down_w = []
    for k0, kn in k_tiles:
        w = nl.ndarray((kn, HC), dtype=shared_down_weight.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=shared_down_weight[k0:k0 + kn, h0:h0 + HC])
        down_w.append(w)

    rows = nl.ndarray((T, KL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=combined)
    gate_up = nl.ndarray((T, 2 * I), dtype=shared_gate_up.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gate_up, src=shared_gate_up)

    if prog == 0:
        squares = nl.ndarray((T, KL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=squares, op=nl.square, data=rows)
        ss = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=squares, axis=(1,))
        nisa.dma_copy(dst=pack[0:T, 2 * H:2 * H + 1], src=ss)

    # SiTU in fp32: tanh(g / beta) * sigmoid(g) * tanh(u / linear_beta); the two betas
    # scale the product on its cast to bf16 below.
    situ = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.sbuf)
    sig = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.sbuf)
    lin = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=situ, op=nl.tanh, data=gate_up[0:T, 0:I], scale=1.0 / beta)
    nisa.activation(dst=sig, op=nl.sigmoid, data=gate_up[0:T, 0:I])
    nisa.activation(dst=lin, op=nl.tanh, data=gate_up[0:T, I:2 * I], scale=1.0 / linear_beta)
    nisa.tensor_tensor(dst=situ, data1=situ, data2=sig, op=nl.multiply)
    nisa.tensor_tensor(dst=lin, data1=lin, data2=situ, op=nl.multiply)

    # PE transposes put K on the partitions; activations stay stationary in bf16.
    rows_t_ps = nl.ndarray((KL, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=rows_t_ps, data=rows)
    rows_t = nl.ndarray((KL, T), dtype=latent_up_weight.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_t, src=rows_t_ps)
    situ_t = []
    for k0, kn in k_tiles:
        t_ps = nl.ndarray((kn, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=t_ps, data=lin[0:T, k0:k0 + kn])
        t_sb = nl.ndarray((kn, T), dtype=shared_down_weight.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t_sb, data=t_ps, op0=nl.multiply, operand0=beta * linear_beta)
        situ_t.append(t_sb)

    up_out = nl.ndarray((T, HC), dtype=nl.float32, buffer=nl.sbuf)
    shared_out = nl.ndarray((T, HC), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(HC // _COLUMN_TILE):
        cols = slice(j * _COLUMN_TILE, (j + 1) * _COLUMN_TILE)
        up_ps = nl.ndarray((T, _COLUMN_TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=up_ps, stationary=rows_t, moving=up_w[0:KL, cols])
        nisa.tensor_copy(dst=up_out[0:T, cols], src=up_ps, engine=nisa.scalar_engine)
        sh_ps = nl.ndarray((T, _COLUMN_TILE), dtype=nl.float32, buffer=nl.psum)
        for i in range(len(k_tiles)):
            nisa.nc_matmul(
                dst=sh_ps,
                stationary=situ_t[i],
                moving=down_w[i][0:k_tiles[i][1], cols],
                accumulate=(i > 0),
            )
        nisa.tensor_copy(dst=shared_out[0:T, cols], src=sh_ps, engine=nisa.vector_engine)

    nisa.dma_copy(dst=pack[0:T, h0:h0 + HC], src=up_out)
    nisa.dma_copy(dst=pack[0:T, H + h0:H + h0 + HC], src=shared_out)
    return pack
